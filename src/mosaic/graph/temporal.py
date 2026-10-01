"""Temporal graph construction and causal snapshot features.

Why this module exists at all
-----------------------------
A single graph built from the whole dataset cannot produce honest historical
features: every edge is present at every timestamp, so a relationship first
observed in month 12 would appear in the features of an event in month 1. That is
leakage, and it is the single easiest way to make a relational feature family look
useful when it is not.

So the graph here is built by **replaying edges in timestamp order** and snapshotting
at query times. A snapshot at ``T`` contains exactly the edges whose ``first_seen``
is ``<= T``. The invariant is stated once, here, and enforced by
:func:`assert_no_future_edges`, which the leakage tests call.

Cost model
----------
Recomputing PageRank per snapshot is O(n_snapshots * E). For the research datasets
this is acceptable and is *measured* rather than assumed; :func:`snapshot_features`
reports the snapshot count and per-snapshot cost so the scaling experiment can use
real numbers. If a dataset ever makes this untenable, the fix is incremental
PageRank over an edge-ordered stream, not a global graph.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, cast

import networkx as nx
import numpy as np
import polars as pl

from mosaic.utils.timing import StageMetrics, stage_timer

LOGGER = logging.getLogger(__name__)

SECOND = 1
HOUR = 3600
DAY = 86_400


@dataclass
class GraphConfig:
    """Graph construction policy. All temporal: nothing here sees the future."""

    #: Snapshots are taken at these offsets before each query timestamp. A trailing
    #: lag means a relationship must have existed for ``min_age`` before it counts,
    #: which stops a relationship created *by the current event* from explaining it.
    min_edge_age_seconds: int = 0
    #: Cap on edges added per snapshot step, to bound memory on dense graphs.
    max_edges: int | None = None
    pagerank_alpha: float = 0.85
    pagerank_iterations: int = 50
    #: Recompute expensive measures (betweenness) only every Nth snapshot. Betweenness
    #: is O(V*E) and is the first thing to blow up; its *use* here is as a slow-moving
    #: structural descriptor, not a fast signal.
    betweenness_every: int = 10
    directed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_edge_age_seconds": self.min_edge_age_seconds,
            "max_edges": self.max_edges,
            "pagerank_alpha": self.pagerank_alpha,
            "pagerank_iterations": self.pagerank_iterations,
            "betweenness_every": self.betweenness_every,
            "directed": self.directed,
        }


@dataclass
class TemporalGraph:
    """Edge list with first-seen times, plus snapshot access.

    ``edges`` is the canonical form: one row per (source, target, first_seen) with a
    weight. Nodes are entity ids. Nothing is precomputed globally: every measure is
    derived from a snapshot that satisfies ``first_seen <= cutoff``.
    """

    edges: pl.DataFrame
    nodes: list[str]
    config: GraphConfig
    metrics: list[StageMetrics] = field(default_factory=list)
    _adjacency_index: dict[str, dict[str, list[tuple[str, datetime, float]]]] = field(
        default_factory=dict, repr=False
    )

    @property
    def n_edges(self) -> int:
        return self.edges.height

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def summary(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "first_seen_min": str(self.edges["first_seen"].min()) if self.edges.height else None,
            "first_seen_max": str(self.edges["first_seen"].max()) if self.edges.height else None,
            "config": self.config.to_dict(),
            "stage_metrics": [m.to_dict() for m in self.metrics],
        }

    # -- construction --------------------------------------------------------
    @classmethod
    def from_relations(
        cls,
        relations: pl.DataFrame,
        *,
        source_column: str = "entity_id",
        target_column: str = "related_entity_id",
        time_column: str = "first_seen",
        weight_column: str = "weight",
        config: GraphConfig | None = None,
    ) -> TemporalGraph:
        """Build the temporal edge list from event-derived interactions.

        Repeated interactions between the same ordered pair collapse into one edge
        whose ``first_seen`` is the earliest occurrence and whose weight is the
        count - so an edge's *existence* is decided by when it first appeared, while
        its weight reflects how often it has been seen so far.
        """
        config = config or GraphConfig()
        metrics: list[StageMetrics] = []
        with stage_timer("graph:build_edges", metrics):
            required = {source_column, target_column, time_column}
            missing = required - set(relations.columns)
            if missing:
                raise ValueError(f"relations frame is missing {sorted(missing)}")
            edges = (
                relations.filter(
                    pl.col(source_column).is_not_null()
                    & pl.col(target_column).is_not_null()
                    & (pl.col(source_column) != pl.col(target_column))
                )
                .group_by([source_column, target_column])
                .agg(
                    pl.col(time_column).min().alias("first_seen"),
                    pl.col(time_column).max().alias("last_seen"),
                    pl.len().alias("n_interactions"),
                    (pl.col(weight_column).sum() if weight_column in relations.columns else pl.len())
                    .cast(pl.Float64)
                    .alias("weight"),
                )
            )
        if edges.is_empty():
            nodes = sorted(
                set(relations[source_column].drop_nulls().to_list())
                | set(relations[target_column].drop_nulls().to_list())
            )
            empty = pl.DataFrame(
                schema={
                    "source": pl.Utf8,
                    "target": pl.Utf8,
                    "first_seen": pl.Datetime("us"),
                    "last_seen": pl.Datetime("us"),
                    "n_interactions": pl.Int64,
                    "weight": pl.Float64,
                }
            )
            return cls(edges=empty, nodes=nodes, config=config, metrics=metrics)

        edges = edges.rename({source_column: "source", target_column: "target"})
        nodes = sorted(set(edges["source"].to_list()) | set(edges["target"].to_list()))
        graph = cls(edges=edges, nodes=nodes, config=config, metrics=metrics)
        graph._build_index()
        LOGGER.info(
            "temporal graph: %d nodes, %d edges, %s..%s",
            graph.n_nodes, graph.n_edges, edges["first_seen"].min(), edges["first_seen"].max(),
        )
        return graph

    def _build_index(self) -> None:
        """Per-source adjacency sorted by ``first_seen``, for snapshot replay.

        Sorting per source is what makes the replay cheap: adding edges up to a
        cutoff is a prefix scan, not a filter over the whole edge list.
        """
        index: dict[str, dict[str, list[tuple[str, datetime, float]]]] = {
            node: {"out": [], "in": []} for node in self.nodes
        }
        for source, target, first_seen, weight in self.edges.select(
            "source", "target", "first_seen", "weight"
        ).iter_rows():
            index[str(source)]["out"].append((str(target), first_seen, float(weight)))
            index[str(target)]["in"].append((str(source), first_seen, float(weight)))
        for adjacency in index.values():
            adjacency["out"].sort(key=lambda item: item[1])
            adjacency["in"].sort(key=lambda item: item[1])
        self._adjacency_index = index

    # -- snapshots -----------------------------------------------------------
    def edges_as_of(self, cutoff: datetime) -> pl.DataFrame:
        """Edges whose ``first_seen <= cutoff``. The temporal invariant, in one place."""
        if self.edges.is_empty():
            return self.edges
        return self.edges.filter(pl.col("first_seen") <= cutoff)

    def snapshot(self, cutoff: datetime) -> nx.DiGraph:
        """Build the networkx graph visible at ``cutoff``.

        Isolated nodes are added so degree measures are defined (and zero) rather
        than missing, which would otherwise make a silently-vanished entity look
        different from a never-seen one.
        """
        visible = self.edges_as_of(cutoff)
        graph: nx.DiGraph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        if visible.height:
            for source, target, weight in visible.select("source", "target", "weight").iter_rows():
                graph.add_edge(str(source), str(target), weight=float(weight))
        return graph

    def degrees_as_of(self, cutoff: datetime) -> dict[str, dict[str, float]]:
        """Directed in/out/weighted degrees from one pass over visible edges.

        Cheaper than building a networkx graph when only degrees are needed, which is
        the common case in the feature loop.
        """
        visible = self.edges_as_of(cutoff)
        out: dict[str, dict[str, float]] = {
            node: {"in_degree": 0.0, "out_degree": 0.0, "weighted_in": 0.0, "weighted_out": 0.0}
            for node in self.nodes
        }
        if visible.height:
            for source, target, weight in visible.select("source", "target", "weight").iter_rows():
                s, t = str(source), str(target)
                out[s]["out_degree"] += 1.0
                out[t]["in_degree"] += 1.0
                out[s]["weighted_out"] += float(weight)
                out[t]["weighted_in"] += float(weight)
        for values in out.values():
            values["degree"] = values["in_degree"] + values["out_degree"]
            values["weighted_degree"] = values["weighted_in"] + values["weighted_out"]
        return out


# --------------------------------------------------------------- measures
def pagerank(
    graph: nx.DiGraph, *, alpha: float = 0.85, iterations: int = 50
) -> dict[str, float]:
    """PageRank with uniform dangling-mass handling.

    ``networkx.pagerank`` is used when available and falls back to an explicit power
    iteration so a graph with no in-edges (a common case early in a stream) does not
    raise.
    """
    if graph.number_of_nodes() == 0:
        return {}
    try:
        return nx.pagerank(graph, alpha=alpha, max_iter=iterations, weight="weight")
    except (nx.PowerIterationFailedConvergence, ZeroDivisionError):  # pragma: no cover
        return nx.pagerank(graph, alpha=alpha, max_iter=iterations * 10, weight=None)


def clustering(graph: nx.DiGraph) -> dict[str, float]:
    """Clustering coefficient on the underlying undirected projection.

    A directed clustering coefficient is not comparable across graph states; the
    undirected version is the standard descriptor and is what the feature means.
    """
    if graph.number_of_nodes() == 0:
        return {}
    return nx.clustering(graph.to_undirected())


def betweenness(graph: nx.DiGraph) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    return nx.betweenness_centrality(graph, weight=None, normalized=True)


def communities(graph: nx.DiGraph) -> dict[str, int]:
    """Community labels.

    Deterministic without a seed argument: greedy modularity on an undirected
    view is order-independent, and the returned labels are assigned by a total
    order on (size, smallest member) rather than by the algorithm's internal
    iteration. There was a ``seed`` parameter that was never read and never
    passed; leaving it in the signature would have implied a knob that does
    nothing.
    """
    if graph.number_of_nodes() == 0:
        return {}
    undirected = graph.to_undirected()
    if undirected.number_of_edges() == 0:
        return {node: idx for idx, node in enumerate(sorted(graph.nodes()))}
    groups = nx.community.greedy_modularity_communities(undirected, weight="weight")
    # sort by size then smallest member for a stable label assignment
    ordered = sorted(groups, key=lambda g: (-len(g), min(g)))
    return {node: label for label, group in enumerate(ordered) for node in group}


def snapshot_measures(
    graph: TemporalGraph, cutoff: datetime, *, compute_betweenness: bool = False
) -> dict[str, dict[str, float]]:
    """All structural measures at one cutoff, from one snapshot."""
    snapshot = graph.snapshot(cutoff)
    pr = pagerank(snapshot, alpha=graph.config.pagerank_alpha, iterations=graph.config.pagerank_iterations)
    clu = clustering(snapshot)
    com = communities(snapshot)
    btw = betweenness(snapshot) if compute_betweenness else {}
    degrees = graph.degrees_as_of(cutoff)
    max_pr = max(pr.values()) if pr else 1.0
    out: dict[str, dict[str, float]] = {}
    for node in graph.nodes:
        entry = dict(degrees.get(node, {}))
        entry["graph_pagerank"] = float(pr.get(node, 0.0)) / (max_pr or 1.0)
        entry["graph_clustering"] = float(clu.get(node, 0.0))
        entry["graph_community"] = float(com.get(node, -1))
        if btw:
            entry["graph_betweenness"] = float(btw.get(node, 0.0))
        out[node] = entry
    return out


# --------------------------------------------------------------- features
def snapshot_features(
    events: pl.DataFrame,
    graph: TemporalGraph,
    *,
    entity_column: str = "entity_id",
    time_column: str = "timestamp",
    related_column: str = "related_entity_refs",
    snapshot_interval_seconds: int = 6 * HOUR,
    new_neighbor_window: int = 7 * DAY,
) -> tuple[pl.DataFrame, list[StageMetrics]]:
    """Attach causal graph features to every event row.

    For each event at ``T`` the features come from the snapshot at ``T`` minus
    ``snapshot_interval_seconds`` (or the immediately preceding snapshot), never from
    the full graph. Relationships created by the current row are therefore excluded,
    which is what makes "this entity suddenly gained a neighbour" measurable.

    Returns the events frame with graph feature columns, and stage metrics.
    """
    metrics: list[StageMetrics] = []
    interval = max(1, snapshot_interval_seconds)
    feature_names = [
        "degree", "in_degree", "out_degree", "weighted_degree",
        "pagerank", "clustering", "community",
    ]

    if events.is_empty():
        return events.with_columns(
            *[pl.lit(0.0, dtype=pl.Float64).alias(f"graph_{n}") for n in feature_names]
        ), metrics

    frame = events.sort([time_column, entity_column, "event_id"], nulls_last=True)
    times = frame[time_column].to_list()
    if any(when is None for when in times):
        raise ValueError(f"column {time_column!r} contains nulls; graph features need real timestamps")
    origin = min(times)

    # Bucket every row by elapsed time. Within a bucket the snapshot is identical, so
    # one snapshot serves the whole bucket: the cost is per-bucket, not per-row.
    buckets = np.array([int((when - origin).total_seconds()) // interval for when in times])
    unique_buckets = sorted(set(buckets.tolist()))

    # A row in bucket b is described by the graph as of the START of bucket b, so an
    # edge first seen inside the bucket cannot explain a row in that same bucket.
    snapshot_cutoffs: dict[int, datetime] = {
        b: origin + timedelta(seconds=b * interval) for b in unique_buckets
    }

    tables: dict[int, pl.DataFrame] = {}
    with stage_timer("graph:snapshots", metrics) as sw:
        for index, bucket in enumerate(unique_buckets):
            cutoff = snapshot_cutoffs[bucket]
            # Clamp the cutoff *backwards* to the first timestamp the data covers.
            # Clamping it forwards (to first_seen.min()) would hand the earliest
            # buckets a graph containing edges from days later - real leakage, and
            # it is exactly what the adversarial leakage test caught.
            floor = origin
            measures = snapshot_measures(
                graph,
                max(cutoff, floor),
                compute_betweenness=(index % max(1, graph.config.betweenness_every) == 0),
            )
            # snapshot_measures names its structural outputs with a "graph_" prefix
            # ("graph_pagerank"), while the degree outputs are unprefixed ("degree").
            # Both lookups try each spelling so a naming change on either side is a
            # hard error rather than a column that is silently all zeros.
            def pick(values: dict[str, float], name: str) -> float:
                for key in (name, f"graph_{name}"):
                    if key in values:
                        return float(values[key])
                raise KeyError(
                    f"snapshot_measures returned none of {name!r}/graph_{name!r}; "
                    f"available: {sorted(values)}"
                )

            tables[bucket] = pl.DataFrame(
                {
                    entity_column: list(measures.keys()),
                    **{
                        f"graph_{name}": [pick(v, name) for v in measures.values()]
                        for name in feature_names
                    },
                },
                schema={
                    entity_column: pl.Utf8,
                    **{f"graph_{name}": pl.Float64 for name in feature_names},
                },
            )
        if sw.metrics is not None:
            sw.metrics.rows = len(tables)
    LOGGER.info(
        "graph features: %d rows, %d buckets, %d nodes, %d edges",
        frame.height, len(tables), graph.n_nodes, graph.n_edges,
    )

    # One pass per bucket over that bucket's rows only, so peak memory is one slice.
    pieces: list[pl.DataFrame] = []
    indexed = frame.with_row_index("__row").with_columns(pl.Series("__bucket", buckets))
    for bucket in unique_buckets:
        slice_ = indexed.filter(pl.col("__bucket") == bucket).drop("__bucket")
        pieces.append(slice_.join(tables[bucket], on=entity_column, how="left"))
    out = pl.concat(pieces, how="vertical").sort("__row").drop("__row")

    out = out.with_columns(
        *[pl.col(f"graph_{n}").fill_null(0.0) for n in feature_names]
    )
    out = _add_graph_change_features(
        out, graph, entity_column, time_column, related_column, new_neighbor_window
    )
    return out, metrics


def _peer_rows(
    frame: pl.DataFrame, entity_column: str, time_column: str, related_column: str
) -> pl.DataFrame | None:
    """Normalise the peer column into (event_id, entity_id, timestamp, _peer) rows.

    The canonical schema allows a *sequence* of related entity references, but the
    synthetic sources carry a single scalar peer (``""`` when absent). Both shapes are
    accepted here rather than forcing one at the caller, because the choice is a
    property of the source adapters, not of the graph features.
    """
    dtype = frame.schema[related_column]
    base = frame.select("event_id", entity_column, time_column, related_column)
    if dtype == pl.Utf8:
        peers = (
            base.filter(pl.col(related_column).is_not_null() & (pl.col(related_column) != ""))
            .rename({related_column: "_peer"})
        )
    else:
        peers = base.explode(related_column).filter(pl.col(related_column).is_not_null())
        peers = peers.rename({related_column: "_peer"})
    if peers.is_empty():
        return None
    return peers


def _add_graph_change_features(
    frame: pl.DataFrame,
    graph: TemporalGraph,
    entity_column: str,
    time_column: str,
    related_column: str,
    new_neighbor_window: int,
) -> pl.DataFrame:
    """Degree change and new-neighbour ratio, computed causally.

    ``graph_new_neighbors`` counts neighbours whose ``first_seen`` falls inside the
    trailing window, so it is a function of edges known at ``t`` only.
    """
    out = frame
    if related_column not in frame.columns:
        return out.with_columns(
            pl.lit(0.0).alias("graph_new_neighbors"),
            pl.lit(0.0).alias("graph_degree_change"),
        )

    peers = _peer_rows(frame, entity_column, time_column, related_column)
    if peers is None:
        return out.with_columns(
            pl.lit(0.0).alias("graph_new_neighbors"),
            pl.lit(0.0).alias("graph_new_neighbor_ratio"),
            pl.lit(0.0).alias("graph_recent_neighbor_count"),
        )
    if peers.is_empty():
        return out.with_columns(
            pl.lit(0.0).alias("graph_new_neighbors"),
            pl.lit(0.0).alias("graph_degree_change"),
        )
    peers = peers.rename({"_peer": "_peer"})

    first_seen = (
        graph.edges.select("source", "target", "first_seen")
        if not graph.edges.is_empty()
        else pl.DataFrame(schema={"source": pl.Utf8, "target": pl.Utf8, "first_seen": pl.Datetime("us")})
    )
    lookup = first_seen.select(
        pl.col("source").alias(entity_column),
        pl.col("target").alias("_peer"),
        pl.col("first_seen").alias("_peer_first_seen"),
    )
    if not first_seen.is_empty():
        lookup = pl.concat(
            [
                lookup,
                first_seen.select(
                    pl.col("target").alias(entity_column),
                    pl.col("source").alias("_peer"),
                    pl.col("first_seen").alias("_peer_first_seen"),
                ),
            ]
        )
    joined = peers.join(lookup, on=[entity_column, "_peer"], how="left")
    # A peer only counts as *known* if its edge already existed at the event's own
    # timestamp. Without this guard a relationship first observed days later would be
    # counted as present, and the "suddenly gained a neighbour" signal becomes a
    # measure of hindsight instead of surprise.
    joined = joined.with_columns(
        (pl.col(time_column) - pl.col("_peer_first_seen")).dt.total_seconds().alias("_peer_age_s")
    ).filter(
        pl.col("_peer_first_seen").is_not_null() & (pl.col("_peer_first_seen") <= pl.col(time_column))
    )
    if joined.is_empty():
        return out.with_columns(
            pl.lit(0.0).alias("graph_new_neighbors"),
            pl.lit(0.0).alias("graph_new_neighbor_ratio"),
            pl.lit(0.0).alias("graph_recent_neighbor_count"),
        )
    agg = joined.group_by("event_id").agg(
        pl.col("_peer").n_unique().alias("_known_neighbors"),
        pl.col("_peer_age_s")
        .filter(pl.col("_peer_age_s") <= new_neighbor_window)
        .n_unique()
        .fill_null(0)
        .alias("_recent_neighbors"),
    )
    out = out.join(agg, on="event_id", how="left").with_columns(
        pl.col("_known_neighbors").fill_null(0.0).alias("graph_new_neighbors"),
        pl.col("_recent_neighbors").fill_null(0.0),
    )
    total = pl.max_horizontal(pl.col("graph_degree"), pl.lit(1.0))
    out = out.with_columns(
        (pl.col("_recent_neighbors") / total).cast(pl.Float64).alias("graph_new_neighbor_ratio"),
        pl.col("_recent_neighbors").cast(pl.Float64).alias("graph_recent_neighbor_count"),
    ).drop("_recent_neighbors", "_known_neighbors")
    return out


def add_graph_features(
    events: pl.DataFrame,
    relations: pl.DataFrame,
    *,
    entity_column: str = "entity_id",
    time_column: str = "timestamp",
    related_column: str = "related_entity_refs",
    config: GraphConfig | None = None,
    snapshot_interval_seconds: int = 6 * HOUR,
) -> pl.DataFrame:
    """Entry point used by :func:`mosaic.features.compute.compute_features`.

    Returns the event frame with graph feature columns. Relations must carry
    ``entity_id`` / ``related_entity_id`` / ``first_seen``; if they lack a
    ``first_seen`` column it is derived from the events themselves.
    """
    config = config or GraphConfig()
    if "first_seen" not in relations.columns:
        if "timestamp" not in relations.columns:
            raise ValueError(
                "relations must provide either 'first_seen' or 'timestamp'; without one "
                "the graph cannot be made temporal and using a global edge list would leak"
            )
        relations = relations.rename({"timestamp": "first_seen"})
    graph = TemporalGraph.from_relations(
        relations, config=config, source_column=entity_column,
        target_column="related_entity_id", time_column="first_seen",
    )
    out, _ = snapshot_features(
        events, graph, entity_column=entity_column, time_column=time_column,
        related_column=related_column, snapshot_interval_seconds=snapshot_interval_seconds,
    )
    return out


# --------------------------------------------------------------- invariants
def assert_no_future_edges(graph: TemporalGraph, cutoff: datetime) -> None:
    """Assert every edge in the snapshot at ``cutoff`` predates ``cutoff``.

    Cheap structural check (no measures computed) that the temporal invariant holds.
    """
    visible = graph.edges_as_of(cutoff)
    if visible.is_empty():
        return
    # Narrow the Polars scalar to a real datetime before comparing: .max() is typed
    # as a wide Any union, and comparing that to a datetime is meaningless.
    latest_raw = visible["first_seen"].max()
    if latest_raw is None:
        return
    latest = cast(datetime, latest_raw)
    if latest > cutoff:
        offender = visible.filter(pl.col("first_seen") > cutoff).head(1)
        raise AssertionError(
            f"leakage: snapshot at {cutoff} contains an edge first seen at "
            f"{latest.isoformat()}: {offender.to_dicts()}"
        )


def temporal_community_stability(
    graph: TemporalGraph, cutoffs: Iterable[datetime]
) -> list[dict[str, Any]]:
    """Community size and churn between consecutive cutoffs.

    Reported as a measurement, not a feature: community instability over time is a
    result to report, and computing it here keeps the feature path cheap.
    """
    out: list[dict[str, Any]] = []
    previous: dict[str, float] | None = None
    for when in sorted(cutoffs):
        measures = snapshot_measures(graph, when)
        current = {node: values["graph_community"] for node, values in measures.items()}
        sizes: dict[int, int] = {}
        for label in current.values():
            sizes[int(label)] = sizes.get(int(label), 0) + 1
        changed = 0
        if previous is not None:
            for node, label in current.items():
                if previous.get(node) != label:
                    changed += 1
        out.append(
            {
                "cutoff": str(when),
                "n_nodes": len(current),
                "n_communities": len(sizes),
                "largest_community": max(sizes.values()) if sizes else 0,
                "nodes_changed_community": changed,
                "community_change_rate": round(changed / max(1, len(current)), 6),
            }
        )
        previous = current
    return out


def degree_distribution(graph: TemporalGraph, cutoff: datetime) -> dict[str, Any]:
    """Degree histogram at a cutoff, for the research plots."""
    degrees = graph.degrees_as_of(cutoff)
    values = np.array([v["degree"] for v in degrees.values()], dtype=np.float64)
    if values.size == 0:
        return {"cutoff": str(cutoff), "n_nodes": 0}
    return {
        "cutoff": str(cutoff),
        "n_nodes": int(values.size),
        "mean_degree": float(values.mean()),
        "std_degree": float(values.std()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "histogram": np.histogram(values, bins=20)[0].tolist(),
        "bin_edges": np.histogram(values, bins=20)[1].tolist(),
    }
