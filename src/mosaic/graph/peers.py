"""Attach resolved canonical entity ids to both ends of a relationship.

Why this module exists
----------------------
An event names two parties: the subject (``entity_ref_source``) and, optionally,
a peer (``peer_ref_raw``). Both are *raw source references*, not entity ids --
``site::mf142`` in one source and ``site mf142`` in another are the same place,
and neither is an id. Until they are resolved, a relationship between two events
cannot be stated in entity space, and every graph measure computed on raw
strings is meaningless: the two spellings of one site look like two nodes.

The subject side was resolved by the caller. The **peer** side was not: it was
resolved in a scratch harness. That is not acceptable as production behaviour,
because the harness is not part of the package, is not tested, and silently
disappears -- taking peer resolution with it. This module moves it into the
data model, where it is tested and cannot be skipped.

What resolution means for a peer
--------------------------------
A peer reference is resolved through the *same* :class:`EntityResolver` and the
*same* entity map as a subject. It is not matched a second time and it is not
heuristically normalized. One resolver, one clustering, one id space: if the
peer is the subject of some other event, the two share an ``entity_id``, and
that is what makes the resulting edge a real relationship rather than a string
comparison that happened to be true once.

Provenance, because a resolved peer is a claim
----------------------------------------------
Resolution is lossy and can be wrong, so the raw reference is never overwritten.
:func:`resolve_peer_references` keeps the source reference next to the canonical
id and records *how* the peer was resolved:

==========================  ==========================================
column                      meaning
==========================  ==========================================
``related_entity_refs``     the canonical peer entity id (graph input)
``peer_ref_raw``            the peer's raw source reference, untouched
``peer_ref_norm``           its normalized form, the join key used
``peer_source_id``          which source the peer reference came from
``peer_resolution_status``  ``resolved`` / ``self`` / ``unresolved``
``peer_resolution_method``  the matcher's own name, or the reason it failed
==========================  ==========================================

``unresolved`` is a real outcome, not an error. A peer reference that no event
ever names as a subject has no cluster, because entity resolution only clusters
references it has been given. Those rows keep their raw reference and are
excluded from the graph rather than being attached to an invented entity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import polars as pl

LOGGER = logging.getLogger(__name__)

#: Written into ``peer_resolution_status`` when the peer matched no cluster.
UNRESOLVED = "unresolved"
#: Written when the peer resolved to the same entity as the event's subject.
SELF_REFERENCE = "self"
#: Written when the peer reference resolved to a canonical entity id.
RESOLVED = "resolved"

#: Columns this module adds to the event frame. Declared once so a caller can
#: assert the data model produced what the graph layer expects.
PEER_RESOLUTION_COLUMNS = (
    "related_entity_refs",
    "peer_source_id",
    "peer_resolution_status",
    "peer_resolution_method",
)


@dataclass
class PeerResolutionResult:
    """Outcome of peer canonicalization, for reporting and tests."""

    events: pl.DataFrame
    #: ref_norm -> canonical entity id, as actually used. Keys are
    #: ``(source_id, ref_norm)`` pairs, matching the subject-side entity map.
    peer_map: pl.DataFrame
    counts: dict[str, int] = field(default_factory=dict)
    unresolved_examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_resolved(self) -> int:
        return int(self.counts.get(RESOLVED, 0))

    @property
    def n_unresolved(self) -> int:
        return int(self.counts.get(UNRESOLVED, 0))

    def summary(self) -> dict[str, Any]:
        total = sum(self.counts.values())
        return {
            "rows": total,
            "resolved": self.n_resolved,
            "unresolved": self.n_unresolved,
            "self_reference": int(self.counts.get(SELF_REFERENCE, 0)),
            "distinct_peer_refs": int(self.peer_map.height) if self.peer_map.height else 0,
            "resolution_rate": round(self.n_resolved / total, 6) if total else 0.0,
        }


def attach_subject_entities(
    events: pl.DataFrame,
    entity_map: pl.DataFrame,
    *,
    drop_existing: bool = True,
) -> pl.DataFrame:
    """Join canonical entity ids onto the event frame for the subject.

    The key is ``(entity_ref_norm, source_id)``, not ``entity_ref_norm`` alone:
    the same normalized string can appear in two sources with different damage,
    and keying on the string alone silently picks whichever source the map
    happens to list first.
    """
    needed = {"entity_ref_norm", "source_id", "entity_id"}
    missing = needed - set(entity_map.columns)
    if missing:
        raise ValueError(f"entity_map is missing {sorted(missing)}")
    carry = [c for c in ("entity_id", "cluster_root") if c in entity_map.columns]
    left = events.drop([c for c in ("entity_id", "cluster_root") if c in events.columns]) if drop_existing else events
    return left.join(
        entity_map.select("entity_ref_norm", "source_id", *carry),
        on=["entity_ref_norm", "source_id"],
        how="left",
    )


def _peer_references(events: pl.DataFrame) -> pl.DataFrame:
    """One row per distinct (source_id, peer_ref_norm), with a raw ref and count.

    Built as its own reference table rather than joined row-by-row: a peer
    reference that appears in 40,000 events is one fact about the world, and
    resolving it 40,000 times is both slower and a chance for the 40,000 copies
    to disagree.
    """
    return (
        events.filter(pl.col("peer_ref_norm").is_not_null() & (pl.col("peer_ref_norm") != ""))
        .group_by("source_id", "peer_ref_norm")
        .agg(
            pl.col("peer_ref_raw").min().alias("peer_ref_raw"),
            pl.len().alias("peer_occurrences"),
        )
        .sort("source_id", "peer_ref_norm")
    )


def _entity_lookup(entity_map: pl.DataFrame) -> pl.DataFrame:
    """(source_id, ref_norm) -> (entity_id, method) for peer joins.

    One row per key. ``mode().first()`` rather than the first row in file order:
    two references sharing a cluster is exactly the case where a deterministic
    choice matters, and file order is not a defensible tie-break.
    """
    return entity_map.group_by("source_id", "entity_ref_norm").agg(
        pl.col("entity_id").mode().first().alias("entity_id"),
        pl.col("cluster_root").mode().first().alias("cluster_root"),
    ).rename({"entity_ref_norm": "peer_ref_norm"})


def resolve_peer_references(
    events: pl.DataFrame,
    entity_map: pl.DataFrame,
    *,
    matching_method: str = "entity_resolution_cluster",
    keep_unresolved_raw: bool = True,
) -> PeerResolutionResult:
    """Canonicalize ``related_entity_refs`` into a peer entity id.

    The event frame keeps every raw peer reference it arrived with. Only the
    derived ``related_entity_refs`` column is replaced with a canonical id, and
    a peer that resolved to nothing keeps its normalized raw form there so the
    graph layer can recognise and drop it rather than joining on a string that
    was never an entity.

    ``keep_unresolved_raw=False`` nulls the column instead. The graph layer
    treats null and raw-string peers identically (both are excluded), so this
    only matters to a caller that wants nulls to be visibly distinct.
    """
    for col in ("source_id", "peer_ref_norm", "peer_ref_raw"):
        if col not in events.columns:
            raise ValueError(
                f"events frame is missing {col!r}; peer resolution needs the raw peer "
                "reference and its source to join on, not just a normalized string"
            )

    lookup = _entity_lookup(entity_map)
    peers = _peer_references(events)

    if peers.is_empty():
        out = events.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("related_entity_refs"),
            pl.lit(UNRESOLVED, dtype=pl.Utf8).alias("peer_resolution_status"),
            pl.lit(matching_method, dtype=pl.Utf8).alias("peer_resolution_method"),
            pl.col("source_id").alias("peer_source_id"),
        )
        return PeerResolutionResult(
            events=out, peer_map=pl.DataFrame(schema={"source_id": pl.Utf8, "peer_ref_norm": pl.Utf8, "entity_id": pl.Utf8, "peer_resolution_status": pl.Utf8}),
            counts={},
        )

    joined = peers.join(
        lookup, on=["source_id", "peer_ref_norm"], how="left"
    ).with_columns(
        pl.when(pl.col("entity_id").is_not_null())
        .then(pl.lit(RESOLVED))
        .otherwise(pl.lit(UNRESOLVED))
        .alias("peer_resolution_status"),
        pl.when(pl.col("entity_id").is_not_null())
        .then(pl.lit(matching_method, dtype=pl.Utf8))
        .otherwise(pl.lit("no_cluster_for_this_reference", dtype=pl.Utf8))
        .alias("peer_resolution_method"),
    )

    peer_map = joined.select(
        "source_id",
        "peer_ref_norm",
        pl.col("peer_ref_raw"),
        pl.col("entity_id"),
        "peer_resolution_status",
        "peer_resolution_method",
        "peer_occurrences",
    )

    # One join to put the resolution back onto every event that mentions the peer.
    # Joined with `on=` (not left_on/right_on) because the event frame already has
    # `source_id` and `peer_ref_norm` under those exact names: with left_on/right_on
    # the right-hand key columns are consumed by the join and cannot be dropped
    # afterwards, and renaming them to survive is what created the shadowing bug.
    events_joined = events.join(
        peer_map.select(
            "source_id",
            "peer_ref_norm",
            pl.col("entity_id").alias("_peer_entity_id"),
            pl.col("peer_resolution_status").alias("_peer_status"),
            pl.col("peer_resolution_method").alias("_peer_method"),
        ),
        on=["source_id", "peer_ref_norm"],
        how="left",
    ).with_columns(
        # A null peer_ref_norm means "this event names no peer" - not a peer that
        # failed to resolve. Conflating the two would report every peerless event
        # as an unresolved peer.
        pl.when(pl.col("peer_ref_norm").is_null() | (pl.col("peer_ref_norm") == ""))
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(pl.col("_peer_status") == RESOLVED)
        .then(pl.col("_peer_entity_id"))
        .when(pl.col("_peer_status") == UNRESOLVED)
        .then(pl.col("peer_ref_norm") if keep_unresolved_raw else pl.lit(None, dtype=pl.Utf8))
        .alias("related_entity_refs"),
        pl.when(pl.col("peer_ref_norm").is_null() | (pl.col("peer_ref_norm") == ""))
        .then(pl.lit(None, dtype=pl.Utf8))
        .otherwise(pl.col("_peer_status"))
        .alias("peer_resolution_status"),
        pl.when(pl.col("peer_ref_norm").is_null() | (pl.col("peer_ref_norm") == ""))
        .then(pl.lit(None, dtype=pl.Utf8))
        .otherwise(pl.col("_peer_method"))
        .alias("peer_resolution_method"),
        pl.col("source_id").alias("peer_source_id"),
    ).drop(
        # The three aliased value columns are the only ones the join added.
        "_peer_entity_id",
        "_peer_status",
        "_peer_method",
    )

    # A self-relationship is a real observation (an event naming its own subject)
    # but it carries no relational information, so it is labelled and left for
    # the graph layer to exclude rather than being silently dropped here.
    # Only possible once the subject's entity_id is on the frame: peer resolution
    # can legitimately run before subject ids are attached, and a self-edge check
    # that assumes otherwise would force a call order on the caller.
    if "entity_id" in events_joined.columns:
        events_joined = events_joined.with_columns(
            pl.when(pl.col("related_entity_refs") == pl.col("entity_id"))
            .then(pl.lit(SELF_REFERENCE))
            .otherwise(pl.col("peer_resolution_status"))
            .alias("peer_resolution_status")
        )

    counts = {
        str(k): int(v)
        for k, v in events_joined.group_by("peer_resolution_status")
        .len()
        .iter_rows()
        if k is not None
    }
    unresolved_examples = (
        peer_map.filter(pl.col("peer_resolution_status") == UNRESOLVED)
        .select("source_id", "peer_ref_norm", "peer_occurrences")
        .head(5)
        .to_dicts()
    )
    if unresolved_examples:
        LOGGER.warning(
            "peer resolution: %d distinct peer references had no cluster "
            "(sample: %s); their edges are excluded from the graph",
            len(unresolved_examples),
            unresolved_examples[:3],
        )
    return PeerResolutionResult(
        events=events_joined, peer_map=peer_map, counts=counts, unresolved_examples=unresolved_examples
    )


def build_relations(
    events: pl.DataFrame,
    *,
    entity_column: str = "entity_id",
    peer_column: str = "related_entity_refs",
    time_column: str = "timestamp",
    status_column: str = "peer_resolution_status",
) -> pl.DataFrame:
    """The edge list the graph layer consumes, built from canonical ids only.

    Filters on ``peer_resolution_status`` rather than on a null check, because an
    unresolved peer is deliberately non-null: it holds its raw reference so it
    can be reported. Filtering on null would let that string reach the graph and
    become a node named after a source value.
    """
    required = {entity_column, peer_column, time_column, status_column}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(
            f"events frame is missing {sorted(missing)}; run resolve_peer_references "
            "before build_relations so peer references are canonical entity ids"
        )
    usable = events.filter(
        pl.col(peer_column).is_not_null() & (pl.col(peer_column) != "")
        & (pl.col(status_column) == RESOLVED)
    ).select(
        # The subject column keeps its name; the peer column is renamed to
        # ``related_entity_id`` because that is the name
        # TemporalGraph.from_relations and add_graph_features read.
        entity_column,
        pl.col(peer_column).alias("related_entity_id"),
        pl.col(time_column).alias("first_seen"),
    )
    # A self-edge is not a relationship; it would add a self-loop to every
    # measure and inflate degree. Compared after the rename, so the comparison
    # uses the names the frame actually has.
    return (
        usable.filter(pl.col(entity_column) != pl.col("related_entity_id"))
        .unique()
        .sort(entity_column, "related_entity_id")
    )


def peer_resolution_report(result: PeerResolutionResult) -> dict[str, Any]:
    """Small dict for a run report. Counts only -- never the reference strings.

    A peer reference is a raw source value, and putting unresolved ones into a
    report that gets written next to results is how a malformed value ends up
    looking like a finding.
    """
    summary = result.summary()
    summary["unresolved_distinct"] = int(
        result.peer_map.filter(pl.col("peer_resolution_status") == UNRESOLVED).height
    )
    summary["resolved_distinct"] = int(
        result.peer_map.filter(pl.col("peer_resolution_status") == RESOLVED).height
    )
    return summary


@dataclass
class ResolvedRelations:
    """Canonical events, the edge list derived from them, and the provenance."""

    events: pl.DataFrame
    relations: pl.DataFrame
    peer_result: PeerResolutionResult
    graph: Any = None

    def summary(self) -> dict[str, Any]:
        out = peer_resolution_report(self.peer_result)
        out["relations"] = self.relations.height
        if self.graph is not None:
            out["nodes"] = self.graph.n_nodes
            out["edges"] = self.graph.n_edges
        return out


def build_resolved_graph(
    events: pl.DataFrame,
    entity_map: pl.DataFrame,
    *,
    config: Any = None,
    matching_method: str = "entity_resolution_cluster",
) -> ResolvedRelations:
    """The production path from cleaned events to a temporal graph.

    One function, because the three steps have to agree: resolving peers without
    subject ids attached, or building relations from a frame where peers are
    still raw, are both silent ways to produce a graph whose edges mean nothing.
    Doing them here makes the correct order the only order, and it is what lets
    a scratch harness stop being load-bearing.

    ``config`` is passed through to :class:`~mosaic.graph.temporal.GraphConfig`;
    left untyped at the signature boundary to keep this module free of an import
    cycle with ``temporal`` (``temporal`` already imports nothing from here, but
    keeping the arrow one-way is deliberate).
    """
    from mosaic.graph.temporal import GraphConfig, TemporalGraph

    resolved = attach_subject_entities(events, entity_map)
    peer_result = resolve_peer_references(
        resolved, entity_map, matching_method=matching_method
    )
    relations = build_relations(peer_result.events)
    graph = TemporalGraph.from_relations(relations, config=config or GraphConfig())
    return ResolvedRelations(
        events=peer_result.events,
        relations=relations,
        peer_result=peer_result,
        graph=graph,
    )
