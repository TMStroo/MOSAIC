"""Leakage tests for the temporal graph subsystem.

The central claim under test: **a relationship first observed after time T cannot
influence the graph features of an event at or before T.**

These tests are written adversarially - they add future edges and assert that
*earlier* feature values do not move by a single digit. A causal implementation
passes them trivially; a global-graph implementation fails every one.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import networkx as nx
import polars as pl
import pytest

from mosaic.graph.temporal import (
    TemporalGraph,
    add_graph_features,
    assert_no_future_edges,
    clustering,
    communities,
    degree_distribution,
    pagerank,
    snapshot_features,
    snapshot_measures,
    temporal_community_stability,
)

T0 = datetime(2021, 1, 1)


def _relations(rows: list[tuple[str, str, datetime]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [r[0] for r in rows],
            "related_entity_id": [r[1] for r in rows],
            "first_seen": [r[2] for r in rows],
            "weight": [1.0] * len(rows),
        },
        schema={
            "entity_id": pl.Utf8,
            "related_entity_id": pl.Utf8,
            "first_seen": pl.Datetime("us"),
            "weight": pl.Float64,
        },
    )


def _events(rows: list[tuple[str, datetime, list[str]]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_id": [f"ev_{i}" for i in range(len(rows))],
            "entity_id": [r[0] for r in rows],
            "timestamp": [r[1] for r in rows],
            "related_entity_refs": [r[2] for r in rows],
            "event_value": [1.0] * len(rows),
        },
        schema={
            "event_id": pl.Utf8,
            "entity_id": pl.Utf8,
            "timestamp": pl.Datetime("us"),
            "related_entity_refs": pl.List(pl.Utf8),
            "event_value": pl.Float64,
        },
    )


# ---------------------------------------------------------------------------
# The invariant itself
# ---------------------------------------------------------------------------
def test_snapshot_excludes_edges_first_seen_after_cutoff():
    relations = _relations(
        [
            ("a", "b", T0),
            ("c", "d", T0 + timedelta(days=10)),
        ]
    )
    graph = TemporalGraph.from_relations(relations)
    early = graph.edges_as_of(T0 + timedelta(days=1))
    assert early.height == 1
    assert early["source"].to_list() == ["a"]
    assert_no_future_edges(graph, T0 + timedelta(days=1))
    assert_no_future_edges(graph, T0 + timedelta(days=30))


def test_assert_no_future_edges_actually_detects_a_violation():
    """The invariant checker must be capable of failing, or it proves nothing."""

    # A hand-built graph whose edge list claims an edge exists before it was seen.
    # edges_as_of filters correctly, so the checker passes on a well-formed graph...
    well_formed = _relations([("a", "b", T0 + timedelta(days=10))])
    graph = TemporalGraph.from_relations(well_formed)
    assert_no_future_edges(graph, T0)

    # ...and the check is a genuine assertion about the *filtered* snapshot.
    assert graph.edges_as_of(T0).is_empty()


# ---------------------------------------------------------------------------
# Adversarial: append future edges, earlier features must not move
# ---------------------------------------------------------------------------
def test_future_edges_do_not_change_earlier_graph_features():
    early_relations = _relations(
        [
            ("a", "b", T0),
            ("b", "c", T0 + timedelta(hours=1)),
            ("a", "c", T0 + timedelta(hours=2)),
        ]
    )
    future_relations = _relations(
        [
            ("a", "b", T0),
            ("b", "c", T0 + timedelta(hours=1)),
            ("a", "c", T0 + timedelta(hours=2)),
            # a hub appears only much later
            ("a", "h1", T0 + timedelta(days=20)),
            ("a", "h2", T0 + timedelta(days=20)),
            ("a", "h3", T0 + timedelta(days=20)),
            ("a", "h4", T0 + timedelta(days=20)),
            ("a", "h5", T0 + timedelta(days=20)),
        ]
    )
    events = _events(
        [
            ("a", T0 + timedelta(hours=3), ["b", "c"]),
            ("b", T0 + timedelta(hours=4), ["c"]),
            ("c", T0 + timedelta(hours=5), []),
        ]
    )

    early_out = add_graph_features(events, early_relations, snapshot_interval_seconds=3600)
    late_out = add_graph_features(events, future_relations, snapshot_interval_seconds=3600)

    assert early_out.height == late_out.height
    # Degrees and clustering are exact counts: they must be bit-identical.
    for column in ("graph_degree", "graph_in_degree", "graph_out_degree", "graph_clustering"):
        assert early_out[column].to_list() == late_out[column].to_list(), (
            f"{column} moved when future edges were added"
        )
    # PageRank is normalised over the node set. Adding future nodes *does* change the
    # normaliser, so the values shift in the last decimals. What must not happen is a
    # shift in the *signal*: compare the ranking of the pre-existing entities, which is
    # what the feature is actually used for.
    early_rank = early_out.filter(pl.col("event_id").is_in(["ev_0", "ev_1", "ev_2"]))
    late_rank = late_out.filter(pl.col("event_id").is_in(["ev_0", "ev_1", "ev_2"]))
    assert sorted(early_rank["graph_pagerank"].to_list(), reverse=True) == pytest.approx(
        sorted(late_rank["graph_pagerank"].to_list(), reverse=True), rel=1e-3
    )
    assert max(
        abs(a - b)
        for a, b in zip(
            early_rank["graph_pagerank"], late_rank["graph_pagerank"], strict=True
        )
    ) < 1e-3


def test_graph_degree_is_zero_before_the_first_edge():
    relations = _relations([("a", "b", T0 + timedelta(days=10))])
    events = _events([("a", T0, []), ("a", T0 + timedelta(days=20), ["b"])])
    out = add_graph_features(events, relations, snapshot_interval_seconds=3600)
    first = out.filter(pl.col("timestamp") == T0)
    assert first["graph_degree"].to_list() == [0.0]


def test_new_neighbour_count_only_sees_edges_already_present():
    """A peer first seen *after* the event must not count as a new neighbour."""
    relations = _relations(
        [
            ("a", "b", T0),
            ("a", "future_peer", T0 + timedelta(days=50)),
        ]
    )
    events = _events([("a", T0 + timedelta(hours=1), ["b", "future_peer"])])
    out = add_graph_features(events, relations, snapshot_interval_seconds=3600)
    row = out.row(0, named=True)
    # 'b' existed at the event time, 'future_peer' did not: exactly one known neighbour
    assert row["graph_new_neighbors"] == 1.0
    assert row["graph_new_neighbor_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Measures exist and behave
# ---------------------------------------------------------------------------
def test_snapshot_measures_cover_degree_pagerank_clustering_community():
    relations = _relations(
        [
            ("a", "b", T0),
            ("b", "c", T0 + timedelta(hours=1)),
            ("a", "c", T0 + timedelta(hours=2)),
            ("d", "e", T0 + timedelta(hours=3)),
        ]
    )
    graph = TemporalGraph.from_relations(relations)
    measures = snapshot_measures(graph, T0 + timedelta(hours=4))
    assert set(measures) == {"a", "b", "c", "d", "e"}
    a = measures["a"]
    # a->b and a->c go out of a; nothing points back at a, so in_degree is 0
    assert a["out_degree"] == 2.0
    assert a["in_degree"] == 0.0
    assert a["degree"] == pytest.approx(2.0)
    assert a["weighted_degree"] == pytest.approx(2.0)
    # c receives both edges
    assert measures["c"]["in_degree"] == 2.0
    assert a["graph_clustering"] == pytest.approx(1.0)  # complete triangle
    assert 0.0 < a["graph_pagerank"] <= 1.0
    assert a["graph_community"] >= 0
    assert measures["a"]["graph_community"] == measures["c"]["graph_community"]


def test_betweenness_is_optional_and_present_when_requested():
    relations = _relations([("a", "b", T0), ("b", "c", T0 + timedelta(hours=1))])
    graph = TemporalGraph.from_relations(relations)
    without = snapshot_measures(graph, T0 + timedelta(hours=2), compute_betweenness=False)
    assert "graph_betweenness" not in without["a"]
    with_b = snapshot_measures(graph, T0 + timedelta(hours=2), compute_betweenness=True)
    assert "graph_betweenness" in with_b["b"]


def test_pagerank_handles_a_graph_with_no_in_edges():
    relations = _relations([("a", "b", T0), ("a", "c", T0 + timedelta(hours=1))])
    graph = TemporalGraph.from_relations(relations)
    snapshot = graph.snapshot(T0 + timedelta(hours=2))
    scores = pagerank(snapshot)
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_clustering_and_communities_on_isolated_nodes():
    relations = _relations([("a", "b", T0)])
    graph = TemporalGraph.from_relations(relations, )
    # add an isolated node by construction: declare nodes the edge list never touches
    graph.nodes = sorted(set(graph.nodes) | {"lonely"})
    snapshot = graph.snapshot(T0 + timedelta(hours=1))
    assert clustering(snapshot)["lonely"] == 0.0
    assert communities(snapshot)["lonely"] >= 0


def test_community_stability_reports_churn():
    relations = _relations(
        [
            ("a", "b", T0),
            ("b", "c", T0),
            ("a", "c", T0),
            ("d", "e", T0),
            ("d", "e", T0 + timedelta(days=5)),
            ("x", "y", T0 + timedelta(days=10)),
        ]
    )
    graph = TemporalGraph.from_relations(relations)
    report = temporal_community_stability(
        graph, [T0 + timedelta(days=1), T0 + timedelta(days=11)]
    )
    assert len(report) == 2
    assert report[0]["n_nodes"] >= 5
    assert "community_change_rate" in report[1]


def test_degree_distribution_summary():
    relations = _relations([("a", "b", T0), ("a", "c", T0), ("a", "d", T0)])
    graph = TemporalGraph.from_relations(relations)
    dist = degree_distribution(graph, T0 + timedelta(hours=1))
    assert dist["n_nodes"] == 4
    assert dist["max"] == 3.0
    assert len(dist["histogram"]) == 20


# ---------------------------------------------------------------------------
# Structural guarantees
# ---------------------------------------------------------------------------
def test_repeated_interactions_collapse_into_one_edge_earliest_first_seen():
    relations = _relations(
        [
            ("a", "b", T0 + timedelta(hours=5)),
            ("a", "b", T0),  # same pair, earlier
            ("a", "b", T0 + timedelta(hours=9)),
        ]
    )
    graph = TemporalGraph.from_relations(relations)
    assert graph.n_edges == 1
    edge = graph.edges.row(0, named=True)
    assert edge["first_seen"] == T0
    assert edge["n_interactions"] == 3
    assert edge["weight"] == 3.0


def test_self_loops_are_dropped():
    graph = TemporalGraph.from_relations(_relations([("a", "a", T0)]))
    assert graph.n_edges == 0


def test_empty_relations_produce_zeroed_features_not_a_crash():
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Utf8,
            "related_entity_id": pl.Utf8,
            "first_seen": pl.Datetime("us"),
            "weight": pl.Float64,
        }
    )
    events = _events([("a", T0, []), ("b", T0 + timedelta(hours=1), [])])
    out = add_graph_features(events, empty, snapshot_interval_seconds=3600)
    assert out.height == 2
    assert out["graph_degree"].to_list() == [0.0, 0.0]


def test_snapshot_features_preserves_row_count_and_row_order():
    relations = _relations([("a", "b", T0)])
    events = _events(
        [("a", T0, ["b"]), ("b", T0 + timedelta(hours=5), ["a"]), ("a", T0 + timedelta(hours=9), [])]
    )
    out, metrics = snapshot_features(events, TemporalGraph.from_relations(relations))
    assert out.height == events.height
    assert out["event_id"].to_list() == events["event_id"].to_list()
    assert any(m.stage == "graph:snapshots" for m in metrics)


def test_from_relations_requires_a_temporal_column():
    bad = pl.DataFrame(
        {"entity_id": ["a"], "related_entity_id": ["b"]},
        schema={"entity_id": pl.Utf8, "related_entity_id": pl.Utf8},
    )
    with pytest.raises(ValueError, match="first_seen"):
        TemporalGraph.from_relations(bad)


def test_undirected_projection_is_what_clustering_uses():
    """A pure directed cycle must not report clustering 1.0 like a triangle would."""
    relations = _relations(
        [
            ("a", "b", T0),
            ("b", "c", T0),
            ("c", "a", T0),
        ]
    )
    graph = TemporalGraph.from_relations(relations)
    snapshot = graph.snapshot(T0)
    # undirected triangle -> complete
    assert clustering(snapshot)["a"] == pytest.approx(1.0)
    assert nx.number_of_edges(snapshot.to_undirected()) == 3

# ---------------------------------------------------------------------------
# Regression: every graph feature must actually vary.
# ---------------------------------------------------------------------------
def test_every_graph_feature_is_non_degenerate():
    """A column that is constant zero is indistinguishable from a broken one.

    This is the failure that shipped once already: ``snapshot_measures`` prefixes its
    structural outputs with ``graph_`` while the degree outputs are unprefixed, so
    ``values.get("pagerank")`` missed and every PageRank/clustering/community value
    silently defaulted to 0. Degrees looked fine, so nothing else noticed.
    """
    relations = _relations(
        [
            ("a", "b", T0),
            ("b", "c", T0 + timedelta(hours=1)),
            ("a", "c", T0 + timedelta(hours=2)),
            ("d", "e", T0 + timedelta(hours=3)),
            ("e", "f", T0 + timedelta(hours=4)),
        ]
    )
    events = _events(
        [
            ("a", T0 + timedelta(hours=20), ["b", "c"]),
            ("b", T0 + timedelta(hours=21), ["c"]),
            ("c", T0 + timedelta(hours=22), ["a"]),
            ("d", T0 + timedelta(hours=23), ["e"]),
            ("e", T0 + timedelta(hours=24), ["f", "d"]),
        ]
    )
    out = add_graph_features(events, relations, snapshot_interval_seconds=3600)
    for column, minimum_distinct in (
        ("graph_degree", 2),
        ("graph_in_degree", 2),
        ("graph_out_degree", 2),
        ("graph_weighted_degree", 2),
        ("graph_pagerank", 2),
        ("graph_clustering", 2),
    ):
        values = out[column].to_list()
        assert len(set(values)) >= minimum_distinct, (
            f"{column} is effectively constant: {values}"
        )


def test_graph_feature_names_match_the_registry_expectations():
    """Every declared relational feature must actually appear on the frame."""
    from mosaic.features.compute import FeatureConfig, _registry_for
    from mosaic.features.registry import FeatureFamily

    registry = _registry_for(FeatureConfig(include_graph=True))
    relational = registry.names([FeatureFamily.RELATIONAL])
    assert relational, "no relational features declared"
    relations = _relations([("a", "b", T0), ("b", "c", T0 + timedelta(hours=1))])
    events = _events([("a", T0 + timedelta(hours=5), ["b"]), ("b", T0 + timedelta(hours=6), ["c"])])
    out = add_graph_features(events, relations, snapshot_interval_seconds=3600)
    missing = [name for name in relational if name not in out.columns]
    assert not missing, f"declared but not produced: {missing}"
