"""Regression tests for peer entity resolution in the production graph pipeline.

Before this module, peer canonicalisation lived only in ``scratch/smoke_graph.py``.
These tests pin the behaviour that harness implemented, so the graph can no longer
silently depend on a dev script: peers resolve through the same entity map as
subjects, unresolved peers are excluded and reported, and provenance survives.
"""

from __future__ import annotations

import polars as pl
import pytest

from mosaic.graph.peers import (
    SELF_REFERENCE,
    UNRESOLVED,
    attach_subject_entities,
    build_relations,
    build_resolved_graph,
    peer_resolution_report,
    resolve_peer_references,
)


@pytest.fixture
def entity_map() -> pl.DataFrame:
    """Two subjects in source_a plus one in source_b that clusters with BETA.

    Built to match the resolver's real semantics: a reference only acquires a
    distinct entity_id when it clusters with a reference from ANOTHER source, so
    a peer that appears in the same source as its subject is a same-source
    reference and deliberately has no entity of its own.
    """
    return pl.DataFrame(
        {
            "source_id": ["source_a", "source_a", "source_b", "source_b"],
            "entity_ref_norm": ["ALPHA", "BETA", "BETA", "GAMMA"],
            "entity_ref_raw": ["Alpha", "Beta", "BETA-UK", "Gamma"],
            "entity_id": ["en_aaa", "en_bbb", "en_bbb", "en_ccc"],
            "cluster_root": ["en_aaa", "en_bbb", "en_bbb", "en_ccc"],
        }
    )


@pytest.fixture
def events() -> pl.DataFrame:
    """Five events: two known peers, a self-reference, a peerless one, an outsider.

    e5 resolves because the map holds (source_b, BETA); e4 does not because
    UNKNOWN appears in no source. That pair is the whole distinction under test.
    """
    return pl.DataFrame(
        {
            "event_id": ["e1", "e2", "e3", "e4", "e5"],
            "source_id": ["source_a", "source_a", "source_a", "source_a", "source_b"],
            "entity_ref_norm": ["ALPHA", "ALPHA", "BETA", "BETA", "GAMMA"],
            "entity_ref_raw": ["Alpha", "Alpha", "Beta", "Beta", "Gamma"],
            "peer_ref_norm": ["BETA", "ALPHA", None, "UNKNOWN", "BETA"],
            "peer_ref_raw": ["Beta", "Alpha", None, "Unknown", "Beta"],
            "related_entity_refs": ["BETA", "ALPHA", None, "UNKNOWN", "BETA"],
            "timestamp": [
                _ts("2021-01-01 00:00:00"),
                _ts("2021-01-02 00:00:00"),
                _ts("2021-01-03 00:00:00"),
                _ts("2021-01-04 00:00:00"),
                _ts("2021-01-05 00:00:00"),
            ],
        }
    )


def _ts(value: str) -> object:
    from datetime import datetime

    return datetime.fromisoformat(value)


def test_resolved_peers_reach_the_graph(entity_map: pl.DataFrame, events: pl.DataFrame) -> None:
    """The headline regression: a peer resolved by the pipeline becomes an edge.

    Asserted on the graph's own node set, not just the intermediate frame, because
    the failure this guards against is a graph that built successfully from edges
    that no participant was ever a real part of.
    """
    built = build_resolved_graph(events, entity_map)
    graph = built.graph
    assert graph is not None

    edges = set(
        built.relations.select("entity_id", "related_entity_id").iter_rows()
    )
    # e1: ALPHA -> BETA. Both are real entities, so the edge must exist.
    assert ("en_aaa", "en_bbb") in edges
    # e2: ALPHA -> ALPHA is a self-reference, never an edge.
    assert ("en_aaa", "en_aaa") not in edges
    # e4's peer is unknown to the map, so BETA gains no second edge from it.
    assert built.relations.height == 2  # ALPHA->BETA and GAMMA->BETA

    # And the edge is visible in the graph's own edge list, not just the frame.
    visible = graph.edges_as_of(_ts("2021-01-05 00:00:00"))
    pairs = set(visible.select("source", "target").iter_rows())
    assert ("en_aaa", "en_bbb") in pairs


def test_unresolved_peer_is_labelled_and_excluded(
    entity_map: pl.DataFrame, events: pl.DataFrame
) -> None:
    """A peer absent from the map keeps its raw ref, is marked, and makes no edge."""
    result = resolve_peer_references(attach_subject_entities(events, entity_map), entity_map)
    out = result.events

    unresolved = out.filter(pl.col("event_id") == "e4")
    assert unresolved.height == 1
    assert unresolved["peer_resolution_status"][0] == UNRESOLVED
    # The raw reference is retained, so the miss can be explained rather than guessed.
    assert unresolved["peer_ref_raw"][0] == "Unknown"
    assert unresolved["peer_source_id"][0] == "source_a"
    # An unresolved peer is deliberately non-null; only the status gates the edge.
    assert unresolved["related_entity_refs"][0] == "UNKNOWN"

    relations = build_relations(out)
    assert "UNKNOWN" not in set(relations["related_entity_id"].to_list())


def test_peerless_event_is_not_reported_as_an_unresolved_peer(
    entity_map: pl.DataFrame, events: pl.DataFrame
) -> None:
    """A null peer means "no peer named", which is different from "peer failed".

    Conflating them inflates the unresolved count with every peerless event and
    makes the resolution rate a meaningless number.
    """
    result = resolve_peer_references(attach_subject_entities(events, entity_map), entity_map)
    peerless = result.events.filter(pl.col("event_id") == "e3")
    assert peerless["peer_resolution_status"][0] != UNRESOLVED
    assert result.counts.get(UNRESOLVED, 0) == 1


def test_self_reference_is_labelled_separately(
    entity_map: pl.DataFrame, events: pl.DataFrame
) -> None:
    """A self-edge is a real observation but carries no relational information."""
    result = resolve_peer_references(attach_subject_entities(events, entity_map), entity_map)
    self_ref = result.events.filter(pl.col("event_id") == "e2")
    assert self_ref["peer_resolution_status"][0] == SELF_REFERENCE
    assert result.counts.get(SELF_REFERENCE, 0) == 1


def test_provenance_survives_resolution(
    entity_map: pl.DataFrame, events: pl.DataFrame
) -> None:
    """Every resolved peer can be traced back to the source and raw string."""
    result = resolve_peer_references(attach_subject_entities(events, entity_map), entity_map)
    row = result.events.filter(pl.col("event_id") == "e1").row(0, named=True)
    for column in ("source_id", "peer_ref_raw", "peer_source_id", "peer_resolution_method"):
        assert row.get(column) not in (None, ""), f"lost provenance column {column}"
    assert row["peer_ref_raw"] == "Beta"
    assert row["peer_source_id"] == "source_a"
    assert row["related_entity_refs"] == "en_bbb"


def test_peer_ids_come_from_the_same_id_space_as_subjects(
    entity_map: pl.DataFrame, events: pl.DataFrame
) -> None:
    """Peers resolve into the existing map, never into a second identity space.

    Re-clustering peers separately would let the same party hold two ids and make
    every cross-entity measure meaningless.
    """
    built = build_resolved_graph(events, entity_map)
    subject_ids = set(built.events["entity_id"].to_list())
    peer_ids = set(built.relations["related_entity_id"].to_list())
    assert peer_ids <= subject_ids
    assert peer_ids == {"en_bbb"}


def test_resolution_report_counts_only_never_raw_refs(
    entity_map: pl.DataFrame, events: pl.DataFrame
) -> None:
    """A report is written next to results; raw refs must not leak into it."""
    result = resolve_peer_references(attach_subject_entities(events, entity_map), entity_map)
    report = peer_resolution_report(result)
    assert report["resolved"] == 2
    assert report["unresolved"] == 1
    assert report["self_reference"] == 1
    assert "UNKNOWN" not in str(report)


def test_missing_required_columns_raise_instead_of_defaulting(
    events: pl.DataFrame, entity_map: pl.DataFrame
) -> None:
    """A missing input must be loud. A silent default is how this bug survived."""
    resolved = resolve_peer_references(attach_subject_entities(events, entity_map), entity_map)
    with pytest.raises(ValueError, match="missing"):
        build_relations(resolved.events.drop("peer_resolution_status"))
    with pytest.raises(ValueError, match="missing"):
        build_relations(resolved.events.drop("entity_id"))
    with pytest.raises(ValueError, match="missing"):
        attach_subject_entities(events, entity_map.drop("entity_id"))


def test_resolved_frame_supports_snapshot_features(
    entity_map: pl.DataFrame, events: pl.DataFrame
) -> None:
    """The graph built from resolved peers must still answer causal queries."""
    built = build_resolved_graph(events, entity_map)
    graph = built.graph
    assert graph is not None
    degrees = graph.degrees_as_of(_ts("2021-01-05 00:00:00"))
    # At the final cutoff both resolved endpoints are present in the graph.
    assert degrees["en_aaa"]["out_degree"] == 1.0
    assert degrees["en_ccc"]["out_degree"] == 1.0
    assert degrees["en_bbb"]["in_degree"] == 2.0
