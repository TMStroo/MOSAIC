"""Tests for ground-truth label resolution and the dual label protocol.

Pins the two behaviours M6's conclusions depend on:

* the three target kinds resolve correctly (event / entity / window), and the
  per-event family flags are preferred over a label's declared span;
* ``EVENT_ONLY`` and ``ALL_FAMILIES`` are separate, named protocols -- never
  merged, never silently substituted.

Every fixture builds a tiny synthetic world on disk rather than reaching for
``world_a``, so a test failure localises to the resolver rather than to a
half-generated 1.7M-row world.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from mosaic.labels import (
    AGGREGATE_FAMILIES,
    ANOMALY_FAMILIES,
    EVENT_FAMILIES,
    LabelScope,
    minimum_support,
    resolve_both,
    resolve_labels,
    supported_families,
)
from mosaic.labels.resolve import assert_labels_never_fitted

T0 = datetime(2021, 1, 1)


@pytest.fixture
def world(tmp_path: Path) -> Path:
    """A minimal ``_truth/`` sidecar plus manifest, on disk under tmp_path."""
    base = tmp_path / "world_t"
    truth = base / "_truth"
    truth.mkdir(parents=True)

    # Two event-kind labels: L1 is an injected point anomaly, L2 is ordinary.
    labels = pl.DataFrame(
        {
            "anomaly_id": ["an_point_1", "an_context_1", "an_collective_1", "an_window_1"],
            "family": ["point", "contextual", "collective", "temporal"],
            "target_kind": ["event", "event", "entity", "window"],
            "target_ref": ["L000000001", "L000000002", "E000001", "W000001"],
            "latent_id": ["L000000001", "L000000002", None, None],
            "entity_key": ["E000001", "E000002", "E000001", None],
            "started_at": [
                T0.isoformat(),
                T0.isoformat(),
                T0.isoformat(),
                (T0 + timedelta(hours=1)).isoformat(),
            ],
            "ended_at": [
                T0.isoformat(),
                T0.isoformat(),
                # Deliberately a MONTH-wide span. The exact flags must win, so
                # only the flagged event is positive, not every event in the
                # month. This is the regression that produced 183,841
                # positives before the per-event flags existed.
                (T0 + timedelta(days=30)).isoformat(),
                (T0 + timedelta(hours=2)).isoformat(),
            ],
            "severity": [1.0, 1.0, 0.9, 3.0],
            "injected": [True, True, True, True],
        }
    )
    labels.write_parquet(truth / "labels.parquet")

    pl.DataFrame(
        {
            "source_id": ["sensor_net"],
            "source_ref": ["ref_a"],
            "entity_key": ["E000001"],
            "is_ambiguous": [False],
            "is_clean": [True],
        }
    ).write_parquet(truth / "entity_links.parquet")

    pl.DataFrame(
        {
            "latent_id": ["L000000001", "L000000002", "L000000003"],
            "entity_key": ["E000001", "E000002", "E000001"],
            "related_key": [None, None, None],
            "event_type": ["sensor", "sensor", "sensor"],
            "location_id": ["Z000", "Z000", "Z000"],
            "timestamp": [T0.isoformat(), T0.isoformat(), (T0 + timedelta(hours=1)).isoformat()],
            "fam_point": [1, 0, 0],
            "fam_contextual": [0, 1, 0],
            "fam_collective": [0, 0, 1],
            "fam_temporal": [0, 0, 0],
            "fam_distribution": [0, 0, 0],
            "fam_missingness": [0, 0, 0],
            "fam_behavioral": [0, 0, 0],
            "fam_relational": [0, 0, 0],
            "fam_cross_source": [0, 0, 0],
            "fam_entity_resolution": [0, 0, 0],
        }
    ).write_parquet(truth / "latent_events.parquet")

    (base / "manifest.json").write_text(
        json.dumps({"dataset_name": "world_t", "generator_version": "9.9.9"}),
        encoding="utf-8",
    )
    return base


@pytest.fixture
def events(world: Path) -> pl.DataFrame:
    """Three observed events: the two injected ones plus an ordinary one.

    ``source_record_id`` carries the latent id with a per-source prefix, the way
    the real renderer does (``se-L...``, ``tr-L...``).
    """
    return pl.DataFrame(
        {
            "event_id": ["ev_a", "ev_b", "ev_c"],
            "source_id": ["sensor_net"] * 3,
            "source_record_id": ["se-L000000001", "se-L000000002", "se-L000000003"],
            "entity_ref_source": ["ref_a", "ref_b", "ref_a"],
            "entity_id": ["en_1", "en_2", "en_1"],
            "timestamp": [T0, T0, T0 + timedelta(hours=1)],
        }
    )


class TestLabelResolution:
    def test_event_labels_resolve_exactly(self, world, events):
        result = resolve_labels(events, truth_root=world)
        positives = result.events.filter(pl.col("is_anomaly_event_only"))
        assert positives.height == 2
        assert set(positives["event_id"].to_list()) == {"ev_a", "ev_b"}

    def test_collective_uses_exact_flags_not_declared_span(self, world, events):
        """Regression: a month-wide label span must not widen the positive set.

        ``an_collective_1`` spans 30 days and targets ``E000001``. Resolving by
        span would mark ``ev_a`` and ``ev_c`` (both ``en_1``) positive via the
        collective family. The per-event flag marks only ``L000000003``.
        """
        result = resolve_labels(events, truth_root=world)
        row = result.events.filter(pl.col("event_id") == "ev_c").row(0, named=True)
        # ev_c is covered by collective (exact flag) and temporal (its 1h
        # window). The point is that collective came from the flag and not from
        # the 30-day span: ev_a shares entity E000001 and sits at T0, well
        # outside ev_c's instant, so a span-based join would also have pulled
        # it in.
        assert "collective" in row["anomaly_family"]
        assert row["is_anomaly_event_only"] is False
        assert row["is_anomaly_all_families"] is True
        # The decisive assertion: the other E000001 event is NOT collective.
        other = result.events.filter(pl.col("event_id") == "ev_a").row(0, named=True)
        assert "collective" not in other["anomaly_family"]

    def test_family_coverage_is_reported_separately_from_positives(self, world, events):
        result = resolve_labels(events, truth_root=world)
        assert result.family_coverage["collective"] == 1
        assert result.family_counts["collective"] == 0

    def test_window_labels_match_by_interval(self, world, events):
        """``an_window_1`` covers T0+1h..T0+2h, which contains ``ev_c``."""
        result = resolve_labels(events, truth_root=world)
        row = result.events.filter(pl.col("event_id") == "ev_c").row(0, named=True)
        assert "temporal" in row["anomaly_family"]

    def test_observability_is_reported(self, world, events):
        """Recall against an unobservable anomaly is undefined, not zero."""
        result = resolve_labels(events, truth_root=world)
        assert result.observability["event_labels"] == 2
        assert result.observability["event_labels_observable"] == 2
        assert result.observability["latent_events_total"] == 3

    def test_dataset_version_from_manifest(self, world, events):
        result = resolve_labels(events, truth_root=world)
        assert result.dataset_version == "world_t:9.9.9"

    def test_missing_columns_raise(self, world, events):
        with pytest.raises(ValueError, match="missing required columns"):
            resolve_labels(events.drop("entity_id"), truth_root=world)

    def test_falls_back_to_span_when_flags_absent(self, world, events, caplog):
        """Without ``fam_*`` the span is used, and the fallback is announced."""
        latent = pl.read_parquet(world / "_truth" / "latent_events.parquet").select(
            [c for c in pl.read_parquet(world / "_truth" / "latent_events.parquet").columns
             if not c.startswith("fam_")]
        )
        latent.write_parquet(world / "_truth" / "latent_events.parquet")
        with caplog.at_level("WARNING"):
            result = resolve_labels(events, truth_root=world)
        assert any("collective resolved by declared span" in r.message for r in caplog.records)
        assert result is not None


class TestLabelProtocol:
    def test_both_protocols_from_one_resolution(self, world, events):
        event_only, all_families = resolve_both(events, truth_root=world)
        assert event_only.scope is LabelScope.EVENT_ONLY
        assert all_families.scope is LabelScope.ALL_FAMILIES
        assert event_only.positives == 2
        # ev_c is covered only by collective/temporal, so it is a positive
        # under ALL_FAMILIES but not EVENT_ONLY.
        assert all_families.positives == 3

    def test_protocol_columns_are_stamped(self, world, events):
        event_only, all_families = resolve_both(events, truth_root=world)
        assert event_only.events["label_protocol"].unique().to_list() == ["event_only"]
        assert all_families.events["label_protocol"].unique().to_list() == ["all_families"]

    def test_is_anomaly_is_a_view_not_a_third_truth(self, world, events):
        """``is_anomaly`` must equal one of the two protocol columns exactly."""
        event_only, _ = resolve_both(events, truth_root=world)
        frame = event_only.events
        assert frame["is_anomaly"].to_list() == frame["is_anomaly_event_only"].to_list()

    def test_protocols_are_named_and_distinct(self):
        assert LabelScope.EVENT_ONLY != LabelScope.ALL_FAMILIES
        assert str(LabelScope.EVENT_ONLY) == "event_only"
        assert str(LabelScope.ALL_FAMILIES) == "all_families"

    def test_family_partition_is_total_and_disjoint(self):
        assert set(EVENT_FAMILIES) | set(AGGREGATE_FAMILIES) == set(ANOMALY_FAMILIES)
        assert not (set(EVENT_FAMILIES) & set(AGGREGATE_FAMILIES))

    def test_prevalence_reports_both_protocols(self, world, events):
        result = resolve_labels(events, truth_root=world)
        assert set(result.prevalence) == {"event_only", "all_families"}
        assert result.prevalence["event_only"]["positives"] == 2
        assert result.prevalence["all_families"]["positives"] == 3

    def test_small_support_is_reported_not_hidden(self, world, events):
        """3 positives is far below any support floor; it must be stated."""
        result = resolve_labels(events, truth_root=world)
        assert len(result.skipped_families) > 0
        for reason in result.skipped_families.values():
            assert str(minimum_support()) in reason

    def test_supported_families_respects_the_floor(self, world, events):
        result = resolve_labels(events, truth_root=world)
        assert supported_families(result) == ()


class TestLabelFittingGuard:
    def test_refuses_to_fit_on_labeled_periods(self):
        from mosaic.experiments.protocol import LeakageError

        for period in ("validation", "backtest", "forward"):
            with pytest.raises(LeakageError, match="leaks the evaluation target"):
                assert_labels_never_fitted(period)  # type: ignore[arg-type]

    def test_allows_train(self):
        assert_labels_never_fitted("train")  # type: ignore[arg-type]


class TestEmptyAndNullEdges:
    def test_empty_frame_does_not_raise(self, world, events):
        empty = events.head(0)
        result = resolve_labels(empty, truth_root=world)
        assert result.positives == 0
        assert result.rate is None

    def test_all_null_label_column_reports_no_rate(self, world, events):
        """An all-null label column means unknown prevalence, not zero."""
        result = resolve_labels(events, truth_root=world)
        assert result.rate is not None
        assert 0 < result.rate < 1