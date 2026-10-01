"""Regression tests for the temporal feature families.

Each test here corresponds to a real defect found by running the pipeline against
the synthetic world, not to a hypothetical. The docstring names the bug.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from mosaic.features.compute import (
    DAY,
    HOUR,
    FeatureConfig,
    _trailing_count,
    compute_features,
    fit_state,
)
from mosaic.experiments.protocol import build_split_plan

BASE = pl.datetime_range(
    start=datetime(2021, 1, 1), end=datetime(2021, 1, 3), interval="1h", eager=True
)


def _at(days: int, hours: int = 0, minutes: int = 0, seconds: int = 0) -> datetime:
    """Explicit datetime, not ``pl.datetime`` (which returns a python object and
    makes Polars infer an object column that cannot be sorted or compared)."""
    return datetime(2021, 1, 1) + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def _frame(rows: list[tuple[str, datetime, float]]) -> pl.DataFrame:
    """Minimal canonical frame: (entity, timestamp, value).

    Carries every column the feature families reference, so a family that starts
    demanding a new input fails loudly here instead of on the real dataset.
    """
    n = len(rows)
    return pl.DataFrame(
        {
            "event_id": [f"ev_{i}" for i in range(n)],
            "entity_id": [r[0] for r in rows],
            "timestamp": [r[1] for r in rows],
            "event_value": [r[2] for r in rows],
            "source_id": ["s1"] * n,
            "source_record_id": [f"rec_{i}" for i in range(n)],
            "event_type": ["access"] * n,
            "source_event_type": ["A"] * n,
            "location_id": ["Z001"] * n,
            "source_confidence": [1.0] * n,
            "related_entity_refs": [[] for _ in range(n)],
        },
        schema={
            "event_id": pl.Utf8,
            "entity_id": pl.Utf8,
            "timestamp": pl.Datetime("us"),
            "event_value": pl.Float64,
            "source_id": pl.Utf8,
            "source_record_id": pl.Utf8,
            "event_type": pl.Utf8,
            "source_event_type": pl.Utf8,
            "location_id": pl.Utf8,
            "source_confidence": pl.Float64,
            "related_entity_refs": pl.List(pl.Utf8),
        },
    ).sort(["entity_id", "timestamp", "event_id"])


def _brute_trailing(frame: pl.DataFrame, entity: str, when, window: int) -> int:
    """Deliberately naive reference implementation, independent of the vectorised one."""
    return sum(
        1
        for e, ts in frame.select("entity_id", "timestamp").iter_rows()
        if e == entity and when - timedelta(seconds=window) < ts <= when
    )


# --------------------------------------------------------------------------
# Bug 1: tied timestamps gave different counts depending on sort position.
# The real dataset has 5,296 exact (entity, timestamp) ties.
# --------------------------------------------------------------------------
def test_trailing_count_counts_all_tied_rows():
    ts = _at(0, 0, 0, 5)
    frame = _frame([("a", ts, 1.0), ("a", ts, 2.0)])
    out = frame.with_columns(
        _trailing_count(frame, "timestamp", "entity_id", 24 * HOUR).alias("count_24h")
    )
    assert out["count_24h"].to_list() == [2, 2]


# --------------------------------------------------------------------------
# Bug 2: composite-stride window search leaked across entity blocks when the
# window was wider than an entity's own time span.
# --------------------------------------------------------------------------
def test_trailing_count_does_not_cross_entity_blocks():
    # ent_a lives entirely AFTER ent_b; a naive global searchsorted lets ent_b's
    # early rows reach back into ent_a's block (and vice versa).
    frame = _frame(
        [
            ("ent_b", _at(0, 0, 0, 0), 1.0),
            ("ent_b", _at(0, 0, 0, 10), 1.0),
            ("ent_a", _at(4, 0, 0, 0), 1.0),
            ("ent_a", _at(4, 0, 0, 10), 1.0),
        ]
    )
    out = frame.with_columns(
        _trailing_count(frame, "timestamp", "entity_id", 7 * DAY).alias("count_7d")
    )
    # A 7-day window spans both entities' spans, but each entity must still see
    # only its own rows: ent_b's first event is the only one at or before its own
    # timestamp, so 1, then 2. ent_a likewise. A cross-block leak would show 3 or 4.
    assert out["count_7d"].to_list() == [1, 2, 1, 2]


@pytest.mark.parametrize("window", [1 * HOUR, 6 * HOUR, 24 * HOUR, 7 * DAY, 400 * DAY])
def test_trailing_count_matches_brute_force(window: int):
    """Randomised: the vectorised path must equal the naive filter, always."""
    rng = np.random.default_rng(20240921)
    rows: list[tuple[str, object, float]] = []
    for entity in ("a", "b", "c"):
        for _ in range(60):
            offset = int(rng.integers(0, 20 * DAY))
            rows.append(
                (entity, BASE[0] + timedelta(seconds=int(offset)), float(rng.normal()))
            )
    frame = _frame(rows)
    out = frame.with_columns(
        _trailing_count(frame, "timestamp", "entity_id", window).alias("c")
    )
    for entity, when, value in out.select("entity_id", "timestamp", "c").iter_rows():
        assert int(value) == _brute_trailing(frame, entity, when, window), (
            f"window={window} entity={entity} at {when}"
        )


# --------------------------------------------------------------------------
# Bug 3: `.alias()` bound tighter than `/`, so `rate_vs_base` was naming the
# clipped baseline rather than the ratio.
# --------------------------------------------------------------------------
def test_rate_vs_base_is_a_ratio_not_the_baseline():
    frame = _frame(
        [
            ("a", _at(0, 0, 0, 0), 1.0),
            ("a", _at(0, 1, 0, 0), 1.0),
            ("a", _at(0, 2, 0, 0), 1.0),
        ]
    )
    state = fit_state(frame)
    feats, registry, _ = compute_features(frame, state, config=FeatureConfig(include_graph=False))
    assert "rate_vs_base" in feats.columns
    assert "rate_vs_base" in registry.specs
    last = feats.filter(pl.col("timestamp") == _at(0, 2, 0, 0))
    baseline = state.entity_baselines["a"]["events_per_day"]
    expected = feats.filter(pl.col("timestamp") == _at(0, 2, 0, 0))["rate_24h"].item() / baseline
    assert last["rate_vs_base"].item() == pytest.approx(expected)
    # the regression this guards: the column used to hold the clipped baseline
    assert last["rate_vs_base"].item() != pytest.approx(baseline)


# --------------------------------------------------------------------------
# Bug 4: sequential feature construction referenced a column created in the
# same with_columns call.
# --------------------------------------------------------------------------
def test_unseen_source_features_are_sequential_not_aliased_in_one_call():
    frame = _frame([("a", BASE[0], 1.0), ("a", BASE[1], 1.0)])
    state = fit_state(frame)
    feats, _, _ = compute_features(frame, state, config=FeatureConfig(include_graph=False))
    assert "source_unseen_in_train" in feats.columns
    assert "event_type_unseen_in_train" in feats.columns


# --------------------------------------------------------------------------
# Bug 5: cross-entity contamination via an ungrouped cumulative sum.
# --------------------------------------------------------------------------
def test_rolling_statistics_do_not_leak_across_entities():
    """ent_b's running mean must not include ent_a's magnitudes."""
    frame = _frame(
        [
            ("ent_a", _at(0, 0, 0, 0), 100.0),
            ("ent_a", _at(0, 1, 0, 0), 100.0),
            ("ent_b", _at(0, 0, 0, 0), 1.0),
            ("ent_b", _at(0, 1, 0, 0), 1.0),
        ]
    )
    state = fit_state(frame)
    feats, _, _ = compute_features(frame, state, config=FeatureConfig(include_graph=False))
    ent_b = feats.filter(pl.col("entity_id") == "ent_b").sort("timestamp")
    # ent_b's mean at its second event sees only its own single prior value (1.0)
    assert ent_b["mean_value_24h"].to_list()[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Bug 6: `_safe_div` cast its own Expr arguments in the scalar branch.
# --------------------------------------------------------------------------
def test_rate_features_are_finite_for_a_single_event_entity():
    frame = _frame([("solo", BASE[0], 1.0)])
    state = fit_state(frame)
    feats, _, _ = compute_features(frame, state, config=FeatureConfig(include_graph=False))
    for name in ("rate_1h", "rate_6h", "rate_24h", "rate_168h"):
        assert np.isfinite(feats[name].to_numpy()).all(), name


# --------------------------------------------------------------------------
# Behavioural baselines must be fit on training rows only.
# --------------------------------------------------------------------------
def test_behavioral_baseline_is_unaffected_by_forward_rows():
    """Changing forward magnitudes must not move the fitted baseline at all."""
    early = [
        ("a", _at(0, 0, 0, 0), 10.0),
        ("a", _at(0, 12, 0, 0), 12.0),
        ("a", _at(1, 0, 0, 0), 14.0),
    ]
    late_low = [("a", _at(19, 0, 0, 0), -999.0)]
    late_high = [("a", _at(19, 0, 0, 0), 99999.0)]
    baselines = []
    for late in (late_low, late_high):
        frame = _frame(early + late)
        # train = everything before 2021-01-10; the 2021-01-20 row is forward
        state = fit_state(frame.filter(pl.col("timestamp") < _at(9)))
        baselines.append(state.entity_baselines["a"]["value_median"])
    assert baselines[0] == baselines[1], "fitted baseline moved with forward data"


# --------------------------------------------------------------------------
# Determinism: same input, same output.
# --------------------------------------------------------------------------
def test_feature_names_and_values_are_deterministic():
    frame = _frame(
        [("a", BASE[i], float(i)) for i in range(12)] + [("b", BASE[i], float(i)) for i in range(12)]
    )
    state = fit_state(frame)
    config = FeatureConfig(include_graph=False)
    a, reg_a, _ = compute_features(frame, state, config=config)
    b, reg_b, _ = compute_features(frame, state, config=config)
    assert reg_a.names() == reg_b.names()
    assert reg_a.fingerprint() == reg_b.fingerprint()
    for name in reg_a.names():
        if name in a.columns and a.schema[name].is_float():
            np.testing.assert_allclose(a[name].to_numpy(), b[name].to_numpy())


def test_split_plan_boundaries_are_ordered():
    frame = _frame([("a", BASE[i], 1.0) for i in range(48)])
    plan = build_split_plan(frame)
    bounds = plan.bounds
    assert bounds["train"].end <= bounds["validation"].start
    assert bounds["validation"].end <= bounds["backtest"].start
    assert bounds["backtest"].end <= bounds["forward"].start
