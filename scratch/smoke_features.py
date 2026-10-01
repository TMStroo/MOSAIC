"""Smoke test for the feature pipeline with graph features disabled.

Run: ``.venv/Scripts/python.exe scratch/smoke_features.py``
"""

from __future__ import annotations

import sys
import time
import traceback

sys.path.insert(0, r"F:\projects\MOSAIC\src")
sys.path.insert(0, r"F:\projects\MOSAIC\scratch")


from build_resolved import build

from mosaic.experiments.protocol import build_split_plan, period_frames
from mosaic.features.compute import (
    FeatureConfig,
    assert_causal,
    compute_features,
    fit_state,
)

EXPECTED = [
    "count_1h", "count_6h", "count_24h", "count_168h",
    "rate_1h", "rate_6h", "rate_24h", "rate_168h",
    "entity_events_so_far",
    "mean_value_24h", "std_value_24h", "mean_value_7d", "std_value_7d",
    "ewma_value", "ewma_abs_residual",
    "hour_of_day", "day_of_week", "day_of_year", "seconds_since_prev",
    "value_robust_z", "rate_vs_base",
    "source_share_train", "transition_probability", "sources_in_hour",
]


def main() -> int:
    events, meta = build()
    print(f"events={events.height} entities={events['entity_id'].n_unique()} sources={len(meta['sources'])}")

    plan = build_split_plan(events)
    events = plan.assign(events)
    print("split counts:", plan.counts(events))

    train = period_frames(events, "train")
    print(f"train rows={train.height}")

    t0 = time.perf_counter()
    state = fit_state(train)
    print(f"fit_state: {time.perf_counter() - t0:.2f}s  {state.summary()}")

    config = FeatureConfig(include_graph=False)
    t0 = time.perf_counter()
    try:
        feats, registry, metrics = compute_features(events, state, config=config)
    except Exception:
        traceback.print_exc()
        return 1
    elapsed = time.perf_counter() - t0
    print(f"\ncompute_features: {elapsed:.2f}s for {events.height} rows -> {feats.height} rows")
    for m in metrics:
        print(f"  {m.stage:26s} {m.duration_s:7.3f}s  rss={m.peak_rss_mb}")

    cols = set(feats.columns)
    missing = [c for c in EXPECTED if c not in cols]
    print("\nmissing expected columns:", missing or "none")

    print("\n--- dtypes ---")
    for name in registry.names():
        if name in cols:
            print(f"  {name:34s} {feats.schema[name]}")

    print("\n--- null counts (should be 0 except by design) ---")
    nulls = {c: feats[c].null_count() for c in registry.names() if c in cols and feats[c].null_count()}
    print(nulls or "none")

    print("\n--- assert_causal on raw stream ---")
    try:
        assert_causal(feats, events)
        print("  assert_causal PASSED")
    except AssertionError as exc:
        print("  assert_causal FAILED:", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
