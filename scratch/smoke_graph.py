"""End-to-end smoke test: features WITH causal graph features on the real world.

Run: ``.venv/Scripts/python.exe scratch/smoke_graph.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, r"F:\projects\MOSAIC\src")
sys.path.insert(0, r"F:\projects\MOSAIC\scratch")

import polars as pl  # noqa: E402

from build_resolved import build  # noqa: E402
from mosaic.experiments.protocol import build_split_plan, period_frames  # noqa: E402
from mosaic.features.compute import FeatureConfig, compute_features, fit_state  # noqa: E402


def relations_from_events(events: pl.DataFrame) -> pl.DataFrame:
    """Event-derived interactions: a row carrying peer references implies an edge."""
    # related_entity_refs is a scalar peer string in this dataset ("" when absent)
    pairs = (
        events.filter(pl.col("related_entity_refs").is_not_null() & (pl.col("related_entity_refs") != ""))
        .select(
            pl.col("entity_id"),
            pl.col("related_entity_refs").alias("related_entity_id"),
            pl.col("timestamp").alias("first_seen"),
        )
        .filter(pl.col("related_entity_id") != pl.col("entity_id"))
        .unique()
    )
    print(f"relations: {pairs.height} edges from {pairs['entity_id'].n_unique()} entities")
    return pairs


def main() -> int:
    events, meta = build()
    print(f"events={events.height} entities={events['entity_id'].n_unique()}")

    plan = build_split_plan(events)
    events = plan.assign(events)
    train = period_frames(events, "train")
    state = fit_state(train)
    print(f"fit_state on train ({train.height} rows): {state.summary()}")

    relations = relations_from_events(events)

    config = FeatureConfig(include_graph=True)
    t0 = time.perf_counter()
    try:
        feats, registry, metrics = compute_features(
            events, state, config=config, relations=relations
        )
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"FAILED: {exc}")
        return 1
    elapsed = time.perf_counter() - t0
    print(f"\ncompute_features(graph=True): {elapsed:.1f}s for {events.height} rows")
    for m in metrics:
        print(f"  {m.stage:28s} {m.duration_s:8.3f}s rows={m.rows}")

    graph_cols = [c for c in feats.columns if c.startswith("graph_")]
    print(f"\ngraph columns produced ({len(graph_cols)}): {graph_cols}")
    for c in sorted(graph_cols):
        s = feats[c]
        print(f"  {c:32s} {s.dtype}  nulls={s.null_count():7d}  "
              f"min={s.min()} max={s.max()}")

    feats.select(["event_id", "entity_id", "timestamp", "period", *graph_cols]).write_parquet(
        Path(r"F:\projects\MOSAIC\scratch") / "features_with_graph.parquet"
    )
    print("\nwrote scratch/features_with_graph.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())