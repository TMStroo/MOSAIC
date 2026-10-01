"""End-to-end smoke test: features WITH causal graph features on the real world.

Run: ``.venv/Scripts/python.exe scratch/smoke_graph.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, r"F:\projects\MOSAIC\src")
sys.path.insert(0, r"F:\projects\MOSAIC\scratch")

import polars as pl
from build_resolved import build

from mosaic.experiments.protocol import build_split_plan, period_frames
from mosaic.features.compute import FeatureConfig, compute_features, fit_state


def relations_from_events(events: pl.DataFrame) -> pl.DataFrame:
    """Event-derived interactions.

    ``related_entity_refs`` already holds canonical entity ids by the time this is
    called (see the canonicalisation step in ``main``), so an edge is simply a
    (subject, peer, timestamp) triple.
    """
    pairs = (
        events.filter(
            pl.col("related_entity_refs").is_not_null()
            & (pl.col("related_entity_refs") != "")
        )
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

    # Canonicalise the PEER reference too. The event frame keeps raw source
    # references for lineage, but graph features need canonical ids on both ends of
    # an edge; leaving one side raw makes every peer lookup miss (overlap was exactly
    # 0 between raw peer refs and canonical ids).
    from mosaic.cleaning.events import clean_events as _clean
    from mosaic.entity_resolution.base import EntityResolver as _Res
    from mosaic.ingestion.pipeline import ingest_dataset as _ing

    _ingest = _ing(Path(r"F:\projects\MOSAIC\data\synthetic"),
                   Path(r"F:\projects\MOSAIC\data\interim"), dataset="world_a")
    _cleaned, _ledger = _clean(_ingest.raw_frame, dataset="world_a",
                               dataset_version=_ingest.dataset_version)
    _emap = _Res().run(_cleaned).entity_map
    _peer_lookup = _emap.group_by("entity_ref_norm").agg(
        pl.col("entity_id").mode().first().alias("_peer_canonical")
    )
    events = events.join(
        _peer_lookup, left_on="related_entity_refs", right_on="entity_ref_norm", how="left"
    ).with_columns(
        pl.col("_peer_canonical").alias("related_entity_refs")
    ).drop(["_peer_canonical", "entity_ref_norm"])
    resolved_peers = int(events.filter(pl.col("related_entity_refs").is_not_null()).height)
    print(f"peer refs canonicalised: {resolved_peers}")

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
    print("\n--- per-period variation (is each measure actually alive?) ---")
    for c in sorted(graph_cols):
        per = (
            feats.group_by("period")
            .agg(
                pl.col(c).mean().round(6).alias("mean"),
                pl.col(c).max().alias("max"),
                pl.len().alias("rows"),
            )
            .sort("period")
        )
        means = {r["period"]: r["mean"] for r in per.to_dicts()}
        print(f"  {c:32s} means by period: {means}")

    feats.select(["event_id", "entity_id", "timestamp", "period", *graph_cols]).write_parquet(
        Path(r"F:\projects\MOSAIC\scratch") / "features_with_graph.parquet"
    )
    print("\nwrote scratch/features_with_graph.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
