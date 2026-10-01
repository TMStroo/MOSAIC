"""Dev harness: build the pipeline up to resolved events and cache it.

Not part of the package. Writes ``scratch/resolved.parquet`` so feature work can
iterate without re-running generation + ingestion + entity resolution.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, r"F:\projects\MOSAIC\src")

import polars as pl

ROOT = Path(r"F:\projects\MOSAIC")
SCRATCH = ROOT / "scratch"
DATASET = "world_a"


def build(force: bool = False) -> tuple[pl.DataFrame, dict]:
    cached = SCRATCH / "resolved.parquet"
    meta_path = SCRATCH / "resolved.meta.json"
    if cached.exists() and meta_path.exists() and not force:
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["events"] = pl.read_parquet(cached)
        return meta["events"], meta

    from mosaic.cleaning.events import clean_events
    from mosaic.entity_resolution.base import EntityResolver
    from mosaic.ingestion.pipeline import ingest_dataset

    SCRATCH.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ingest = ingest_dataset(
        ROOT / "data" / "synthetic",
        ROOT / "data" / "interim",
        dataset=DATASET,
    )
    t_ingest = time.perf_counter() - t0

    t0 = time.perf_counter()
    events, ledger = clean_events(
        ingest.raw_frame, dataset=DATASET, dataset_version=ingest.dataset_version
    )
    t_clean = time.perf_counter() - t0

    t0 = time.perf_counter()
    resolution = EntityResolver().run(events)
    t_resolve = time.perf_counter() - t0

    # attach resolved entity ids back onto the event frame.
    # The map is keyed on (entity_ref_norm, source_id): the same normalized ref
    # string can appear in more than one source with different damage, so the
    # source is part of the key.
    entity_map = resolution.entity_map
    resolved = events.drop([c for c in ("entity_id", "cluster_root") if c in events.columns]).join(
        entity_map.select("entity_ref_norm", "source_id", "entity_id", "cluster_root"),
        on=["entity_ref_norm", "source_id"],
        how="left",
    )

    # Deliberately NOT joining truth labels here. Feature construction must not be
    # able to see is_anomaly; evaluation joins labels in a separate frame.

    resolved.write_parquet(cached)
    meta = {
        "dataset": DATASET,
        "dataset_version": ingest.dataset_version,
        "rows_raw": ingest.row_count,
        "rows_clean": events.height,
        "rows_resolved": resolved.height,
        "sources": ingest.sources(),
        "n_entities_resolved": resolution.n_entities,
        "t_ingest": round(t_ingest, 2),
        "t_clean": round(t_clean, 2),
        "t_resolve": round(t_resolve, 2),
        "ledger": ledger.to_dict() if hasattr(ledger, "to_dict") else None,
        "resolution": resolution.to_dict(),
    }
    meta_path.write_text(
        __import__("json").dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return resolved, meta


if __name__ == "__main__":
    force = "--force" in sys.argv
    events, meta = build(force=force)
    print("rows:", events.height, "| entities:", meta["n_entities_resolved"], "| sources:", len(meta["sources"]))
    print("cols:", events.columns)
    if "is_anomaly" in events.columns:
        print("anomalies:", int(events["is_anomaly"].sum()))
    print("timings:", {k: meta[k] for k in ("t_ingest", "t_clean", "t_resolve")})
