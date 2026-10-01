"""Ingestion pipeline: adapters -> raw Parquet + a dataset manifest.

Responsibilities, in order:

1. Ask each adapter to discover/load/validate/normalize its source.
2. Write per-source raw Parquet to ``data/interim/<dataset>/raw/<source_id>/``.
3. Concatenate into ``data/interim/<dataset>/events_raw.parquet`` with lineage.
4. Emit ``data/manifests/<dataset>.json``: per-source checksums, schema hashes,
   row counts, time ranges, and the *measured* limitations of this run.

The manifest's ``version`` is a content hash, so an experiment can pin a dataset
version and ``mosaic reproduce`` can detect drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from mosaic.ingestion.synthetic.adapters import SyntheticMultiSourceAdapter
from mosaic.ingestion.synthetic.builder import load_manifest
from mosaic.schema.ids import file_checksum, stable_hash
from mosaic.utils.io import write_json, write_parquet
from mosaic.utils.timing import StageMetrics, stage_timer

LOGGER = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Everything downstream stages and the API need from one ingestion run."""

    dataset: str
    dataset_version: str
    raw_frame: pl.DataFrame
    manifest: dict[str, Any]
    source_manifests: list[dict[str, Any]] = field(default_factory=list)
    schema_checks: list[dict[str, Any]] = field(default_factory=list)
    stage_metrics: list[StageMetrics] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return self.raw_frame.height

    def sources(self) -> list[str]:
        return sorted(self.raw_frame["source_id"].unique().to_list())


def ingest_dataset(
    synthetic_root: Path | str,
    interim_root: Path | str,
    *,
    dataset: str = "world_a",
    source_version: str = "1.0.0",
    only_sources: list[str] | None = None,
) -> IngestResult:
    """Run the full ingestion for one dataset and persist raw + manifest."""
    synthetic_root = Path(synthetic_root)
    interim_root = Path(interim_root)
    target = interim_root / dataset
    target.mkdir(parents=True, exist_ok=True)

    facade = SyntheticMultiSourceAdapter(synthetic_root, dataset=dataset, source_version=source_version)
    adapters = facade.adapters()
    if only_sources:
        wanted = set(only_sources)
        adapters = [a for a in adapters if a.source_id in wanted]
    if not adapters:
        raise FileNotFoundError(
            f"dataset {dataset!r} has no readable sources; generate it first "
            f"(`python -m mosaic data generate --profile {dataset}`)"
        )

    metrics: list[StageMetrics] = []
    frames: list[pl.DataFrame] = []
    source_manifests: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    with stage_timer("ingest_total", metrics) as sw:
        for adapter in adapters:
            with stage_timer(f"ingest:{adapter.source_id}", metrics) as source_sw:
                result = adapter.run()
                normalized = adapter.normalize(result.frame)
                write_parquet(target / "raw" / adapter.source_id / "part-000.parquet", normalized)
            if source_sw.metrics is not None:
                source_sw.metrics.rows = result.row_count
            frames.append(normalized)
            source_manifests.append(
                {
                    "source_id": result.source_id,
                    "source_name": adapter.dataset_slug,
                    "source_version": source_version,
                    "rows_in": result.row_count,
                    "rows_normalized": normalized.height,
                    "column_count": result.metadata.column_count,
                    "schema_hash": result.metadata.schema_hash,
                    "data_checksum": result.metadata.data_checksum,
                    "time_range": (
                        [result.metadata.time_range[0], result.metadata.time_range[1]]
                        if result.metadata.time_range
                        else None
                    ),
                    "known_limitations": list(adapter.known_limitations),
                    "raw_path": f"{dataset}/raw/{adapter.source_id}/part-000.parquet",
                }
            )
            checks.extend(
                {"source_id": result.source_id, **check.to_dict()} for check in result.schema_checks
            )
            LOGGER.info(
                "ingested %s: %d rows", adapter.source_id, result.row_count, extra={"stage": adapter.source_id}
            )

        raw = pl.concat(frames, how="diagonal_relaxed", rechunk=True)
        raw = raw.with_columns(
            pl.lit(dataset, dtype=pl.Utf8).alias("dataset_name"),
            pl.lit("raw", dtype=pl.Utf8).alias("transform_stage"),
        )
        raw_path = write_parquet(target / "events_raw.parquet", raw)
    if sw.metrics is not None:
        sw.metrics.rows = raw.height

    version = _dataset_version(raw, source_manifests)
    synthetic_manifest: dict[str, Any] = {}
    try:
        synthetic_manifest = load_manifest(synthetic_root, dataset)
    except FileNotFoundError:  # pragma: no cover - only when ingesting external data
        LOGGER.warning("no synthetic manifest for %s; manifest will omit generator spec", dataset)

    manifest = {
        "dataset": dataset,
        "dataset_version": version,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "row_count": raw.height,
        "column_count": raw.width,
        "columns": {c: str(d) for c, d in raw.schema.items()},
        "schema_hash": stable_hash({c: str(d) for c, d in sorted(raw.schema.items())}, length=32),
        "data_checksum": file_checksum(raw_path),
        "sources": source_manifests,
        "time_range": (
            [str(raw["timestamp_raw"].drop_nulls().min()), str(raw["timestamp_raw"].drop_nulls().max())]
            if raw["timestamp_raw"].drop_nulls().len()
            else None
        ),
        "generator": {
            "spec": synthetic_manifest.get("spec"),
            "spec_fingerprint": synthetic_manifest.get("spec_fingerprint"),
            "truth_summary": synthetic_manifest.get("truth_summary"),
            "entity_link_summary": synthetic_manifest.get("entity_link_summary"),
        },
        "raw_path": str(raw_path).replace("\\", "/"),
        "schema_checks": checks,
        "stage_metrics": [m.to_dict() for m in metrics],
    }
    manifest_path = write_json(
        Path(interim_root).parent / "manifests" / f"{dataset}.json", manifest
    )
    manifest["manifest_path"] = str(manifest_path).replace("\\", "/")

    return IngestResult(
        dataset=dataset,
        dataset_version=version,
        raw_frame=raw,
        manifest=manifest,
        source_manifests=source_manifests,
        schema_checks=checks,
        stage_metrics=metrics,
    )


def _dataset_version(raw: pl.DataFrame, sources: list[dict[str, Any]]) -> str:
    """Content-addressed dataset version.

    Uses per-source row counts and checksums rather than the whole file hash, so
    re-running ingestion on identical data yields an identical version (idempotence)
    while any real content change moves it.
    """
    return stable_hash(
        {
            "rows": raw.height,
            "sources": sorted((s["source_id"], s["rows_in"], s["data_checksum"]) for s in sources),
        },
        length=20,
    )
