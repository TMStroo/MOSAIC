"""Data-quality report: one JSON + one Markdown per dataset.

The report is generated *from the data*, never from config claims, and it is the
input to the Data Quality page and to the dataset card in the research report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from mosaic.utils.io import write_json
from mosaic.validation.checks import ValidationReport


@dataclass
class DataQualityReport:
    """Machine-readable quality summary for one dataset version."""

    dataset: str
    dataset_version: str
    generated_at: str
    row_counts: dict[str, Any] = field(default_factory=dict)
    time_coverage: dict[str, Any] = field(default_factory=dict)
    missingness: dict[str, dict[str, float]] = field(default_factory=dict)
    duplicates: dict[str, int] = field(default_factory=dict)
    invalid: dict[str, int] = field(default_factory=dict)
    cardinalities: dict[str, int] = field(default_factory=dict)
    coverage_by_day: dict[str, list[int]] = field(default_factory=dict)
    volumes: dict[str, list[int]] = field(default_factory=dict)
    distributions: dict[str, dict[str, float]] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    source_drift: dict[str, Any] = field(default_factory=dict)
    markdown_path: str | None = None
    json_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "generated_at": self.generated_at,
            "row_counts": self.row_counts,
            "time_coverage": self.time_coverage,
            "missingness": self.missingness,
            "duplicates": self.duplicates,
            "invalid": self.invalid,
            "cardinalities": self.cardinalities,
            "coverage_by_day": self.coverage_by_day,
            "volumes": self.volumes,
            "distributions": self.distributions,
            "validation": self.validation,
            "source_drift": self.source_drift,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
        }

    def to_markdown(self) -> str:
        v = self.validation
        lines = [
            f"# Data Quality Report - {self.dataset}",
            "",
            f"- dataset version: `{self.dataset_version}`",
            f"- generated: {self.generated_at}",
            f"- rows: **{self.row_counts.get('total', 0):,}** across "
            f"**{self.row_counts.get('sources', 0)}** sources",
            f"- validation: **{v.get('worst_status', 'PASS')}** "
            f"(quality score {v.get('quality_score', 1.0)})",
            "",
            "## Per source",
            "",
            "| source | rows | share | first day | last day | dupes | invalid |",
            "|---|---:|---:|---|---|---:|---:|",
        ]
        per_source = self.row_counts.get("per_source", {})
        for source_id, rows in sorted(per_source.items()):
            cov = self.time_coverage.get(source_id, {})
            lines.append(
                f"| `{source_id}` | {rows:,} | {rows / max(1, self.row_counts.get('total', 1)):.1%} "
                f"| {cov.get('first', '-')} | {cov.get('last', '-')} "
                f"| {self.duplicates.get(source_id, 0):,} | {self.invalid.get(source_id, 0):,} |"
            )
        lines += ["", "## Validation checks", ""]
        counts = v.get("counts", {})
        lines.append(
            f"PASS {counts.get('PASS', 0)} / WARN {counts.get('WARN', 0)} / FAIL {counts.get('FAIL', 0)}"
        )
        lines += ["", "| source | check | expected | observed | status | rows |", "|---|---|---|---|---|---:|"]
        for row in v.get("results", []):
            if row["status"] == "PASS":
                continue
            lines.append(
                f"| `{row['source_id']}` | {row['check']} | {row['expected']} "
                f"| {row['observed']} | **{row['status']}** | {row['rows_affected']:,} |"
            )
        lines += ["", "## Source drift", ""]
        for source_id, drift in sorted(self.source_drift.items()):
            lines.append(f"- `{source_id}`: {drift}")
        lines.append("")
        return "\n".join(lines)


def build_quality_report(
    frame: pl.DataFrame,
    validation: ValidationReport,
    *,
    out_dir: Path | str,
    timestamp_column: str = "timestamp",
    entity_column: str = "entity_id",
) -> DataQualityReport:
    """Assemble and persist the quality report for a cleaned event frame."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = frame.height

    per_source_rows: dict[str, int] = {}
    per_source_cov: dict[str, dict[str, str]] = {}
    per_source_missing: dict[str, dict[str, float]] = {}
    per_source_dupes: dict[str, int] = {}
    per_source_invalid: dict[str, int] = {}
    per_source_card: dict[str, int] = {}
    per_source_daily: dict[str, list[int]] = {}

    lookups = {
        r.check: r for r in validation.results if r.check.startswith("duplicate_records")
    }

    numeric_cols = [c for c in ("event_value", "source_confidence") if c in frame.columns]
    distributions: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        series = frame[col].cast(pl.Float64, strict=False).drop_nulls()
        if series.is_empty():
            continue
        distributions[col] = {
            "min": round(float(series.min()), 4),
            "p25": round(float(series.quantile(0.25)), 4),
            "median": round(float(series.median()), 4),
            "p75": round(float(series.quantile(0.75)), 4),
            "p99": round(float(series.quantile(0.99)), 4),
            "max": round(float(series.max()), 4),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std() or 0.0), 4),
        }

    for source_id in sorted(frame["source_id"].unique().to_list()):
        part = frame.filter(pl.col("source_id") == source_id)
        per_source_rows[source_id] = part.height
        if timestamp_column in part.columns and part.height:
            ts = part[timestamp_column].drop_nulls()
            per_source_cov[source_id] = {
                "first": str(ts.min()),
                "last": str(ts.max()),
                "days": int((ts.max().date() - ts.min().date()).days) + 1 if ts.len() else 0,
            }
            daily = (
                part.group_by(timestamp_column)
                .agg(pl.len().alias("n"))
                .sort(timestamp_column)
            )
            per_source_daily[source_id] = [
                int(n) for n in daily["n"].to_list()[:400]
            ]
        else:
            per_source_daily[source_id] = []
        per_source_missing[source_id] = {
            col: round(float(part[col].null_count() / max(1, part.height)), 4)
            for col in ("event_value", "event_type", "location_id", "entity_id")
            if col in part.columns
        }
        dup_check = lookups.get("duplicate_records")
        per_source_dupes[source_id] = int(
            next((r.rows_affected for r in validation.results if r.source_id == source_id and r.check == "duplicate_records"), 0)
        )
        del dup_check
        per_source_invalid[source_id] = int(
            next(
                (
                    r.rows_affected
                    for r in validation.results
                    if r.source_id == source_id and r.check == "out_of_range_values"
                ),
                0,
            )
        )
        if "event_type" in part.columns:
            per_source_card[source_id] = int(part["event_type"].n_unique())

    # source drift: coverage changes over the life of the dataset
    drift: dict[str, Any] = {}
    for source_id, counts in per_source_daily.items():
        if len(counts) < 8:
            continue
        head = sum(counts[: len(counts) // 4]) / max(1, len(counts) // 4)
        tail = sum(counts[-len(counts) // 4 :]) / max(1, len(counts) // 4)
        zero_days = sum(1 for c in counts if c == 0)
        drift[source_id] = (
            f"first-quarter mean {head:.0f} events/day vs last-quarter {tail:.0f} "
            f"(ratio {tail / head if head else float('nan'):.2f}), {zero_days} empty days"
        )

    report = DataQualityReport(
        dataset=validation.dataset,
        dataset_version=validation.dataset_version,
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        row_counts={"total": total, "sources": len(per_source_rows), "per_source": per_source_rows},
        time_coverage=per_source_cov,
        missingness=per_source_missing,
        duplicates=per_source_dupes,
        invalid=per_source_invalid,
        cardinalities=per_source_card,
        coverage_by_day={k: len(v) for k, v in per_source_daily.items()},
        volumes=per_source_daily,
        distributions=distributions,
        validation=validation.to_dict(),
        source_drift=drift,
    )
    json_path = out_dir / f"{validation.dataset}_quality.json"
    md_path = out_dir / f"{validation.dataset}_quality.md"
    report.json_path = str(write_json(json_path, report.to_dict())).replace("\\", "/")
    report.markdown_path = str(md_path).replace("\\", "/")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return report
