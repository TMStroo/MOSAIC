"""Concrete validators for the raw ingested frame.

The raw frame is what adapters emit: string timestamps, per-source columns, no
cross-source normalization. These checks run before cleaning, so their job is to
quantify how broken the input is, per source.

Check list (each one answers a question a reviewer would ask):

* ``row_count``            - is anything there at all?
* ``required_columns``     - did the adapter's contract hold?
* ``timestamp_parseable``  - how many timestamps survive a strict parse?
* ``timestamp_ordered``    - is each source's stream ordered by time?
* ``duplicate_records``    - exact and near-duplicate volume
* ``null_ratio``           - per required column, per source
* ``invalid_event_type``   - did category mapping leave holes?
* ``out_of_range_values``  - negative / non-finite magnitudes
* ``unknown_source_ref``   - do actor references look like the source's id style?
* ``referential_integrity``- are peer references present in the same source?
* ``source_completeness``  - per-source daily volume: gaps and outages
* ``event_type_vocabulary``- cardinality of each source's categories
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import polars as pl

from mosaic.ingestion.synthetic.generator import SOURCE_ID_STYLE
from mosaic.schema.canonical import EventType
from mosaic.validation.checks import Status, ValidationReport

#: Tolerances are configurable through the cleaning profile; these are defaults.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"
NULL_WARN_RATIO = 0.10
OUTAGE_GAP_DAYS = 2


def _status_for_ratio(ratio: float, warn: float, fail: float) -> Status:
    if ratio >= fail:
        return Status.FAIL
    if ratio >= warn:
        return Status.WARN
    return Status.PASS


def validate_raw(
    raw: pl.DataFrame,
    *,
    dataset: str,
    dataset_version: str,
    adapter_columns: dict[str, dict[str, str]] | None = None,
    null_warn_ratio: float = NULL_WARN_RATIO,
) -> ValidationReport:
    """Run every raw-frame check and return a populated report."""
    report = ValidationReport(dataset, dataset_version)
    sources = sorted(raw["source_id"].unique().to_list()) if raw.height else []

    report.add(
        "ALL",
        "row_count",
        expected="> 0",
        observed=raw.height,
        status=Status.PASS if raw.height else Status.FAIL,
        rows_affected=raw.height,
    )
    if not raw.height:
        return report

    required = ("source_id", "source_record_id", "timestamp_raw", "entity_ref_raw", "event_type_raw")

    for source_id in sources:
        part = raw.filter(pl.col("source_id") == source_id)
        _check_required_columns(report, source_id, part, required, adapter_columns)
        parsed = _check_timestamps(report, source_id, part)
        _check_nulls(report, source_id, part, required, null_warn_ratio)
        _check_duplicates(report, source_id, part)
        _check_event_types(report, source_id, part)
        _check_value_ranges(report, source_id, part)
        _check_id_style(report, source_id, part)
        _check_referential_integrity(report, source_id, part)
        if parsed is not None:
            _check_ordering(report, source_id, part, parsed)
            _check_completeness(report, source_id, part, parsed)

    return report


def _check_required_columns(
    report: ValidationReport,
    source_id: str,
    part: pl.DataFrame,
    required: tuple[str, ...],
    adapter_columns: dict[str, dict[str, str]] | None,
) -> None:
    missing = [c for c in required if c not in part.columns]
    report.add(
        source_id,
        "required_columns",
        expected=list(required),
        observed="all present" if not missing else f"missing {missing}",
        status=Status.PASS if not missing else Status.FAIL,
        rows_affected=0,
        adapter_columns=(adapter_columns or {}).get(source_id),
    )


def _check_timestamps(
    report: ValidationReport, source_id: str, part: pl.DataFrame
) -> pl.Series | None:
    """Strict ISO parse. Blanks, suffixes and day/month swaps all fail here."""
    total = part.height
    if total == 0:
        return None
    try:
        parsed = (
            part.select(
                pl.col("timestamp_raw")
                .str.strip_chars()
                .str.to_datetime(strict=False, format=TIMESTAMP_FORMAT)
                .alias("ts")
            )
            .get_column("ts")
        )
    except Exception as exc:  # pragma: no cover - polars version guard
        report.add(
            source_id,
            "timestamp_parseable",
            expected=f"format {TIMESTAMP_FORMAT}",
            observed=f"parse error: {exc}",
            status=Status.FAIL,
            rows_affected=total,
        )
        return None
    ok = parsed.is_not_null().sum()
    bad = total - int(ok)
    report.add(
        source_id,
        "timestamp_parseable",
        expected=f">= {99.5}% parseable as {TIMESTAMP_FORMAT}",
        observed=f"{ok}/{total} ({ok / total:.4%})",
        status=_status_for_ratio(bad / total, 0.0005, 0.02),
        rows_affected=int(bad),
        parse_failures=int(bad),
    )
    return parsed


def _check_ordering(
    report: ValidationReport, source_id: str, part: pl.DataFrame, parsed: pl.Series
) -> None:
    """Ingestion should preserve file order; a non-monotonic stream means a
    concatenated/sharded file, which the feature layer must not assume."""
    series = parsed.drop_nulls()
    if series.len() < 2:
        return
    descending = int((series.diff().dt.total_milliseconds() < 0).sum())
    report.add(
        source_id,
        "timestamp_ordered",
        expected="monotonically non-decreasing",
        observed=f"{descending} backward steps of {series.len()}",
        status=Status.PASS if descending == 0 else Status.WARN,
        severity="warn",
        rows_affected=descending,
    )


def _check_nulls(
    report: ValidationReport,
    source_id: str,
    part: pl.DataFrame,
    required: tuple[str, ...],
    warn_ratio: float,
) -> None:
    total = max(1, part.height)
    for column in required:
        if column not in part.columns:
            continue
        nulls = int(part[column].null_count())
        ratio = nulls / total
        report.add(
            source_id,
            f"null_ratio::{column}",
            expected=f"< {warn_ratio:.0%}",
            observed=f"{nulls} ({ratio:.4%})",
            status=_status_for_ratio(ratio, warn_ratio, 0.5),
            severity="warn",
            rows_affected=nulls,
        )


def _check_duplicates(report: ValidationReport, source_id: str, part: pl.DataFrame) -> None:
    total = part.height
    if total == 0:
        return
    exact = int(total - part.select(pl.struct(part.columns).n_unique()).item())
    normalized = pl.col("source_record_id").str.replace("-Lr", "-L", literal=True)
    latent_distinct = int(part.select(normalized.n_unique()).item())
    near = total - latent_distinct - exact
    report.add(
        source_id,
        "duplicate_records",
        expected="exact duplicates reported, not dropped",
        observed=f"exact={exact}, near_duplicate={max(0, near)}, rows={total}",
        status=Status.PASS if exact == 0 else Status.WARN,
        severity="warn",
        rows_affected=exact + max(0, near),
        exact_duplicates=exact,
        near_duplicates=max(0, near),
    )


def _check_event_types(report: ValidationReport, source_id: str, part: pl.DataFrame) -> None:
    total = max(1, part.height)
    unmapped = int((part["event_type_mapped"] == "other").sum()) if "event_type_mapped" in part.columns else 0
    report.add(
        source_id,
        "invalid_event_type",
        expected="all native categories map to a canonical type",
        observed=f"{unmapped} unmapped of {total} ({unmapped / total:.4%})",
        status=Status.PASS if unmapped == 0 else Status.WARN,
        severity="warn",
        rows_affected=unmapped,
    )
    if "event_type_raw" in part.columns:
        cardinality = part["event_type_raw"].n_unique()
        report.add(
            source_id,
            "category_cardinality",
            expected="<= 20 distinct native categories",
            observed=cardinality,
            status=Status.PASS if cardinality <= 20 else Status.WARN,
            severity="warn",
            categories=sorted(
                part["event_type_raw"].drop_nulls().unique().to_list()
            )[:20],
        )


def _check_value_ranges(report: ValidationReport, source_id: str, part: pl.DataFrame) -> None:
    """Range check on magnitudes.

    A null magnitude is *missing data*, not a data-quality failure, so it is
    counted by ``null_ratio`` and excluded here. Only genuinely impossible values
    (negative, NaN, inf) count against the source: sources declare 3-12% missing
    values by design, and treating those as errors would make every report FAIL.
    """
    if "event_value" not in part.columns:
        return
    values = part["event_value"]
    total = max(1, part.height)
    negative = int((values < 0).sum())
    non_finite = int(((~values.is_finite()) & values.is_not_null()).sum()) if values.dtype.is_float() else 0
    missing = int(values.null_count())
    bad = negative + non_finite
    report.add(
        source_id,
        "out_of_range_values",
        expected="event_value >= 0 and finite",
        observed=f"negative={negative}, non_finite={non_finite}, missing={missing} of {total}",
        status=_status_for_ratio(bad / total, 0.005, 0.05),
        rows_affected=bad,
        negative=negative,
        non_finite=non_finite,
        missing=missing,
    )


def _check_id_style(report: ValidationReport, source_id: str, part: pl.DataFrame) -> None:
    """Do actor references still look like this source's identifier style?

    A low match rate is the early signal that entity resolution will be hard, and
    it is measured *before* any resolution attempt so the two can be compared.
    """
    style = SOURCE_ID_STYLE.get(source_id)
    if style is None or "entity_ref_raw" not in part.columns:
        return
    values = part["entity_ref_raw"].drop_nulls()
    total = max(1, values.len())
    if style == "site::{slug}":
        expr = pl.col("entity_ref_raw").str.starts_with("site::")
    elif style.startswith("entity_"):
        expr = pl.col("entity_ref_raw").str.starts_with("entity_")
    elif style.startswith("ENT-"):
        expr = pl.col("entity_ref_raw").str.starts_with("ENT-")
    else:
        expr = pl.col("entity_ref_raw").str.contains(r"^\d{5}")
    conforming = int(part.select(expr.fill_null(False).sum()).item())
    report.add(
        source_id,
        "source_ref_style",
        expected=f"identifiers matching {style!r}",
        observed=f"{conforming}/{total} ({conforming / total:.4%}) conforming",
        status=Status.PASS if conforming / total > 0.9 else Status.WARN,
        severity="warn",
        rows_affected=total - conforming,
        style=style,
    )


def _check_referential_integrity(report: ValidationReport, source_id: str, part: pl.DataFrame) -> None:
    """Peer references should resolve inside the same source's vocabulary."""
    if "peer_ref_raw" not in part.columns:
        return
    peers = part["peer_ref_raw"].drop_nulls()
    peers = peers.filter(peers.str.len_chars() > 0)
    if peers.is_empty():
        return
    known = part["entity_ref_raw"].drop_nulls().unique()
    unresolved = int((~peers.is_in(known)).sum())
    ratio = unresolved / max(1, peers.len())
    report.add(
        source_id,
        "referential_integrity",
        expected="peer refs exist among this source's own refs",
        observed=f"{unresolved}/{peers.len()} ({ratio:.4%}) unresolved",
        status=_status_for_ratio(ratio, 0.05, 0.25),
        rows_affected=unresolved,
    )


def _check_completeness(
    report: ValidationReport, source_id: str, part: pl.DataFrame, parsed: pl.Series
) -> None:
    """Daily volume: detect whole-day gaps (source outages) and volume collapse."""
    if parsed.len() == 0:
        return
    days = parsed.drop_nulls().dt.date()
    counts = (
        pl.DataFrame({"day": days})
        .group_by("day")
        .len(name="rows")
        .sort("day")
    )
    span_days = int((counts["day"].max() - counts["day"].min()).days) + 1
    observed_days = counts.height
    coverage = observed_days / max(1, span_days)
    report.add(
        source_id,
        "source_completeness",
        expected="daily coverage >= 90% of span",
        observed=f"{observed_days}/{span_days} days with data ({coverage:.4%})",
        status=Status.PASS if coverage >= 0.9 else Status.WARN,
        severity="warn",
        rows_affected=span_days - observed_days,
        missing_days=_missing_days(counts["day"].to_list()),
    )
    volumes = counts["rows"].cast(pl.Float64)
    median = float(volumes.median() or 0.0)
    if median > 0:
        low_days = int((volumes < median * 0.25).sum())
        report.add(
            source_id,
            "daily_volume_stability",
            expected="few days below 25% of median volume",
            observed=f"{low_days} low-volume days, median={median:.0f}",
            status=Status.PASS if low_days <= 2 else Status.WARN,
            severity="warn",
            rows_affected=low_days,
        )


def _missing_days(days: list[datetime]) -> list[str]:
    if len(days) < 2:
        return []
    present = set(days)
    out: list[str] = []
    cursor = min(days)
    end = max(days)
    while cursor <= end:
        if cursor not in present:
            out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return out[:40]


def event_type_vocabulary() -> list[str]:
    return [e.value for e in EventType]
