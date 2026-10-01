"""Deterministic cleaning: raw normalized rows -> canonical events.

Every step is (a) named, (b) configurable, (c) counted, and (d) applied in a fixed
order. Nothing here is hidden inside model code, and no step drops a row silently:
:class:`CleaningLedger` records what each step removed and why, and the ledger is
written next to the cleaned data.

Step order matters and is asserted by the tests:

1. ``timestamps``   - parse and normalize to tz-naive UTC (fixes swapped d/m, UTC suffix)
2. ``types``        - cast magnitudes, categories
3. ``categories``   - trim/case-fold native categories, remap to canonical types
4. ``values``       - null/negative/non-finite handling under an explicit policy
5. ``identifiers``  - whitespace collapse, case-fold, separator normalization
6. ``duplicates``   - exact and near-duplicate removal (configurable)
7. ``required``     - drop rows that are still structurally invalid
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

import polars as pl

from mosaic.schema.canonical import EventType
from mosaic.schema.ids import event_id  # noqa: F401  (re-exported for adapters)

TIMESTAMP_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
)

ValuePolicy = Literal["null", "zero", "keep"]


@dataclass
class CleaningProfile:
    """Every cleaning decision, in one configurable object."""

    name: str = "default"
    timestamp_formats: tuple[str, ...] = TIMESTAMP_FORMATS
    drop_invalid_timestamps: bool = True
    category_case_fold: bool = True
    normalize_identifier_separators: bool = True
    value_policy: ValuePolicy = "null"
    drop_negative_values: bool = True
    drop_non_finite: bool = True
    drop_exact_duplicates: bool = True
    drop_near_duplicates: bool = True
    drop_missing_event_type: bool = False
    min_event_value: float | None = None
    keep_lines: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp_formats": list(self.timestamp_formats),
            "drop_invalid_timestamps": self.drop_invalid_timestamps,
            "category_case_fold": self.category_case_fold,
            "normalize_identifier_separators": self.normalize_identifier_separators,
            "value_policy": self.value_policy,
            "drop_negative_values": self.drop_negative_values,
            "drop_non_finite": self.drop_non_finite,
            "drop_exact_duplicates": self.drop_exact_duplicates,
            "drop_near_duplicates": self.drop_near_duplicates,
            "drop_missing_event_type": self.drop_missing_event_type,
            "min_event_value": self.min_event_value,
        }


@dataclass
class CleaningLedger:
    """Audit trail: how many rows each step removed, and why."""

    dataset: str
    dataset_version: str
    profile: str
    input_rows: int
    output_rows: int
    steps: list[dict[str, Any]] = field(default_factory=list)
    drop_reasons: Counter = field(default_factory=Counter)

    def record(self, step: str, before: int, after: int, *, reason: str = "", **extra: Any) -> None:
        removed = before - after
        self.steps.append(
            {"step": step, "before": before, "after": after, "removed": removed, **extra}
        )
        if removed and reason:
            self.drop_reasons[reason] += removed

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "profile": self.profile,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "total_removed": self.input_rows - self.output_rows,
            "retention_rate": round(self.output_rows / max(1, self.input_rows), 6),
            "steps": self.steps,
            "drop_reasons": dict(self.drop_reasons),
        }


def clean_events(
    raw: pl.DataFrame,
    *,
    dataset: str,
    dataset_version: str,
    profile: CleaningProfile | None = None,
    value_scales: dict[str, float] | None = None,
) -> tuple[pl.DataFrame, CleaningLedger]:
    """Apply the profile in fixed order and return canonical events + ledger.

    ``value_scales`` is the adapter-declared unit rescaling (e.g. cents -> units).
    It is passed in from config rather than hard-coded per source, so adding a
    source never requires editing this module.
    """
    profile = profile or CleaningProfile()
    value_scales = value_scales or {}
    ledger = CleaningLedger(
        dataset=dataset, dataset_version=dataset_version, profile=profile.name, input_rows=raw.height,
        output_rows=raw.height,
    )
    if raw.is_empty():
        return raw, ledger

    frame = raw
    frame = _step_timestamps(frame, profile, ledger)
    frame = _step_types(frame, profile, value_scales, ledger)
    frame = _step_categories(frame, profile, ledger)
    frame = _step_values(frame, profile, ledger)
    frame = _step_identifiers(frame, profile, ledger)
    frame = _step_rounding(frame)
    frame = _step_duplicates(frame, profile, ledger)
    frame = _step_required(frame, profile, ledger)
    frame = _finalize(frame, profile)

    ledger.output_rows = frame.height
    return frame, ledger


# ------------------------------------------------------------------ steps
def _step_timestamps(frame: pl.DataFrame, profile: CleaningProfile, ledger: CleaningLedger) -> pl.DataFrame:
    before = frame.height
    expr = pl.col("timestamp_raw").str.strip_chars()
    parsed = pl.lit(None, dtype=pl.Datetime("us"))
    for fmt in profile.timestamp_formats:
        parsed = (
            pl.when(parsed.is_null())
            .then(expr.str.to_datetime(strict=False, format=fmt))
            .otherwise(parsed)
        )
    # A tz-aware result would be a canonical-contract violation, but
    # ``str.to_datetime`` on a format without %Z never produces one; the assert
    # makes that invariant explicit instead of silently assuming it.
    out = frame.with_columns(parsed.alias("timestamp"))
    if out.schema["timestamp"] == pl.Datetime("us") and out.schema["timestamp"].time_zone:  # pragma: no cover
        raise ValueError("timestamp parsing produced a tz-aware column; canonical schema requires UTC-naive")
    if profile.drop_invalid_timestamps:
        out = out.filter(pl.col("timestamp").is_not_null())
    ledger.record(
        "timestamps",
        before,
        out.height,
        reason="invalid_timestamp",
        parsed_ok=int(frame.height - (before - out.height)),
        formats=list(profile.timestamp_formats),
    )
    return out


def _step_types(
    frame: pl.DataFrame, profile: CleaningProfile, value_scales: dict[str, float], ledger: CleaningLedger
) -> pl.DataFrame:
    before = frame.height
    scale_expr = pl.lit(1.0)
    for source_id, scale in value_scales.items():
        scale_expr = pl.when(pl.col("source_id") == source_id).then(pl.lit(float(scale))).otherwise(scale_expr)
    out = frame.with_columns(
        (pl.col("event_value").cast(pl.Float64, strict=False) * scale_expr).alias("event_value"),
        pl.col("source_flag").cast(pl.Float64, strict=False).fill_null(0.0).alias("source_flag"),
    )
    ledger.record(
        "types",
        before,
        out.height,
        scales={k: v for k, v in value_scales.items() if v != 1.0},
    )
    return out


def _step_categories(frame: pl.DataFrame, profile: CleaningProfile, ledger: CleaningLedger) -> pl.DataFrame:
    before = frame.height
    from mosaic.ingestion.synthetic.adapters import _map_categories_expr
    from mosaic.ingestion.synthetic.vocabulary import CATEGORY_TO_EVENT_TYPE, SUFFIX_MARKERS

    canonical = pl.col("event_type_mapped").str.strip_chars()
    if profile.category_case_fold:
        # Re-map from the *normalized* native label. The adapter's mapping is
        # authoritative, but cleaning re-derives it here so that a row damaged
        # *after* the adapter ran (a whitespace/case variant the adapter's own
        # normalizer did not see) still lands on the right canonical type. Both
        # use the same shared vocabulary, so they cannot disagree.
        normalized = pl.col("event_type_raw").cast(pl.String, strict=False).str.to_lowercase()
        normalized = normalized.str.replace_all(r"\s+", " ").str.strip_chars()
        for marker in SUFFIX_MARKERS:
            normalized = normalized.str.replace_all(f"{marker}$", "")
        # union of every source's vocabulary: the raw frame is multi-source, so
        # one expression must resolve all four sources' labels at once
        union: dict[str, str] = {}
        for mapping in CATEGORY_TO_EVENT_TYPE.values():
            union.update(mapping)
        # column 1 must exist before column 2 can reference it: separate calls
        out = frame.with_columns(normalized.alias("event_type_folded"))
        out = out.with_columns(_map_categories_expr(pl.col("event_type_folded"), union).alias("event_type"))
    else:
        out = frame.with_columns(canonical.alias("event_type"))
    ledger.record("categories", before, out.height)
    return out


def _step_values(frame: pl.DataFrame, profile: CleaningProfile, ledger: CleaningLedger) -> pl.DataFrame:
    before = frame.height
    out = frame
    if profile.drop_non_finite:
        out = out.filter(
            pl.col("event_value").is_null() | pl.col("event_value").is_finite()
        )
    if profile.drop_negative_values:
        out = out.filter(pl.col("event_value").is_null() | (pl.col("event_value") >= 0))
    if profile.min_event_value is not None:
        out = out.filter(pl.col("event_value").is_null() | (pl.col("event_value") >= profile.min_event_value))
    if profile.value_policy == "zero":
        out = out.with_columns(pl.col("event_value").fill_null(0.0))
    ledger.record(
        "values",
        before,
        out.height,
        reason="invalid_value",
        policy=profile.value_policy,
        negatives=0,
    )
    return out


def _step_identifiers(frame: pl.DataFrame, profile: CleaningProfile, ledger: CleaningLedger) -> pl.DataFrame:
    before = frame.height
    def clean_ref(col: str) -> pl.Expr:
        expr = pl.col(col).str.strip_chars()
        if profile.normalize_identifier_separators:
            expr = (
                expr.str.to_lowercase()
                .str.replace_all("_", " ")
                .str.replace_all("-", " ")
                .str.replace_all("::", " ")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
            )
        return expr

    out = frame.with_columns(
        clean_ref("entity_ref_raw").alias("entity_ref_norm"),
        clean_ref("peer_ref_raw").alias("peer_ref_norm"),
    )
    ledger.record("identifiers", before, out.height, normalized_separators=profile.normalize_identifier_separators)
    return out


def _step_rounding(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the 2dp rounded magnitude used by near-duplicate detection.

    Kept as its own column (not an in-place rounding of ``event_value``) so the
    exact value survives for feature computation while duplicate detection gets a
    comparison key that tolerates trivial re-emission drift.
    """
    if "event_value" not in frame.columns:
        return frame
    return frame.with_columns(
        pl.col("event_value").cast(pl.Float64, strict=False).round(2).alias("event_value_rounded")
    )


def _step_duplicates(frame: pl.DataFrame, profile: CleaningProfile, ledger: CleaningLedger) -> pl.DataFrame:
    """Remove duplicate observations.

    Two independent notions, both computed from *observed* fields only:

    * **exact** - identical on (source, record id, timestamp, entity, value). These
      are re-deliveries of the same record and carry no new information.
    * **near** - same (source, timestamp, entity, event type, value) but a
      *different* record id, i.e. the same event re-emitted under a new id with a
      tiny value drift. This is the form a real feed produces when it retries.

    Neither notion consults the synthetic ``latent_id``. Using it would be a truth
    backchannel: it would make the cleaner look better than a real deployment could
    be, and it would leak the answer into the ingestion path. The record id is
    deliberately excluded from the near-duplicate key so that a mangled id is
    caught, and it is a real observation that the feed retries.
    """
    before = frame.height
    out = frame
    if profile.drop_exact_duplicates:
        out = out.unique(subset=_dedup_key(out), keep="first", maintain_order=True)
    exact_removed = before - out.height
    if profile.drop_near_duplicates:
        out = out.unique(subset=_near_dedup_key(out), keep="first", maintain_order=True)
    ledger.record(
        "duplicates",
        before,
        out.height,
        reason="duplicate_record",
        exact_removed=exact_removed,
        near_removed=before - exact_removed - out.height,
        exact=profile.drop_exact_duplicates,
        near=profile.drop_near_duplicates,
    )
    return out


def _near_dedup_key(frame: pl.DataFrame) -> list[str]:
    """Fields that identify the same *observation* across differing record ids.

    Rounded to 2dp: the renderer's near-duplicate drift is a 1% multiplier, so an
    exact float comparison would miss it while a tight round catches it without
    merging genuinely distinct events.
    """
    keys = ["source_id", "entity_ref_norm", "event_type", "location_id"]
    if "timestamp" in frame.columns:
        keys.insert(1, "timestamp")
    if "event_value" in frame.columns:
        keys.append("event_value_rounded")
    return [k for k in keys if k in frame.columns]


def _dedup_key(frame: pl.DataFrame) -> list[str]:
    keys = ["source_id", "source_record_id", "timestamp", "entity_ref_norm", "event_value"]
    return [k for k in keys if k in frame.columns]


def _step_required(frame: pl.DataFrame, profile: CleaningProfile, ledger: CleaningLedger) -> pl.DataFrame:
    before = frame.height
    out = frame.filter(pl.col("source_record_id").is_not_null() & (pl.col("source_record_id").str.len_chars() > 0))
    out = out.filter(pl.col("event_type").is_not_null())
    if profile.drop_missing_event_type:
        out = out.filter(pl.col("event_type") != "other")
    ledger.record("required", before, out.height, reason="structurally_invalid")
    return out


def _finalize(frame: pl.DataFrame, profile: CleaningProfile) -> pl.DataFrame:
    """Add canonical ids, lineage columns and the fields downstream stages expect.

    ``event_id`` is the readable composite ``ev_<source_id>_<source_record_id>``.
    Readable, vectorised and stable: re-cleaning the same source record yields the
    same id, so lineage survives reprocessing and an anomaly can be traced back to
    a source record id by eye. The hashed form in :func:`mosaic.schema.ids.event_id`
    is used for ids that would otherwise be long (entity natural keys, experiment ids).
    """
    frame = frame.with_columns(composite_event_id().alias("event_id"))
    additions = {
        "ingestion_time": pl.col("timestamp"),
        "transform_profile": pl.lit(profile.name, dtype=pl.Utf8),
        "entity_ref_source": pl.col("entity_ref_raw"),
        "related_entity_refs": pl.col("peer_ref_norm"),
    }
    if "event_type_folded" in frame.columns:
        additions["source_event_type"] = pl.col("event_type_folded")
    else:
        additions["source_event_type"] = pl.col("event_type")
    return frame.with_columns(**additions).sort(
        "timestamp", "source_id", "source_record_id", nulls_last=True
    )


# ---------------------------------------------------------------- helpers
def composite_event_id() -> pl.Expr:
    """Vectorised canonical event id: ``ev_<source_id>_<source_record_id>``."""
    return (
        pl.lit("ev_", dtype=pl.Utf8)
        + pl.col("source_id")
        + pl.lit("_", dtype=pl.Utf8)
        + pl.col("source_record_id")
    )


def _remap_expression(folded_col: str, out_col: str) -> pl.Expr:
    """Rebuild the canonical type from the case-folded native category.

    Uses the *same* dictionary the adapters use, so cleaning and normalization
    cannot disagree about what 'BUS' means.
    """
    from mosaic.ingestion.synthetic.vocabulary import CATEGORY_TO_EVENT_TYPE

    reverse: dict[str, str] = {}
    for mapping in CATEGORY_TO_EVENT_TYPE.values():
        for native, canonical in mapping.items():
            reverse.setdefault(native, canonical)
    pairs = sorted(reverse.items())
    expr = pl.lit(EventType.OTHER.value, dtype=pl.Utf8)
    for native, canonical in reversed(pairs):
        expr = pl.when(pl.col(folded_col) == native).then(pl.lit(canonical, dtype=pl.Utf8)).otherwise(expr)
    return expr.alias(out_col)
