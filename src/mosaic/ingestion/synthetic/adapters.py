"""Source adapters for the synthetic benchmark, one per native schema.

Each adapter's ``normalize`` maps its own vocabulary onto the canonical contract.
None of them share a column name, so the mapping is the substantive work: the
category dictionaries in :mod:`mosaic.ingestion.synthetic.render` are the
*inverse* of what these adapters encode.

The adapters deliberately do **not** clean anything. Timestamps stay as the source
emitted them (including the corrupted ones) so that
:mod:`mosaic.cleaning` has real defects to find and count.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import polars as pl

from mosaic.ingestion.base import SourceAdapter
from mosaic.ingestion.synthetic.generator import SOURCE_ID_STYLE, SOURCE_PLAN, SOURCE_SLUG
from mosaic.ingestion.synthetic.render import NATIVE_COLUMNS
from mosaic.ingestion.synthetic.vocabulary import CATEGORY_TO_EVENT_TYPE
from mosaic.schema.canonical import EventType


def _map_categories_expr(column: pl.Expr, mapping: dict[str, str]) -> pl.Expr:
    """Vectorised native -> canonical category mapping, with normalization.

    Built as a chain of ``when/then`` on the *normalized* label so ``'  BUS_V2 '``
    resolves to ``'movement'``. An expression (rather than a join) keeps it usable
    inside a larger select without a temporary frame.
    """
    from mosaic.ingestion.synthetic.vocabulary import SUFFIX_MARKERS

    normalized = column.cast(pl.String, strict=False).str.to_lowercase()
    normalized = normalized.str.replace_all(r"\s+", " ").str.strip_chars()
    for marker in SUFFIX_MARKERS:
        normalized = normalized.str.replace_all(f"{marker}$", "")

    expr = pl.lit(EventType.OTHER.value, dtype=pl.Utf8)
    for native, canonical in sorted(mapping.items()):
        expr = (
            pl.when(normalized == native).then(pl.lit(canonical, dtype=pl.Utf8)).otherwise(expr)
        )
    return expr


class _SyntheticAdapterBase(SourceAdapter):
    """Shared plumbing: locate ``<dataset>/<source_id>/part-*.parquet``."""

    dataset_slug: ClassVar[str] = ""
    source_id: ClassVar[str] = ""

    def __init__(self, root: Path, *, dataset: str = "world_a", source_version: str = "1.0.0") -> None:
        super().__init__(root, source_version=source_version)
        self.dataset = dataset
        self.columns = NATIVE_COLUMNS[self.source_id]

    @property
    def dataset_root(self) -> Path:
        return self.root / self.dataset

    def discover(self) -> list[Path]:
        base = self.dataset_root / self.source_id
        if not base.exists():
            return []
        return sorted(base.glob("part-*.parquet"))

    def normalize(self, frame: pl.DataFrame) -> pl.DataFrame:
        c = self.columns
        out = frame.select(
            pl.lit(self.source_id).alias("source_id"),
            pl.col(c["record"]).cast(pl.String, strict=False).alias("source_record_id"),
            pl.col(c["time"]).cast(pl.String, strict=False).alias("timestamp_raw"),
            pl.col(c["actor"]).cast(pl.String, strict=False).alias("entity_ref_raw"),
            pl.col(c["kind"]).cast(pl.String, strict=False).alias("event_type_raw"),
            pl.col(c["value"]).cast(pl.Float64, strict=False).alias("event_value"),
            pl.col(c["place"]).cast(pl.String, strict=False).alias("location_id"),
            pl.col(c["peer"]).cast(pl.String, strict=False).alias("peer_ref_raw"),
            pl.col(c["flag"]).cast(pl.Float64, strict=False).alias("source_flag"),
            pl.col("source_batch").cast(pl.String, strict=False).alias("source_batch"),
        )
        # A float column can arrive holding NaN (the sentinel a source uses for
        # "field absent"). Polars keeps NaN *distinct* from null, and a NaN that
        # reaches a mean/std silently poisons it, so the sentinel is converted to a
        # real null here, at the boundary, and is counted as missingness by the
        # validator rather than as a corrupt value.
        out = out.with_columns(
            pl.when(pl.col("event_value").is_finite())
            .then(pl.col("event_value"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("event_value")
        )
        return out.with_columns(
            self._map_categories().alias("event_type_mapped"),
            pl.lit(SOURCE_PLAN[self.source_id][0], pl.Float64).alias("source_lag_days"),
            pl.lit(SOURCE_SLUG[self.source_id], pl.Utf8).alias("source_name"),
        )

    def _map_categories(self) -> pl.Expr:
        """Map the native category column onto the canonical taxonomy.

        Delegates to the shared normalizer so this adapter, the renderer and the
        cleaning step repair exactly the same damage. Anything unrecognised falls
        through to ``other`` and is counted by the validator.
        """
        return _map_categories_expr(
            pl.col("event_type_raw"), CATEGORY_TO_EVENT_TYPE[self.source_id]
        )

    @property
    def limitations(self) -> list[str]:
        return [
            f"id style {SOURCE_ID_STYLE[self.source_id]!r}: resolution against other sources is lossy",
            f"reporting lag {SOURCE_PLAN[self.source_id][0]} day(s)",
            "unknown category values are mapped to 'other' and reported by validation",
        ]


class TransitFeedAdapter(_SyntheticAdapterBase):
    """Municipal transit feed: movements, taps, stop codes."""

    source_id: ClassVar[str] = "transit_feed"
    dataset_slug: ClassVar[str] = "municipal_transit_feed"
    required_columns: ClassVar[tuple[str, ...]] = (
        "trip_uid",
        "departure_iso",
        "cardholder_ref",
        "leg_type",
        "fare_amount",
        "stop_code",
    )
    known_limitations: ClassVar[tuple[str, ...]] = (
        "1-day reporting lag makes it the natural 'forward-looking' source",
        "only sees movement/access/communication/incident events",
        "cardholder_ref carries typo damage on ~3% of rows",
    )


class LedgerApiAdapter(_SyntheticAdapterBase):
    """Merchant ledger: transactions, logins, chargebacks, in cents."""

    source_id: ClassVar[str] = "ledger_api"
    dataset_slug: ClassVar[str] = "merchant_ledger_api"
    required_columns: ClassVar[tuple[str, ...]] = (
        "txn_hash",
        "posted_at_utc",
        "merchant_customer_id",
        "channel_type",
        "amount_eur",
        "terminal_city",
    )
    known_limitations: ClassVar[tuple[str, ...]] = (
        "amount_eur is in minor units (cents) and must be rescaled by an adapter rule",
        "12% of rows have a missing field",
        "terminal_city is a coarse location, not a coordinate",
    )


class IncidentLogAdapter(_SyntheticAdapterBase):
    """Incident management: outages, scheduled work, after-hours access."""

    source_id: ClassVar[str] = "incident_logs"
    dataset_slug: ClassVar[str] = "incident_management_logs"
    required_columns: ClassVar[tuple[str, ...]] = (
        "ticket_no",
        "opened_at",
        "assignee_ref",
        "category",
        "downtime_minutes",
        "facility_code",
    )
    known_limitations: ClassVar[tuple[str, ...]] = (
        "assignee_ref is a person-ish reference and is shared across facilities (ambiguous)",
        "highest missingness of the four sources (12%)",
        "downtime_minutes is a duration, not a magnitude; units differ per source",
    )


class SensorNetAdapter(_SyntheticAdapterBase):
    """Industrial sensor network: analog readings, calibration, service logins."""

    source_id: ClassVar[str] = "sensor_net"
    dataset_slug: ClassVar[str] = "industrial_sensor_net"
    required_columns: ClassVar[tuple[str, ...]] = (
        "reading_id",
        "observed_at",
        "probe_id",
        "channel",
        "reading",
        "zone",
    )
    known_limitations: ClassVar[tuple[str, ...]] = (
        "probe_id uses the site::slug style, the least similar to the other sources",
        "one row per reading: this source dominates raw volume",
        "reading is float with injected negative outliers",
    )


class SyntheticMultiSourceAdapter:
    """Facade returning one adapter instance per source in a dataset.

    Used by :mod:`mosaic.ingestion.pipeline` so callers do not hard-code the
    adapter list, and so a fifth source can be added without touching the pipeline.
    """

    ADAPTERS: tuple[type[_SyntheticAdapterBase], ...] = (
        TransitFeedAdapter,
        LedgerApiAdapter,
        IncidentLogAdapter,
        SensorNetAdapter,
    )

    def __init__(self, root: Path | str, dataset: str = "world_a", source_version: str = "1.0.0") -> None:
        self.root = Path(root)
        self.dataset = dataset
        self.source_version = source_version

    def adapters(self) -> list[_SyntheticAdapterBase]:
        return [cls(self.root, dataset=self.dataset, source_version=self.source_version) for cls in self.ADAPTERS]

    def source_ids(self) -> list[str]:
        return [cls.source_id for cls in self.ADAPTERS]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "source_id": adapter.source_id,
                "source_name": SOURCE_SLUG[adapter.source_id],
                "id_style": SOURCE_ID_STYLE[adapter.source_id],
                "columns": NATIVE_COLUMNS[adapter.source_id],
                "category_map": CATEGORY_TO_EVENT_TYPE[adapter.source_id],
            }
            for adapter in self.adapters()
        ]
