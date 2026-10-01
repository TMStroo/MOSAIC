"""The source-adapter contract.

Every adapter must answer six questions, and answer them *empirically*
(computed from the data), not by assertion:

* ``discover``      - where is the data, what files, what format?
* ``load``          - rows as they arrived, plus adapter-side lineage columns.
* ``validate_schema`` - does the file still look like the contract?
* ``normalize``     - map native columns onto :class:`CanonicalEvent`.
* ``emit_metadata`` - row count, schema hash, checksum, time range, limitations.

The base class implements the common parts (checksums, metadata assembly,
CSV/JSON/Parquet sniffing) so an adapter is usually ~100 lines.
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import polars as pl

from mosaic.schema.canonical import SourceMeta
from mosaic.schema.ids import data_checksum, file_checksum, schema_hash
from mosaic.utils.io import read_parquet

LOGGER = logging.getLogger(__name__)


@dataclass
class SchemaCheck:
    """One structural check performed against the declared source contract."""

    column: str
    expected: str
    observed: str
    status: str  # PASS | WARN | FAIL
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "expected": self.expected,
            "observed": self.observed,
            "status": self.status,
            "severity": self.severity,
        }


@dataclass
class SourceLoadResult:
    """Raw-but-lineage-aware rows plus metadata. No normalization happens here."""

    source_id: str
    frame: pl.DataFrame
    metadata: SourceMeta
    schema_checks: list[SchemaCheck] = field(default_factory=list)
    dropped_rows: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return self.frame.height


class SourceAdapter(abc.ABC):
    """Base class for all MOSAIC source adapters."""

    #: stable key used in every downstream table
    source_id: ClassVar[str]
    #: native -> canonical column mapping; required fields must be present
    column_map: ClassVar[dict[str, str]] = {}
    #: dtypes the adapter declares for required native columns
    expected_dtypes: ClassVar[dict[str, pl.DataType]] = {}
    #: columns the adapter requires to exist
    required_columns: ClassVar[tuple[str, ...]] = ()
    #: free-text limitations recorded in the manifest and surfaced in the UI
    known_limitations: ClassVar[tuple[str, ...]] = ()
    #: how the adapter names the dataset on disk
    dataset_slug: ClassVar[str] = ""

    def __init__(self, root: Path, *, source_version: str = "1.0.0") -> None:
        self.root = Path(root)
        self.source_version = source_version

    # -- discovery -----------------------------------------------------------
    @abc.abstractmethod
    def discover(self) -> list[Path]:
        """Return the concrete files this adapter would read."""

    def paths(self) -> list[Path]:
        found = self.discover()
        if not found:
            raise FileNotFoundError(f"{self.source_id}: no data found under {self.root}")
        return found

    # -- loading -------------------------------------------------------------
    def load(self) -> pl.DataFrame:
        """Read all discovered files, adding ``_source_file`` lineage."""
        frames: list[pl.DataFrame] = []
        for path in self.paths():
            frame = self._read_one(path)
            frame = frame.with_columns(
                pl.lit(path.name).alias("_source_file"),
                pl.lit(str(path.relative_to(self.root)).replace("\\", "/")).alias("_raw_reference"),
            )
            frames.append(frame)
        if not frames:
            return pl.DataFrame()
        out = pl.concat(frames, how="diagonal_relaxed", rechunk=True)
        if out.height == 0:
            return out
        return out.sort("_raw_reference", "_source_file", nulls_last=True)

    def _read_one(self, path: Path) -> pl.DataFrame:
        """Format sniffing: parquet, csv(.gz), jsonl(.gz), or csv fallback."""
        suffixes = [s.lower() for s in path.suffixes]
        if ".parquet" in suffixes or ".pq" in suffixes:
            return read_parquet(path)
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            return pl.read_ndjson(path)
        if path.suffix.lower() in {".json", ".js"}:
            return pl.read_json(path)
        return pl.read_csv(
            path,
            infer_schema_length=10_000,
            try_parse_dates=False,  # timestamps are normalized in cleaning, not here
            null_values=["", "NA", "N/A", "null", "NULL", "None", "nan", "NaN"],
        )

    # -- schema validation ---------------------------------------------------
    def validate_schema(self, frame: pl.DataFrame) -> list[SchemaCheck]:
        checks: list[SchemaCheck] = []
        present = set(frame.columns)
        for column in self.required_columns:
            checks.append(
                SchemaCheck(
                    column=column,
                    expected="present",
                    observed="present" if column in present else "missing",
                    status="PASS" if column in present else "FAIL",
                )
            )
        for column, dtype in self.expected_dtypes.items():
            if column not in present:
                continue
            observed = frame.schema[column]
            ok = observed == dtype or observed.is_numeric()
            checks.append(
                SchemaCheck(
                    column=column,
                    expected=str(dtype),
                    observed=str(observed),
                    status="PASS" if ok else "WARN",
                    severity="warn",
                )
            )
        extra = sorted(present - set(self.required_columns) - set(self.expected_dtypes))
        for column in extra:
            checks.append(
                SchemaCheck(
                    column=column,
                    expected="declared or ignored",
                    observed="undeclared column",
                    status="WARN",
                    severity="warn",
                )
            )
        return checks

    # -- normalization -------------------------------------------------------
    @abc.abstractmethod
    def normalize(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Map native rows onto canonical event columns (no cleaning yet).

        Must return columns that :func:`mosaic.cleaning.events.build_canonical`
        can consume: ``source_id``, ``source_record_id``, ``timestamp_raw``,
        and whatever the canonical layer needs.
        """

    # -- metadata ------------------------------------------------------------
    def emit_metadata(self, frame: pl.DataFrame) -> SourceMeta:
        """Compute provenance from the data itself (never from a config claim)."""
        checksums = []
        for path in self.paths():
            checksums.append(file_checksum(path))
        time_range = None
        for candidate in ("timestamp", "timestamp_raw", "ts", "event_time"):
            if candidate in frame.columns and frame.height:
                series = frame[candidate].drop_nulls()
                if len(series):
                    time_range = (series.min(), series.max())
                    break
        return SourceMeta(
            source_id=self.source_id,
            source_name=self.source_id.replace("_", " ").title(),
            source_version=self.source_version,
            downloaded_at=datetime.utcnow(),
            row_count=frame.height,
            column_count=frame.width,
            schema_hash=schema_hash({c: str(d) for c, d in frame.schema.items()}),
            data_checksum=data_checksum(sorted(checksums)),
            time_range=time_range,
            known_limitations=list(self.known_limitations),
        )

    # -- orchestration -------------------------------------------------------
    def run(self) -> SourceLoadResult:
        """discover -> load -> validate_schema -> metadata. One call, one result."""
        frame = self.load()
        checks = self.validate_schema(frame)
        failures = [c for c in checks if c.status == "FAIL"]
        if failures:
            LOGGER.warning(
                "%s: %d schema check(s) failed: %s",
                self.source_id,
                len(failures),
                ", ".join(c.column for c in failures),
            )
        return SourceLoadResult(
            source_id=self.source_id,
            frame=frame,
            metadata=self.emit_metadata(frame),
            schema_checks=checks,
        )

    def _required_map(self) -> dict[str, str]:
        missing = [c for c in self.column_map if c not in self.column_map.values()]
        if missing:  # pragma: no cover - programming error guard
            raise ValueError(f"{self.source_id}: column_map values incomplete: {missing}")
        return self.column_map


def coerce_confidence(values: pl.Series, default: float = 1.0) -> pl.Series:
    """Map an arbitrary source confidence column into [0, 1] without leaking NaN."""
    return (
        pl.Series(values)
        .cast(pl.Float64, strict=False)
        .fill_null(default)
        .clip(0.0, 1.0)
        .fill_null(default)
    )


def as_list_column(values: pl.Series) -> pl.Series:
    """Normalize delimited / JSON-ish string columns into ``list[str]``."""
    return (
        pl.Series(values)
        .cast(pl.String, strict=False)
        .str.split(";")
        .list.eval(pl.element.cast(pl.String, strict=False).fill_null(""))
        .list.drop_nulls()
    )


SequenceStr = Sequence[str]
LoadHook = Callable[[SourceLoadResult], None]
