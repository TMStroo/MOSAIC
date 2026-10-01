"""Deterministic identifier and hashing helpers.

All IDs are content-derived so that re-running any pipeline stage on the same
input produces byte-identical artifacts. That property is what makes checksum
validation in ``mosaic reproduce`` meaningful.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

_ID_PREFIX = {
    "event": "ev",
    "entity": "en",
    "relation": "rel",
    "anomaly": "an",
    "experiment": "exp",
    "model": "mdl",
    "claim": "clm",
    "dataset": "ds",
    "feature_set": "fs",
    "run": "run",
}


def stable_hash(*parts: Any, length: int = 16) -> str:
    """Return a short, stable, platform-independent hash of ``parts``.

    Uses a canonical JSON encoding so that dict ordering never changes the result.
    """
    payload = json.dumps(parts, sort_keys=True, default=_json_default, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return sorted(str(item) for item in obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()  # numpy scalars / arrays
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"unserialisable type for stable_hash: {type(obj)!r}")


def _make(kind: str, *parts: Any) -> str:
    return f"{_ID_PREFIX.get(kind, kind)}_{stable_hash(kind, *parts, length=20)}"


def event_id(source_id: str, source_record_id: str) -> str:
    """Canonical event id: unique per (source, source record).

    Deliberately *not* derived from the payload, so that re-cleaning the same
    source record yields the same event id and lineage survives reprocessing.
    """
    return _make("event", source_id, source_record_id)


def event_id_expr(source_col: str = "source_id", record_col: str = "source_record_id") -> str:
    """SQL-expressible form of :func:`event_id` for Polars/DuckDB string ops.

    Keeps the 20-hex-char prefix format so ids stay uniform, but is a plain
    substring of a sha256 hex so it can be evaluated vectorised (or pushed into
    DuckDB) instead of row-by-row in Python. The two forms are interchangeable
    *within* MOSAIC; ids only need to be stable and unique, not to be the same
    bytes as the Python helper.
    """
    return f"ev_{{{source_col}}}\x1f{{{record_col}}}"


def entity_uid(entity_type: str, natural_key: str) -> str:
    """Canonical entity id from the resolved natural key."""
    return _make("entity", entity_type, natural_key)


def relation_id(src: str, dst: str, relation_type: str, bucket: str = "") -> str:
    return _make("relation", src, dst, relation_type, bucket)


def anomaly_id(model_id: str, target_ref: str, started_at: datetime) -> str:
    return _make("anomaly", model_id, target_ref, started_at.isoformat())


def experiment_id(name: str, config_hash: str, created_at: datetime) -> str:
    return _make("experiment", name, config_hash, created_at.isoformat())


def schema_hash(columns: Mapping[str, str]) -> str:
    """Stable hash of a column -> dtype map (schema drift detection)."""
    return stable_hash({str(k): str(v) for k, v in sorted(columns.items())}, length=32)


def data_checksum(values: Iterable[Any], length: int = 32) -> str:
    """Order-insensitive-by-order-sensitive content checksum over a column."""
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(repr(value).encode("utf-8"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()[:length]


def file_checksum(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming sha256 of a file."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            hasher.update(block)
    return hasher.hexdigest()
