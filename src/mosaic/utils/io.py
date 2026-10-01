"""Typed, checksum-aware IO helpers.

Everything MOSAIC persists is Parquet (analytical rows) or JSON (metadata,
configs, metrics, evidence). JSON is always written with sorted keys and a fixed
float format so that checksums are stable across runs and platforms.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from mosaic.schema.ids import stable_hash


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if hasattr(o, "isoformat"):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, frozenset, tuple)):
            return list(o)
        if hasattr(o, "item"):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
        return super().default(o)


def _clean(obj: Any) -> Any:
    """Replace non-finite floats with ``None`` so JSON stays valid and parseable."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, Mapping):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """Write canonical JSON (sorted keys, sanitised floats) and return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(_clean(payload), indent=indent, sort_keys=True, cls=_Encoder) + "\n",
        encoding="utf-8",
    )
    return p


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_parquet(
    path: Path | str,
    df: pl.DataFrame,
    *,
    compression: str = "zstd",
    row_group_size: int | None = None,
) -> Path:
    """Write a Polars DataFrame as a partitioned-friendly Parquet file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(
        p,
        compression=compression,
        row_group_size=row_group_size or max(1024, min(262_144, max(1, df.height))),
    )
    return p


def read_parquet(path: Path | str, columns: list[str] | None = None) -> pl.DataFrame:
    return pl.read_parquet(path, columns=columns)


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML or JSON experiment/dataset config with a stable hash attached.

    The returned dict is the *raw* config; use :func:`config_hash` for the hash so
    that an experiment id can embed it.
    """
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping, got {type(data)!r}")
    return data


def config_hash(config: Mapping[str, Any], *, extra: Mapping[str, Any] | None = None) -> str:
    """Hash of a config, optionally salted with resolved runtime facts.

    ``extra`` typically carries ``dataset_version`` and ``code_version`` so that a
    config change, a data change, or a code change each produce a different id.
    """
    payload: dict[str, Any] = {"config": dict(config)}
    if extra:
        payload["extra"] = dict(extra)
    return stable_hash(payload, length=20)


def count_rows(path: Path | str) -> int:
    """Row count of a Parquet file from its metadata only (no full read)."""
    import pyarrow.parquet as pq

    return pq.ParquetFile(str(path)).metadata.num_rows
