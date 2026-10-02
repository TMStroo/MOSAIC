"""Minimal versioned feature artifact.

Purpose: make "which features, fitted on what, under which label protocol?"
a property of a file on disk rather than of whichever script happened to run.

Scope, deliberately small. This is not a feature store. It writes one Parquet
file plus a JSON sidecar, keyed by a content hash of the inputs that determine
the result, and refuses to reuse an artifact whose key does not match. If the
key is wrong the answer is a recompute, not a stale read.

**Why this exists.** Measured on `world_a`: rebuilding the feature matrix costs
1.42 s; writing and re-reading it costs 0.27 s (56.4 MiB). The saving is modest
in absolute terms, but the reproducibility gain is the real reason -- a run
that cites a feature artifact cites a hash, so two results claiming the same
feature set are provably comparable.

**What the key covers.** The event-level label columns, the fit-state period,
the feature configuration, the feature set name/columns, and the source dataset
version. Anything that changes a feature's value must change the key; anything
that does not is deliberately excluded so the key stays stable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from mosaic.experiments.protocol import PERIODS
from mosaic.schema.ids import stable_hash

#: Bump when the artifact's own format or key inputs change.
ARTIFACT_FORMAT = "feature_artifact_v1"


@dataclass(frozen=True)
class FeatureArtifact:
    """A computed feature matrix with the metadata needed to reproduce it."""

    key: str
    path: Path
    manifest_path: Path
    frame: pl.DataFrame
    manifest: dict[str, Any]

    @property
    def n_rows(self) -> int:
        return self.frame.height

    def to_dict(self) -> dict[str, Any]:
        return {**self.manifest, "key": self.key, "path": str(self.path)}


def _artifact_key(
    *,
    dataset_version: str,
    feature_set_name: str,
    feature_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    fit_period: str,
    fitted_through: str,
    include_graph: bool,
) -> str:
    """Content hash of everything that determines the matrix.

    ``fingerprint`` of the registry is deliberately not included separately: the
    resolved column names already pin the feature definitions, and including
    both would make the key change when nothing about the values changed.
    """
    return stable_hash(
        {
            "format": ARTIFACT_FORMAT,
            "dataset_version": dataset_version,
            "feature_set_name": feature_set_name,
            "feature_columns": list(feature_columns),
            "label_columns": list(label_columns),
            "fit_period": fit_period,
            "fitted_through": fitted_through,
            "include_graph": include_graph,
        },
        length=16,
    )


def artifact_path(root: str | Path, key: str) -> Path:
    return Path(root) / f"features_{key}.parquet"


def manifest_path(root: str | Path, key: str) -> Path:
    return Path(root) / f"features_{key}.json"


def load(root: str | Path, key: str) -> FeatureArtifact | None:
    """Return the artifact for ``key``, or ``None`` if absent or unverifiable.

    A sidecar whose ``key`` disagrees with its own filename is treated as
    missing rather than trusted: a partially-written pair must not produce a
    silently mismatched feature matrix.
    """
    data_path = artifact_path(root, key)
    meta_path = manifest_path(root, key)
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if manifest.get("key") != key:
        return None
    return FeatureArtifact(
        key=key,
        path=data_path,
        manifest_path=meta_path,
        frame=pl.read_parquet(data_path),
        manifest=manifest,
    )


def registry_path(root: str | Path, key: str) -> Path:
    return Path(root) / f"features_{key}.registry.json"


def save(root: str | Path, artifact: FeatureArtifact, registry=None) -> Path:
    """Write the frame, then the manifest, then the registry sidecar.

    The manifest is written last of the two mandatory files so a crash mid-write
    leaves an orphan Parquet without a manifest, which ``load`` treats as a
    miss. The registry sidecar carries the per-column family assignments; when
    supplied it is written too, because the feature sets F0-F5 are defined by
    family and cannot be rebuilt from the column list alone.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    artifact.frame.write_parquet(artifact.path)
    artifact.manifest_path.write_text(
        json.dumps({**artifact.manifest, "key": artifact.key}, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    if registry is not None:
        registry.save(registry_path(root_path, artifact.key))
    return artifact.path


def build_key_and_manifest(
    *,
    dataset_version: str,
    feature_set_name: str,
    feature_set_id: str,
    feature_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    fit_period: str,
    fitted_through: str,
    include_graph: bool,
    registry_version: str,
    extra: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Compute the key and the sidecar payload without touching disk."""
    key = _artifact_key(
        dataset_version=dataset_version,
        feature_set_name=feature_set_name,
        feature_columns=feature_columns,
        label_columns=label_columns,
        fit_period=fit_period,
        fitted_through=fitted_through,
        include_graph=include_graph,
    )
    manifest: dict[str, Any] = {
        "artifact_format": ARTIFACT_FORMAT,
        "dataset_version": dataset_version,
        "feature_set_name": feature_set_name,
        "feature_set_id": feature_set_id,
        "feature_set_version": registry_version,
        "feature_columns": list(feature_columns),
        "label_columns": list(label_columns),
        "fit_period": fit_period,
        "fitted_through": fitted_through,
        "include_graph": include_graph,
        "periods": list(PERIODS),
    }
    if extra:
        manifest.update(extra)
    return key, manifest


def make_artifact(
    frame: pl.DataFrame,
    *,
    root: str | Path,
    key: str,
    manifest: dict[str, Any],
) -> FeatureArtifact:
    return FeatureArtifact(
        key=key,
        path=artifact_path(root, key),
        manifest_path=manifest_path(root, key),
        frame=frame,
        manifest=manifest,
    )
