"""Materialise a generated world to disk: source Parquet + a truth sidecar.

Layout produced under ``data/synthetic/<spec.name>/``::

    <source_id>/part-000.parquet      # the only thing ingestion ever reads
    _truth/labels.parquet             # ground truth; never ingested
    _truth/entity_links.parquet       # ER ground truth; never ingested
    _truth/latent_events.parquet      # latent_id -> latent entity/type (eval only)
    manifest.json                     # spec, checksums, counts, truth summary

The ``_truth/`` prefix is deliberate: :func:`mosaic.ingestion.pipeline` globs
``*/part-*.parquet`` under each source directory and never descends into it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from mosaic.ingestion.synthetic.generator import (
    SyntheticSpec,
    generate_world,
    render_entity_id,
    with_observed_refs,
)
from mosaic.ingestion.synthetic.render import NATIVE_COLUMNS, render_world
from mosaic.ingestion.synthetic.world import SyntheticWorld
from mosaic.schema.ids import file_checksum, stable_hash
from mosaic.utils.io import write_json, write_parquet

TRUTH_DIR = "_truth"
#: Named environments used by the CLI, the experiment configs and the tests.
#:
#: ``n_entities`` is explicit for research profiles (a small population makes entity
#: resolution and community structure degenerate); the ``scale_*`` profiles set
#: ``derive_entities`` so the event count - the variable under study - drives it.
SPECIALS: dict[str, SyntheticSpec] = {
    "small": SyntheticSpec(
        name="world_a", seed=20260901, n_entities=500, target_events=40_000, days=180, n_sources=4
    ),
    "research": SyntheticSpec(
        name="world_a", seed=20260901, n_entities=1_200, target_events=180_000, days=365, n_sources=4
    ),
    "world_b": SyntheticSpec(
        name="world_b",
        seed=777_001,
        n_entities=1_200,
        target_events=180_000,
        days=365,
        n_sources=4,
        profile_weights={
            "commuter": 0.10,
            "logistics": 0.30,
            "sensor": 0.30,
            "sporadic": 0.10,
            "bursty": 0.12,
            "night_owl": 0.08,
        },
        regime_count=3,
        ambiguous_rate=0.14,
    ),
    "scale_100k": SyntheticSpec(
        name="scale_100k", seed=424_242, target_events=100_000, days=180, derive_entities=True
    ),
    "scale_500k": SyntheticSpec(
        name="scale_500k", seed=424_242, target_events=500_000, days=180, derive_entities=True
    ),
    "scale_1m": SyntheticSpec(
        name="scale_1m", seed=424_242, target_events=1_000_000, days=180, derive_entities=True
    ),
    "scale_5m": SyntheticSpec(
        name="scale_5m", seed=424_242, target_events=5_000_000, days=180, derive_entities=True
    ),
    "scale_10m": SyntheticSpec(
        name="scale_10m", seed=424_242, target_events=10_000_000, days=180, derive_entities=True
    ),
}


def spec_for_profile(profile: str, **overrides: Any) -> SyntheticSpec:
    """Named presets used by the CLI, the experiment configs and the tests."""
    if profile not in SPECIALS:
        raise KeyError(f"unknown synthetic profile {profile!r}; have {sorted(SPECIALS)}")
    base = SPECIALS[profile]
    data = base.as_dict()
    data.update(overrides)
    return SyntheticSpec(**data)


def write_world(spec: SyntheticSpec, root: Path | str, *, world: SyntheticWorld | None = None) -> dict[str, Any]:
    """Generate (or reuse) a world and write it to ``root/spec.name``."""
    from mosaic.ingestion.synthetic.generator import SOURCE_ID_STYLE

    w = world or generate_world(spec)
    assert w.stream is not None, "stream is required for rendering"
    spec = w.spec
    base = Path(root) / spec.name
    base.mkdir(parents=True, exist_ok=True)

    frames, observed = render_world(spec, w.stream, w.entity_links)
    links = with_observed_refs(w.entity_links, observed)
    row_counts: dict[str, int] = {}
    schemas: dict[str, dict[str, str]] = {}
    for source_id, frame in frames.items():
        target = base / source_id / "part-000.parquet"
        write_parquet(target, frame)
        row_counts[source_id] = frame.height
        schemas[source_id] = {c: str(d) for c, d in frame.schema.items()}

    truth_dir = base / TRUTH_DIR
    write_parquet(truth_dir / "labels.parquet", w.truth)
    write_parquet(truth_dir / "entity_links.parquet", links)
    latent = w.events.select("latent_id", "entity_key", "related_key", "event_type", "location_id")
    write_parquet(truth_dir / "latent_events.parquet", latent)

    checksums = {
        source_id: file_checksum(base / source_id / "part-000.parquet") for source_id in frames
    }
    manifest: dict[str, Any] = {
        "dataset_name": spec.name,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "generator_version": "1.0.0",
        "spec": spec.as_dict(),
        "spec_fingerprint": spec.fingerprint(),
        "sources": sorted(frames),
        "row_counts": row_counts,
        "total_rows": int(sum(row_counts.values())),
        "schemas": schemas,
        "native_columns": NATIVE_COLUMNS,
        "checksums": checksums,
        "truth_summary": w.truth_summary(),
        "entity_link_summary": {**w.entity_link_summary(), "observed_refs": int(links.height) - int(w.entity_links.height)},
        "er_noise": spec.er_noise,
        "time_span": [
            str(w.events["timestamp"].min()),
            str(w.events["timestamp"].max()),
        ],
        "id_styles": SOURCE_ID_STYLE,
        "truth_files": [
            "labels.parquet",
            "entity_links.parquet",
            "latent_events.parquet",
        ],
        "id_example": {
            source_id: render_entity_id(SOURCE_ID_STYLE[source_id], 1, 0)
            for source_id in sorted(frames)
        },
    }
    manifest["version"] = stable_hash(
        {k: manifest[k] for k in ("spec_fingerprint", "row_counts", "checksums")}, length=20
    )
    write_json(base / "manifest.json", manifest)
    return manifest


def load_manifest(root: Path | str, name: str) -> dict[str, Any]:
    path = Path(root) / name / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no dataset {name!r} under {root}; run `python -m mosaic data generate --profile {name}`"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def available_datasets(root: Path | str) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "manifest.json").exists())
