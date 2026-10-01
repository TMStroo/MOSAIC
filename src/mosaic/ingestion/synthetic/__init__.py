"""Synthetic multi-source benchmark generator, renderers and adapters.

The generator is not decoration: it is the only way to run *controlled*
experiments where ground truth is known by construction, which is what makes
ablation, missingness, entity-resolution-noise and transfer studies defensible.
"""

from mosaic.ingestion.synthetic.builder import (
    available_datasets,
    load_manifest,
    spec_for_profile,
    write_world,
)
from mosaic.ingestion.synthetic.generator import (
    FAMILIES,
    SOURCE_ID_STYLE,
    SOURCE_PLAN,
    SOURCE_SLUG,
    Stream,
    SyntheticSpec,
    generate_world,
    render_entity_id,
)
from mosaic.ingestion.synthetic.render import NATIVE_COLUMNS, render_world
from mosaic.ingestion.synthetic.vocabulary import CATEGORY_TO_EVENT_TYPE
from mosaic.ingestion.synthetic.world import SyntheticWorld

__all__ = [
    "CATEGORY_TO_EVENT_TYPE",
    "FAMILIES",
    "NATIVE_COLUMNS",
    "SOURCE_ID_STYLE",
    "SOURCE_PLAN",
    "SOURCE_SLUG",
    "Stream",
    "SyntheticMultiSourceAdapter",
    "SyntheticSpec",
    "SyntheticWorld",
    "available_datasets",
    "generate_world",
    "load_manifest",
    "render_entity_id",
    "render_world",
    "spec_for_profile",
    "write_world",
]


def __getattr__(name: str):  # lazy: avoids a circular import with ingestion.base
    if name == "SyntheticMultiSourceAdapter":
        from mosaic.ingestion.synthetic.adapters import SyntheticMultiSourceAdapter

        return SyntheticMultiSourceAdapter
    raise AttributeError(name)
