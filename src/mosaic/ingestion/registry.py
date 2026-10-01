"""Adapter registry.

Adapters register themselves by ``source_id`` so the pipeline, the API and the
experiment configs can name a source as a string. Third-party adapters (e.g. a
public CSV) register the same way; nothing in the pipeline special-cases the
synthetic sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from mosaic.ingestion.base import SourceAdapter

_REGISTRY: dict[str, type[SourceAdapter]] = {}


def register_adapter(cls: type[SourceAdapter]) -> type[SourceAdapter]:
    """Class decorator. Keyed on ``cls.source_id``."""
    source_id = getattr(cls, "source_id", None)
    if not source_id:
        raise ValueError(f"{cls.__name__} must define a source_id")
    _REGISTRY[source_id] = cls
    return cls


def get_adapter(source_id: str) -> type[SourceAdapter]:
    """Look up an adapter class, loading the built-ins on first use."""
    if not _REGISTRY:
        _load_builtin()
    if source_id not in _REGISTRY:
        raise KeyError(f"no adapter for source {source_id!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[source_id]


def available_adapters() -> list[str]:
    if not _REGISTRY:
        _load_builtin()
    return sorted(_REGISTRY)


def _load_builtin() -> None:
    """Import the built-in adapter modules so their classes register."""
    from mosaic.ingestion.synthetic.adapters import (  # noqa: F401
        IncidentLogAdapter,
        LedgerApiAdapter,
        SensorNetAdapter,
        TransitFeedAdapter,
    )

    for cls in (TransitFeedAdapter, LedgerApiAdapter, IncidentLogAdapter, SensorNetAdapter):
        register_adapter(cls)


def adapter_for(source_id: str, root: Path | str, **kwargs: object) -> SourceAdapter:
    """Instantiate an adapter for a dataset root."""
    return get_adapter(source_id)(Path(root), **kwargs)  # type: ignore[arg-type]
