"""Source adapters: raw files -> canonical events, with full provenance."""

from mosaic.ingestion.base import SourceAdapter, SourceLoadResult
from mosaic.ingestion.registry import available_adapters, get_adapter, register_adapter
from mosaic.ingestion.synthetic import SyntheticMultiSourceAdapter

__all__ = [
    "SourceAdapter",
    "SourceLoadResult",
    "SyntheticMultiSourceAdapter",
    "available_adapters",
    "get_adapter",
    "register_adapter",
]
