"""Graph feature layer consumed by :mod:`mosaic.features.compute`.

Kept as a thin shim so the feature module has one import target and the graph
package owns its own implementation.
"""

from __future__ import annotations

import polars as pl

from mosaic.graph.temporal import GraphConfig, add_graph_features as _add

GRAPH_FEATURE_NAMES = (
    "graph_degree",
    "graph_in_degree",
    "graph_out_degree",
    "graph_weighted_degree",
    "graph_pagerank",
    "graph_clustering",
    "graph_community",
    "graph_new_neighbors",
    "graph_new_neighbor_ratio",
    "graph_recent_neighbor_count",
)


def add_graph_features(
    events: pl.DataFrame,
    relations: pl.DataFrame,
    *,
    entity_column: str = "entity_id",
    time_column: str = "timestamp",
    related_column: str = "related_entity_refs",
    config: GraphConfig | None = None,
    snapshot_interval_seconds: int = 6 * 3600,
) -> pl.DataFrame:
    """Attach causal graph features. See :mod:`mosaic.graph.temporal`."""
    return _add(
        events,
        relations,
        entity_column=entity_column,
        time_column=time_column,
        related_column=related_column,
        config=config,
        snapshot_interval_seconds=snapshot_interval_seconds,
    )


__all__ = ["add_graph_features", "GRAPH_FEATURE_NAMES"]