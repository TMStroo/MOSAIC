"""Temporal graph subsystem for MOSAIC.

Public surface:

* :class:`TemporalGraph` - an edge list with first-seen times and causal snapshots.
* :func:`add_graph_features` - attach graph features to an event frame.
* :func:`assert_no_future_edges` - the temporal invariant, checkable at runtime.

Nothing in this package builds a graph from the full dataset for the purpose of
historical features. See ``mosaic.graph.temporal`` for why that matters.
"""

from mosaic.graph.temporal import (
    GraphConfig,
    TemporalGraph,
    add_graph_features,
    assert_no_future_edges,
    betweenness,
    clustering,
    communities,
    degree_distribution,
    pagerank,
    snapshot_features,
    snapshot_measures,
    temporal_community_stability,
)

__all__ = [
    "GraphConfig",
    "TemporalGraph",
    "add_graph_features",
    "assert_no_future_edges",
    "betweenness",
    "clustering",
    "communities",
    "degree_distribution",
    "pagerank",
    "snapshot_features",
    "snapshot_measures",
    "temporal_community_stability",
]