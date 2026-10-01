"""Temporal graph subsystem for MOSAIC.

Public surface:

* :func:`resolve_peer_references` - canonicalize raw peer references into entity ids.
* :func:`build_relations` - the edge list, built from canonical ids only.
* :class:`TemporalGraph` - an edge list with first-seen times and causal snapshots.
* :func:`add_graph_features` - attach graph features to an event frame.
* :func:`assert_no_future_edges` - the temporal invariant, checkable at runtime.

Nothing in this package builds a graph from the full dataset for the purpose of
historical features. See ``mosaic.graph.temporal`` for why that matters.
"""

from mosaic.graph.peers import (
    PEER_RESOLUTION_COLUMNS,
    RESOLVED,
    SELF_REFERENCE,
    UNRESOLVED,
    PeerResolutionResult,
    attach_subject_entities,
    build_relations,
    peer_resolution_report,
    resolve_peer_references,
)
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
    "PEER_RESOLUTION_COLUMNS",
    "RESOLVED",
    "SELF_REFERENCE",
    "UNRESOLVED",
    "GraphConfig",
    "PeerResolutionResult",
    "TemporalGraph",
    "add_graph_features",
    "assert_no_future_edges",
    "attach_subject_entities",
    "betweenness",
    "build_relations",
    "clustering",
    "communities",
    "degree_distribution",
    "pagerank",
    "peer_resolution_report",
    "resolve_peer_references",
    "snapshot_features",
    "snapshot_measures",
    "temporal_community_stability",
]
