"""MOSAIC - Multi-Source Analytics, Signal & Intelligence Core.

A research-grade Data Science platform for integrating heterogeneous multi-source
event data and detecting temporal, relational and behavioral anomalies under
rigorous out-of-time evaluation.

Every number reported by MOSAIC is generated from stored experiment evidence;
nothing in the research report is typed by hand.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "EVAL_PROTOCOL_VERSION",
    "FEATURE_SET_VERSION",
]

__version__ = "0.1.0"

#: Bumped on any breaking change to the canonical event schema.
SCHEMA_VERSION = "1.0.0"
#: Bumped on any change to the evaluation protocol (splits, threshold rules, seeds).
EVAL_PROTOCOL_VERSION = "1.0.0"
#: Bumped on any change to feature definitions or their windows.
FEATURE_SET_VERSION = "feature_set_v1"
