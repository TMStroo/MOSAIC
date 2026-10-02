"""Ground-truth label resolution.

**This is the only module permitted to read synthetic truth.** Detectors,
models, metrics and threshold selection must never import it -- they receive
``is_anomaly`` as an opaque column they are permitted to read, and this module
is what produces it.

Ground truth arrives in ``_truth/`` and describes anomalies by *target kind*,
not by row id, so one lookup does not fit all nine families:

======================  ====================  ==========================
``target_kind``         resolution            count (world_a)
======================  ====================  ==========================
``event``               ``latent_id``          709
``entity``              entity key -> entity  248
``window``              time interval          16
``source_window``       source + time interval  1
======================  ====================  ==========================

An event can be anomalous for more than one reason (an injected event inside a
collective window is both ``point``-adjacent and ``collective``-adjacent), so
resolution unions the families rather than picking a winner. The ``family``
column therefore holds a sorted, de-duplicated list.
"""

from __future__ import annotations

from mosaic.labels.resolve import (
    AGGREGATE_FAMILIES,
    ANOMALY_FAMILIES,
    EVENT_FAMILIES,
    DEFAULT_SCOPE,
    MINIMUM_FAMILY_SUPPORT,
    LabelResolution,
    LabelScope,
    assert_labels_never_fitted,
    minimum_support,
    resolve_both,
    resolve_labels,
    supported_families,
)

__all__ = [
    "AGGREGATE_FAMILIES",
    "ANOMALY_FAMILIES",
    "DEFAULT_SCOPE",
    "EVENT_FAMILIES",
    "MINIMUM_FAMILY_SUPPORT",
    "LabelResolution",
    "LabelScope",
    "assert_labels_never_fitted",
    "minimum_support",
    "resolve_both",
    "resolve_labels",
    "supported_families",
]
