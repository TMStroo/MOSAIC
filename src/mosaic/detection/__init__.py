"""Detector base contract, statistical detectors, and threshold selection.

Import order note: ``statistical`` and ``thresholds`` are imported after
``base`` because they subclass / reference ``Detector``, and
``thresholds`` pulls ``confusion_counts`` from ``mosaic.metrics`` lazily inside
a function to keep the module graph acyclic.
"""

from __future__ import annotations

from mosaic.detection.base import (
    RESULT_SCHEMA_VERSION,
    Detector,
    DetectorFit,
    ModelMetadata,
    ScoredFrame,
)
from mosaic.detection.statistical import (
    STATISTICAL_DETECTORS,
    ChangePointDetector,
    EWMAResidualDetector,
    IQRFenceDetector,
    RobustZScoreDetector,
    ZScoreDetector,
)
from mosaic.detection.thresholds import (
    FrozenThreshold,
    ThresholdMethod,
    ThresholdResult,
    assert_validation_only,
    freeze,
    select_threshold,
)

__all__ = [
    "RESULT_SCHEMA_VERSION",
    "STATISTICAL_DETECTORS",
    "ChangePointDetector",
    "Detector",
    "DetectorFit",
    "EWMAResidualDetector",
    "FrozenThreshold",
    "IQRFenceDetector",
    "ModelMetadata",
    "RobustZScoreDetector",
    "ScoredFrame",
    "ThresholdMethod",
    "ThresholdResult",
    "ZScoreDetector",
    "assert_validation_only",
    "freeze",
    "select_threshold",
]
