"""Detector base contract, statistical detectors, ML baselines, threshold selection.

Import order note: ``statistical``, ``ml`` and ``thresholds`` are imported after
``base`` because they subclass / reference ``Detector``. That module graph is
acyclic -- ``mosaic.metrics`` imports nothing from ``mosaic`` -- so
``thresholds`` can import ``confusion_counts`` at module level.
"""

from __future__ import annotations

from mosaic.detection.base import (
    RESULT_SCHEMA_VERSION,
    Detector,
    DetectorFit,
    ModelMetadata,
    ScoredFrame,
)
from mosaic.detection.ml import (
    ML_MODELS,
    SUPERVISED_MODELS,
    UNSUPERVISED_MODELS,
    MLDecisionTree,
    MLDummyClassifier,
    MLGradientBoosting,
    MLIsolationForest,
    MLLocalOutlierFactor,
    MLLogisticRegression,
    MLRandomForest,
    model_specs,
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
    "ML_MODELS",
    "RESULT_SCHEMA_VERSION",
    "STATISTICAL_DETECTORS",
    "SUPERVISED_MODELS",
    "UNSUPERVISED_MODELS",
    "ChangePointDetector",
    "Detector",
    "DetectorFit",
    "EWMAResidualDetector",
    "FrozenThreshold",
    "IQRFenceDetector",
    "MLDecisionTree",
    "MLDummyClassifier",
    "MLGradientBoosting",
    "MLIsolationForest",
    "MLLocalOutlierFactor",
    "MLLogisticRegression",
    "MLRandomForest",
    "ModelMetadata",
    "RobustZScoreDetector",
    "ScoredFrame",
    "ThresholdMethod",
    "ThresholdResult",
    "ZScoreDetector",
    "assert_validation_only",
    "freeze",
    "model_specs",
    "select_threshold",
]
