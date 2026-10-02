"""ML baselines: supervised classifiers and unsupervised anomaly detectors.

Design rules, each of which exists because of something that went wrong in
Phase 2:

1. **Preprocessing is fitted on train only, and is part of the fitted state.**
   The imputer, the scaler and the column selection are estimated from train
   and then frozen. An imputer fitted on validation would leak the validation
   distribution into every later period.

2. **A probability is only exposed when there is one.** ``predict_proba``
   exists on the classifiers, which are genuinely probabilistic.
   ``IsolationForest`` and ``LocalOutlierFactor`` expose ``predict`` only:
   sklearn's own ``score_samples`` output is a *decision function* whose sign
   convention is an implementation detail, so it is converted to a bounded,
   explicitly-uncalibrated score and never presented as a probability.
   Calling ``predict_proba`` on an unsupervised detector raises.

3. **Score direction is uniform: higher means more anomalous.** LocalOutlierFactor
   natively prefers *normal* points, so its score is negated at the boundary.

4. **Seeds are recorded and respected.** Every stochastic model takes
   ``random_seed`` and passes it to sklearn; ``deterministic`` is set from what
   the model actually does, not from what is convenient.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mosaic.detection.base import Detector, ScoredFrame
from mosaic.experiments.protocol import Period


class _SklearnMixin(Detector):
    """Shared plumbing: numpy conversion and a train-fitted preprocessing chain.

    The chain is deliberately simple and explicit -- median impute, then
    optional standardisation -- because a model comparison is only meaningful
    if the preprocessing is identical across models. Anything cleverer would
    confound the model difference with the preprocessing difference.
    """

    #: Whether to standardise. Trees do not need it; distance-based models do,
    #: and sharing one setting across all models would be its own confound.
    standardize: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pipeline: Pipeline | None = None

    # -- conversion ---------------------------------------------------------
    def _to_matrix(self, data: pl.DataFrame) -> np.ndarray:
        """Feature columns as a float matrix, preserving row order.

        ``nan_to_num`` is deliberately *not* used: the imputer handles missing
        values and silently substituting 0.0 for them would conflate "missing"
        with "zero", which for several of these features is a real and very
        different value.
        """
        block = data.select(self.feature_columns)
        return np.column_stack([block[c].cast(pl.Float64).to_numpy() for c in self.feature_columns])

    def _build_pipeline(self, estimator: Any) -> Pipeline:
        steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
        if self.standardize:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", estimator))
        return Pipeline(steps)

    # -- lifecycle ----------------------------------------------------------
    def _fit_impl(self, train: pl.DataFrame, period: Period) -> None:
        X = self._to_matrix(train)
        y = train["is_anomaly"].cast(pl.Boolean).to_numpy().astype(int)
        self._pipeline = self._build_pipeline(self._make_estimator())
        self._pipeline.fit(X, y)

    def _score_impl(self, data: pl.DataFrame) -> pl.Series:
        if self._pipeline is None:
            raise RuntimeError(f"{self.model_type} scored before fit")
        X = self._to_matrix(data)
        if data.is_empty():
            return pl.Series([], dtype=pl.Float64)
        return pl.Series(self._raw_scores(X), dtype=pl.Float64)

    @abc.abstractmethod
    def _make_estimator(self) -> Any:
        """The sklearn estimator, constructed with this run's seed."""

    @abc.abstractmethod
    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""


# =============================================================================
# Supervised
# =============================================================================


class MLDummyClassifier(_SklearnMixin):
    """Predicts the training majority class.

    Included as the floor of the supervised comparison. On a 0.03% positive
    rate it predicts *everything* negative, which makes it the reference point
    for "what does a model achieve by learning nothing". A model that cannot
    beat this has extracted nothing.
    """

    model_type: str = "ml_dummy"
    deterministic: bool = True
    produces_probability: bool = True
    standardize: bool = False

    def _make_estimator(self) -> Any:
        return DummyClassifier(strategy="most_frequent")

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        proba = self._pipeline.predict_proba(X)  # type: ignore[union-attr]
        # Column 1 is P(positive) for the binary case; a one-column output means
        # the training set contained a single class and every row is negative.
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else np.zeros(len(X))

    def predict_proba(
        self, data: pl.DataFrame, *, target_period: Period | str = "forward"
    ) -> ScoredFrame:
        """Genuine probabilities: the dummy emits its observed class frequency."""
        return self.score(data, target_period=target_period)


class MLLogisticRegression(_SklearnMixin):
    """L1/elastic-net logistic regression.

    The interpretable linear baseline: if a linear model on these features
    cannot beat the statistical floor, the signal is not linearly present.
    """

    model_type: str = "ml_logistic"
    deterministic: bool = False
    produces_probability: bool = True

    def _make_estimator(self) -> Any:
        params = dict(self.hyperparameters)
        return LogisticRegression(
            max_iter=params.pop("max_iter", 1000),
            class_weight=params.pop("class_weight", None),
            solver=params.pop("solver", "lbfgs"),
            random_state=self.random_seed if params.pop("use_seed", False) else None,
            **params,
        )

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        proba = self._pipeline.predict_proba(X)  # type: ignore[union-attr]
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else np.zeros(len(X))

    def predict_proba(
        self, data: pl.DataFrame, *, target_period: Period | str = "forward"
    ) -> ScoredFrame:
        """Calibrated by the fitted classifier; returns the same scores as ``score``."""
        return self.score(data, target_period=target_period)


class MLDecisionTree(_SklearnMixin):
    """A single interpretable tree.

    Included to separate "the features carry signal" from "an ensemble averages
    that signal away". Its depth also shows how far the data supports splitting
    at all: on 76 training positives a deep tree will memorise them.
    """

    model_type: str = "ml_tree"
    deterministic: bool = True
    produces_probability: bool = True
    standardize: bool = False

    def _make_estimator(self) -> Any:
        params = dict(self.hyperparameters)
        return DecisionTreeClassifier(
            max_depth=params.pop("max_depth", None),
            min_samples_leaf=params.pop("min_samples_leaf", 1),
            class_weight=params.pop("class_weight", None),
            random_state=self.random_seed,
            **params,
        )

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        proba = self._pipeline.predict_proba(X)  # type: ignore[union-attr]
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else np.zeros(len(X))

    def predict_proba(
        self, data: pl.DataFrame, *, target_period: Period | str = "forward"
    ) -> ScoredFrame:
        """Calibrated by the fitted classifier; returns the same scores as ``score``."""
        return self.score(data, target_period=target_period)


class MLRandomForest(_SklearnMixin):
    """Bagged trees.

    The first model that can actually use interaction between features, and
    robust to the scale and skew of the causal feature block.
    """

    model_type: str = "ml_random_forest"
    deterministic: bool = False
    produces_probability: bool = True
    standardize: bool = False

    def _make_estimator(self) -> Any:
        params = dict(self.hyperparameters)
        return RandomForestClassifier(
            n_estimators=params.pop("n_estimators", 200),
            max_depth=params.pop("max_depth", None),
            min_samples_leaf=params.pop("min_samples_leaf", 1),
            max_features=params.pop("max_features", "sqrt"),
            class_weight=params.pop("class_weight", None),
            n_jobs=params.pop("n_jobs", -1),
            random_state=self.random_seed,
            **params,
        )

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        proba = self._pipeline.predict_proba(X)  # type: ignore[union-attr]
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else np.zeros(len(X))

    def predict_proba(
        self, data: pl.DataFrame, *, target_period: Period | str = "forward"
    ) -> ScoredFrame:
        """Calibrated by the fitted classifier; returns the same scores as ``score``."""
        return self.score(data, target_period=target_period)


class MLGradientBoosting(_SklearnMixin):
    """Boosted trees, sequential rather than parallel.

    Included because it is the strongest of the permitted supervised baselines
    and the natural upper bound for "classical ML on these features". If it
    cannot beat the statistical floor, that is a statement about the features,
    not about model capacity.
    """

    model_type: str = "ml_gradient_boosting"
    deterministic: bool = False
    produces_probability: bool = True
    standardize: bool = False

    def _make_estimator(self) -> Any:
        params = dict(self.hyperparameters)
        return GradientBoostingClassifier(
            n_estimators=params.pop("n_estimators", 100),
            max_depth=params.pop("max_depth", 3),
            learning_rate=params.pop("learning_rate", 0.1),
            random_state=self.random_seed,
            **params,
        )

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        proba = self._pipeline.predict_proba(X)  # type: ignore[union-attr]
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else np.zeros(len(X))

    def predict_proba(
        self, data: pl.DataFrame, *, target_period: Period | str = "forward"
    ) -> ScoredFrame:
        """Calibrated by the fitted classifier; returns the same scores as ``score``."""
        return self.score(data, target_period=target_period)


# =============================================================================
# Unsupervised
# =============================================================================


class _UnsupervisedMixin(Detector):
    """Shared contract for models with no labels in the fit at all.

    Two properties are enforced rather than documented:

    * ``predict_proba`` raises. sklearn's ``score_samples`` is a decision
      function on an arbitrary scale, not a probability, and Phase 2 already
      established that presenting a score as a probability misleads.
    * ``fit`` still refuses non-train periods, even though an unsupervised fit
      has no label to leak. The period rule is about the *data*, and letting an
      unsupervised model differ here would make the two families incomparable.
    """

    produces_probability: bool = False

    #: The fitted sklearn estimator. Declared here because both unsupervised
    #: models use the same attribute name and the base class has no opinion
    #: about it.
    _estimator: Any = None

    def predict_proba(
        self, data: pl.DataFrame, *, target_period: Period | str = "forward"
    ) -> ScoredFrame:
        raise TypeError(
            f"{self.model_type} is unsupervised and does not produce probabilities; "
            "its score is an uncalibrated anomaly strength"
        )

    #: Frozen at fit; applied identically at fit and at score.
    _imputer: Any = None
    _scaler: Any = None

    def _transform(self, data: pl.DataFrame) -> np.ndarray:
        """Feature matrix with the *train-fitted* imputer and scaler applied.

        Both steps are re-applied at score time, not just at fit: a later period
        can contain nulls the training period never had, and passing those to
        sklearn raises. Using the frozen objects (rather than refitting) is what
        keeps the preprocessing causal.
        """
        X = self._to_matrix(data)
        if self._imputer is None or self._scaler is None:
            raise RuntimeError(f"{self.model_type} scored before fit")
        return self._scaler.transform(self._imputer.transform(X))

    def _score_impl(self, data: pl.DataFrame) -> pl.Series:
        if self._estimator is None:
            raise RuntimeError(f"{self.model_type} scored before fit")
        if data.is_empty():
            return pl.Series([], dtype=pl.Float64)
        return pl.Series(self._raw_scores(self._transform(data)), dtype=pl.Float64)

    @abc.abstractmethod
    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous, in ``[0, 1]``, *not* a probability."""

    def _to_matrix(self, data: pl.DataFrame) -> np.ndarray:
        block = data.select(self.feature_columns)
        return np.column_stack([block[c].cast(pl.Float64).to_numpy() for c in self.feature_columns])


class MLIsolationForest(_UnsupervisedMixin):
    """Isolation Forest.

    Chosen as the primary unsupervised baseline because it is the standard
    reference point: cheap, deterministic given a seed, and it makes no
    distributional assumption. Its score is converted to ``[0, 1]`` by a
    monotone map for readability only -- the ordering is what threshold
    selection uses, and the absolute value means nothing.
    """

    model_type: str = "ml_isolation_forest"
    deterministic: bool = False
    standardize: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._estimator: IsolationForest | None = None

    def _fit_impl(self, train: pl.DataFrame, period: Period) -> None:
        # The imputer/scaler are fitted here, on train, and then frozen -- the
        # same discipline the supervised pipeline follows.
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(self._imputer.fit_transform(self._to_matrix(train)))
        params = dict(self.hyperparameters)
        self._estimator = IsolationForest(
            n_estimators=params.pop("n_estimators", 200),
            max_samples=params.pop("max_samples", "auto"),
            contamination=params.pop("contamination", "auto"),
            max_features=params.pop("max_features", 1.0),
            n_jobs=params.pop("n_jobs", -1),
            random_state=self.random_seed,
            **params,
        )
        self._estimator.fit(X)

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        # `decision_function` is higher for *more normal*. Negate so that higher
        # means more anomalous everywhere, matching the supervised models.
        raw = -self._estimator.decision_function(X)  # type: ignore[union-attr]
        # Map to (0, 1) by rank so the value is bounded and seed-stable in
        # ordering. This is a rescaling, NOT a probability.
        return _rank_scale(raw)


class MLLocalOutlierFactor(_UnsupervisedMixin):
    """Local Outlier Factor.

    Included as the second unsupervised baseline because it makes the opposite
    assumption to Isolation Forest: it is *local*, so it scores a point by how
    isolated it is relative to its own neighbourhood. If the anomalies form
    dense clusters rather than sparse points, LOF should find them and IF
    should not. That contrast is the experiment.

    ``novelty=True`` is required to score unseen data. It also means LOF cannot
    be asked for a probability, and its native ``negative_outlier_factor_``
    is a training-set statistic that would leak the fit if reused.
    """

    model_type: str = "ml_lof"
    deterministic: bool = False
    standardize: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._estimator: LocalOutlierFactor | None = None

    def _fit_impl(self, train: pl.DataFrame, period: Period) -> None:
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(self._imputer.fit_transform(self._to_matrix(train)))
        params = dict(self.hyperparameters)
        self._estimator = LocalOutlierFactor(
            n_neighbors=params.pop("n_neighbors", 20),
            contamination=params.pop("contamination", "auto"),
            metric=params.pop("metric", "euclidean"),
            n_jobs=params.pop("n_jobs", -1),
            novelty=True,
            **params,
        )
        self._estimator.fit(X)

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        # LOF's `decision_function` is positive for inliers, like IF, so the
        # same negation applies.
        raw = -self._estimator.decision_function(X)  # type: ignore[union-attr]
        return _rank_scale(raw)


def _rank_scale(values: np.ndarray) -> np.ndarray:
    """Map to ``(0, 1)`` by rank. Order-preserving, so thresholds are unaffected.

    This is emphatically not a probability: it says only where a row sits among
    the rows being scored, which is why ``predict_proba`` raises on these models.

    Ties receive the *average* of the ranks they span, which is what makes the
    mapping a genuine tie-aware ranking rather than an arbitrary tiebreak: two
    rows the model considers equally anomalous get the same score, so a
    threshold cannot separate them.
    """
    if values.size == 0:
        return values.astype(float)
    # Average-rank, the same convention as scipy.stats.rankdata / PR-AUC.
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.arange(1, values.size + 1, dtype=float)
    # For each run of equal values, replace every rank in the run with the mean.
    start = 0
    for i in range(1, values.size + 1):
        if i == values.size or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[start:i] = ranks[start:i].mean()
            start = i
    out = np.empty(values.size, dtype=float)
    out[order] = ranks
    return out / values.size


#: Registry consumed by the experiment runners. Order is the reported order.
SUPERVISED_MODELS: dict[str, type[_SklearnMixin]] = {
    "dummy": MLDummyClassifier,
    "logistic": MLLogisticRegression,
    "decision_tree": MLDecisionTree,
    "random_forest": MLRandomForest,
    "gradient_boosting": MLGradientBoosting,
}

UNSUPERVISED_MODELS: dict[str, type[_UnsupervisedMixin]] = {
    "isolation_forest": MLIsolationForest,
    "lof": MLLocalOutlierFactor,
}

ML_MODELS: dict[str, type[Detector]] = {**SUPERVISED_MODELS, **UNSUPERVISED_MODELS}


def model_specs() -> list[dict[str, Any]]:
    """One declarative record per model, for experiment configuration files."""
    specs: list[dict[str, Any]] = []
    for name, cls in ML_MODELS.items():
        specs.append(
            {
                "name": name,
                "model_type": cls.model_type,
                "supervised": issubclass(cls, _SklearnMixin),
                "deterministic": cls.deterministic,
                "produces_probability": cls.produces_probability,
                "standardize": getattr(cls, "standardize", True),
            }
        )
    return specs
