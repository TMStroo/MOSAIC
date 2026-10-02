"""Detector interface and the model metadata every run must record.

Phase 4's contract, stated once so no experiment can quietly bypass it. Every
detector implements three calls:

``fit(train)``
    Estimate whatever state the method needs, from *train only*.
``score(data)``
    Return a per-row score. Higher means more anomalous, always.
``predict(data, threshold)``
    Apply a *frozen* threshold.

Two invariants are enforced here rather than left to reviewer discipline:

1. **Fitted state is stamped.** ``DetectorFit`` records the period it saw and
   the last timestamp it saw, and every ``score`` call re-checks the frame it is
   given against that stamp. A detector fitted through March cannot silently
   score a frame containing February rows and produce optimistic numbers.

2. **A score is not a probability.** ``score`` returns an uncalibrated
   ranking. Only ``predict_proba`` may claim a probability, and only supervised
   and explicitly-calibrated models implement it. ``Detector.predict`` uses the
   score against a threshold; nothing else is permitted to call a score a
   probability.
"""

from __future__ import annotations

import abc
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from mosaic.experiments.protocol import PERIODS, LeakageError, Period
from mosaic.schema.ids import stable_hash

#: Bumped whenever the recorded schema of a result artifact changes, so a
#: reader can tell two artifact generations apart.
RESULT_SCHEMA_VERSION = "m6/1"


@dataclass(frozen=True)
class DetectorFit:
    """Provenance of a fitted detector, permanently bound to its period.

    ``fitted_through`` is the *maximum* training timestamp. ``assert_applies_to``
    refuses a target frame containing anything earlier, which is the failure
    mode that produces quietly optimistic results: fitting on a window that
    overlaps the evaluation period.
    """

    period: Period
    fitted_through: datetime | None
    n_rows: int
    features: tuple[str, ...] = ()

    def assert_applies_to(self, target_period: Period | str, frame: pl.DataFrame | None = None) -> None:
        """Raise :class:`LeakageError` if applying this fit would leak."""
        if target_period == "unassigned":
            raise LeakageError(f"detector fitted on {self.period} cannot score unassigned rows")
        if target_period in PERIODS and PERIODS.index(target_period) < PERIODS.index(self.period):
            raise LeakageError(
                f"detector fitted on {self.period} was asked to score {target_period}: "
                "that is backward in time"
            )
        # The overlap check applies only to periods *after* the fitted one.
        # Scoring `train` with a train-fitted detector is legitimate and is
        # required to report train-period metrics; what must never happen is a
        # later period containing rows the fit already saw.
        is_later = target_period in PERIODS and PERIODS.index(target_period) > PERIODS.index(self.period)
        has_rows = (
            frame is not None
            and "timestamp" in frame.columns
            and not frame.is_empty()
        )
        if is_later and has_rows and self.fitted_through is not None:
            earliest = frame["timestamp"].min()
            if earliest is not None and earliest < self.fitted_through:
                raise LeakageError(
                    f"detector fitted through {self.fitted_through} but {target_period} "
                    f"contains a row from {earliest}; fit/evaluation overlap"
                )


@dataclass
class ModelMetadata:
    """Everything a result artifact must carry to be auditable.

    Recorded whether or not the value is interesting. In particular
    ``deterministic`` is an explicit claim, so a stochastic model can never be
    silently presented as reproducible.
    """

    model_id: str
    model_type: str
    label_protocol: str
    feature_set_id: str
    dataset_version: str
    feature_set_version: str
    training_period: str
    training_rows: int
    training_start: str | None
    training_end: str | None
    validation_period: str
    random_seed: int
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    deterministic: bool = False
    fitted_features: tuple[str, ...] = ()
    n_parameters: int | None = None
    fit_seconds: float = 0.0
    # Populated by ``to_dict`` rather than at construction: the hash is of the
    # file as it stands when the artifact is written, so a cached value could go
    # stale relative to the code that produced the record.
    code_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["fitted_features"] = list(self.fitted_features)
        out["code_version"] = _code_version()
        return out

    def fingerprint(self) -> str:
        return stable_hash(self.to_dict(), length=16)


def _code_version() -> str:
    """Content hash of the detector source, so artifacts name the code that made them."""
    try:
        source = Path(__file__).read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - source unavailable (zip install)
        return "unknown"
    return stable_hash(source, platform.python_version(), length=12)


@dataclass
class ScoredFrame:
    """A scored frame plus the metadata needed to interpret the scores."""

    frame: pl.DataFrame
    metadata: ModelMetadata
    score_column: str = "score"

    def scores(self) -> pl.Series:
        if self.score_column not in self.frame.columns:
            raise KeyError(f"score column {self.score_column!r} missing from scored frame")
        return self.frame[self.score_column]

    def predictions(self, threshold: float) -> pl.Series:
        """Apply a frozen threshold. Higher score = flagged.

        The comparison is ``>=`` so a threshold selected as one of the observed
        scores flags the rows that produced it.
        """
        return self.scores() >= threshold


class Detector(abc.ABC):
    """Common lifecycle for every statistical detector and ML model."""

    #: Stable identifier fragment, part of ``model_id``.
    model_type: str = "abstract"
    #: Whether ``random_seed`` has any effect on output.
    deterministic: bool = False
    #: Whether the score is a calibrated probability.
    produces_probability: bool = False

    def __init__(
        self,
        *,
        feature_columns: Sequence[str],
        label_protocol: str,
        feature_set_id: str,
        dataset_version: str,
        feature_set_version: str,
        random_seed: int = 0,
        hyperparameters: Mapping[str, Any] | None = None,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns must be non-empty")
        self.feature_columns = tuple(feature_columns)
        self.label_protocol = label_protocol
        self.feature_set_id = feature_set_id
        self.dataset_version = dataset_version
        self.feature_set_version = feature_set_version
        self.random_seed = random_seed
        self.hyperparameters: dict[str, Any] = dict(hyperparameters or {})
        self._fit: DetectorFit | None = None
        self._metadata: ModelMetadata | None = None

    # -- lifecycle ---------------------------------------------------------
    @abc.abstractmethod
    def _fit_impl(self, train: pl.DataFrame, period: Period) -> None:
        """Estimate state from *train only*."""

    @abc.abstractmethod
    def _score_impl(self, data: pl.DataFrame) -> pl.Series:
        """Return a per-row score aligned to ``data``'s row order."""

    def fit(self, train: pl.DataFrame, *, period: Period = "train", timestamp_column: str = "timestamp") -> ModelMetadata:
        """Fit on ``train`` and stamp the fit. Returns the run's metadata."""
        if period != "train":
            # Fitting anywhere other than train is the one error that cannot be
            # caught later, so it is refused at the boundary rather than warned
            # about in a report nobody reads.
            raise LeakageError(
                f"refusing to fit on period {period!r}: detectors may only fit on 'train'; "
                "validation is for threshold selection, backtest/forward for evaluation"
            )
        missing = [c for c in self.feature_columns if c not in train.columns]
        if missing:
            raise ValueError(f"train frame is missing feature columns: {missing}")
        start = time.perf_counter()
        self._fit_impl(train, period)
        elapsed = time.perf_counter() - start

        fitted_through = None
        if timestamp_column in train.columns and not train.is_empty():
            value = train[timestamp_column].max()
            fitted_through = value if isinstance(value, datetime) else None
        self._fit = DetectorFit(
            period=period,
            fitted_through=fitted_through,
            n_rows=train.height,
            features=self.feature_columns,
        )
        self._metadata = ModelMetadata(
            model_id=self._model_id(),
            model_type=self.model_type,
            label_protocol=self.label_protocol,
            feature_set_id=self.feature_set_id,
            dataset_version=self.dataset_version,
            feature_set_version=self.feature_set_version,
            training_period=period,
            training_rows=train.height,
            training_start=str(train[timestamp_column].min()) if timestamp_column in train.columns and not train.is_empty() else None,
            training_end=str(train[timestamp_column].max()) if timestamp_column in train.columns and not train.is_empty() else None,
            validation_period="validation",
            random_seed=self.random_seed,
            hyperparameters=self.hyperparameters,
            deterministic=self.deterministic,
            fitted_features=self.feature_columns,
            fit_seconds=elapsed,
        )
        return self._metadata

    def score(self, data: pl.DataFrame, *, target_period: Period | str = "forward") -> ScoredFrame:
        """Score ``data``. Refuses if the fit does not legitimately apply."""
        if self._fit is None:
            raise LeakageError(
                f"{self.model_type} was asked to score before fit; fitting is mandatory "
                "and must happen on 'train'"
            )
        missing = [c for c in self.feature_columns if c not in data.columns]
        if missing:
            raise ValueError(f"scored frame is missing feature columns: {missing}")
        self._fit.assert_applies_to(target_period, data)

        values = self._score_impl(data)
        if len(values) != data.height:
            raise ValueError(
                f"{self.model_type}.score returned {len(values)} values for {data.height} rows; "
                "scores must align with input row order"
            )
        if self._metadata is None:  # pragma: no cover - set by fit
            raise LeakageError("metadata missing; fit() did not complete")

        scored = data.with_columns(pl.Series("score", values))
        return ScoredFrame(frame=scored, metadata=self._metadata)

    def predict(self, data: pl.DataFrame, threshold: float, *, target_period: Period | str = "forward") -> ScoredFrame:
        """Score then apply a frozen threshold."""
        scored = self.score(data, target_period=target_period)
        scored.frame = scored.frame.with_columns(
            scored.predictions(threshold).alias("predicted_anomaly")
        )
        return scored

    def predict_proba(self, data: pl.DataFrame, *, target_period: Period | str = "forward") -> ScoredFrame:
        """Only for models whose score is genuinely a probability."""
        raise TypeError(
            f"{self.model_type}.score is an uncalibrated ranking, not a probability. "
            "Use a supervised model or an explicitly calibrated detector."
        )

    # -- identity ----------------------------------------------------------
    def _model_id(self) -> str:
        return f"mdl_{stable_hash(self.model_type, self.label_protocol, self.feature_set_id, self.feature_set_version, self.hyperparameters, self.random_seed, length=16)}"

    @property
    def fitted(self) -> DetectorFit | None:
        return self._fit

    @property
    def metadata(self) -> ModelMetadata | None:
        return self._metadata
