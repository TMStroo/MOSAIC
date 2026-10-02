"""Statistical anomaly detectors, each with its assumptions stated.

Every detector here is *deterministic*: no sampling, no seed dependence. That is
recorded explicitly on the model metadata so a stochastic model can never be
presented as reproducible by omission.

For each: input, fitted state, output score, threshold semantics, complexity,
leakage risks, and limitations are in the class docstrings rather than in prose
elsewhere, so they cannot drift from the implementation.

None of these produce a probability. ``|z|`` is a standardised deviation, not a
likelihood; calling it one would make a 3-sigma event look 99.7% certain to be
anomalous, which is a statement about a Gaussian tail, not about the data.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from mosaic.detection.base import Detector
from mosaic.experiments.protocol import Period

#: Tukey's constant for the 1.5*IQR fence.
TUKEY_K = 1.5


def _require_finite(values: pl.Series, name: str, owner: str) -> None:
    """Reject an all-null feature, but allow an empty frame.

    The length check comes first because an empty frame trivially satisfies
    ``null_count() == len()``; scoring nothing must return nothing rather than
    raise as if the feature were missing.
    """
    if values.is_empty():
        return
    if values.null_count() == values.len():
        raise ValueError(f"{owner}: feature {name!r} is entirely null in the scored frame")


class ZScoreDetector(Detector):
    """Global z-score against the training mean and standard deviation.

    Input
        One numeric feature.
    Fitted state
        ``mean``, ``sd`` from train (n, ddof=1).
    Output score
        ``abs(x - mean) / sd``. Absolute, so a value far *below* the mean is as
        anomalous as one far above -- necessary because injected point
        anomalies move in both directions.
    Threshold semantics
        Absolute deviation in standard deviations. 3.0 flags ~0.27% of a
        Gaussian population.
    Complexity
        O(n) time, O(1) state.
    Leakage risks
        None beyond fitting on train. The mean is the weak point: injected point
        anomalies inflate it, which is why :class:`RobustZScoreDetector` exists.
    Limitations
        Assumes approximate symmetry and a light tail. Sensitive to outliers in
        the *training* set. A single event type with a bimodal history will be
        scored poorly regardless of its anomaly.
    """

    model_type = "stat_zscore"
    deterministic = True

    def _fit_impl(self, train: pl.DataFrame, period: Period) -> None:
        column = self.feature_columns[0]
        series = train[column].drop_nulls()
        if series.is_empty():
            raise ValueError(f"stat_zscore: feature {column!r} has no non-null training values")
        self._mean = float(series.mean())
        # `std(ddof=1)` is None for a single observation and 0.0 for a constant
        # one. Both mean "no scale to normalise by", so both take the same
        # branch: score by raw deviation instead of dividing by a missing or
        # zero denominator.
        raw_sd = series.std(ddof=1)
        sd = float(raw_sd) if raw_sd is not None else 0.0
        # A constant feature has no spread to normalise by. Rather than divide by
        # zero, the score degenerates to "distance from the constant", which is
        # the correct limit and keeps the detector usable on e.g. a flag column.
        self._sd = sd if sd > 0 else None

    def _score_impl(self, data: pl.DataFrame) -> pl.Series:
        column = self.feature_columns[0]
        _require_finite(data[column], column, self.model_type)
        if self._sd is None:
            return (data[column] - self._mean).abs().fill_null(0.0)
        return ((data[column] - self._mean).abs() / self._sd).fill_null(0.0)


class RobustZScoreDetector(Detector):
    """Median/MAD z-score -- the contaminated-mean problem, solved.

    Input
        One numeric feature.
    Fitted state
        ``median`` and ``median_absolute_deviation`` from train. The MAD is
        scaled by 1.4826 so that under a normal distribution it estimates the
        same quantity as the standard deviation; without that constant a
        robust-z of 3.0 would mean something different from a z of 3.0.
    Output score
        ``abs(x - median) / (1.4826 * MAD)``.
    Threshold semantics
        Robust standard deviations. Comparable to :class:`ZScoreDetector`
        under normality, far more trustworthy under contamination.
    Complexity
        O(n log n) for the median, O(n) thereafter, O(1) state.
    Leakage risks
        As above: train-only quantiles.
    Limitations
        A MAD of exactly 0 (more than half the training values identical)
        leaves no scale; the score then falls back to raw deviation, which is
        scale-free but not comparable to any other feature.
    """

    model_type = "stat_robust_z"
    deterministic = True

    def _fit_impl(self, train: pl.DataFrame, period: Period) -> None:
        column = self.feature_columns[0]
        series = train[column].drop_nulls()
        if series.is_empty():
            raise ValueError(f"stat_robust_z: feature {column!r} has no non-null training values")
        self._median = float(series.median())
        mad = float((series.abs() - self._median).median())
        self._scale = mad * 1.4826 if mad > 0 else None

    def _score_impl(self, data: pl.DataFrame) -> pl.Series:
        column = self.feature_columns[0]
        _require_finite(data[column], column, self.model_type)
        if self._scale is None:
            return (data[column] - self._median).abs().fill_null(0.0)
        return ((data[column] - self._median).abs() / self._scale).fill_null(0.0)


class IQRFenceDetector(Detector):
    """Tukey fences: flag values outside ``[Q1 - k*IQR, Q3 + k*IQR]``.

    Input
        One numeric feature.
    Fitted state
        ``q1``, ``q3`` from train.
    Output score
        The *excess* beyond the nearer fence, in IQR units, and 0 inside the
        fences. Deliberately 0 rather than small-but-positive inside: a score
        that varies inside the fence would let a threshold below the fence
        flag arbitrary interior points.
    Threshold semantics
        Excess in IQR units above the fence. 0.0 means "flag everything", so a
        threshold must be > 0.
    Complexity
        O(n log n) to quantify, O(1) state.
    Leakage risks
        Train-only quantiles.
    Limitations
        Asymmetric distributions are handled badly -- a right-skewed feature
        will have a high fence and a low one that never triggers. The classic
        1.5 multiplier flags ~0.7% of a normal sample, but that is an
        assumption, not a guarantee for any real feature.
    """

    model_type = "stat_iqr"
    deterministic = True

    def __init__(self, *, k: float = TUKEY_K, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if k <= 0:
            raise ValueError(f"iqr fence multiplier k must be positive, got {k}")
        self.k = k
        self.hyperparameters["k"] = k

    def _fit_impl(self, train: pl.DataFrame, period: Period) -> None:
        column = self.feature_columns[0]
        series = train[column].drop_nulls()
        if series.is_empty():
            raise ValueError(f"stat_iqr: feature {column!r} has no non-null training values")
        self._q1 = float(series.quantile(0.25))
        self._q3 = float(series.quantile(0.75))
        self._iqr = self._q3 - self._q1
        self._low = self._q1 - self.k * self._iqr
        self._high = self._q3 + self.k * self._iqr

    def _score_impl(self, data: pl.DataFrame) -> pl.Series:
        column = self.feature_columns[0]
        _require_finite(data[column], column, self.model_type)
        if self._iqr <= 0:
            return (data[column].clip(self._low, self._high) - data[column]).abs().fill_null(0.0)
        above = (data[column] - self._high) / self._iqr
        below = (self._low - data[column]) / self._iqr
        # `max_horizontal` returns an expression; the contract is a Series
        # aligned to input row order, so it is materialised here.
        return pl.select(
            pl.max_horizontal(above, below).clip(lower_bound=0.0).fill_null(0.0).alias("s")
        )["s"]


class EWMAResidualDetector(Detector):
    """Absolute residual from a train-fitted exponentially weighted mean.

    Input
        One numeric feature, already in a causal (per-event) order. The feature
        pipeline guarantees this: features at time *t* use only data up to *t*.
    Fitted state
        The EWMA level at the end of train, plus the smoothing factor.
    Output score
        ``abs(x - level)``, updated causally -- the level advances with each
        row, so an event is scored against the state *before* it.
    Threshold semantics
        Absolute deviation in the feature's own units. Not scale-free, so a
        threshold from this detector is not comparable to a z-score threshold.
    Complexity
        O(n) time, O(1) state.
    Leakage risks
        The level must start from train and be carried forward, never refitted
        per period. Refitting on validation or forward would reset the level
        using future information and hide exactly the drift this detector
        exists to catch. This implementation therefore updates state in
        ``_score_impl`` and never re-estimates it.
    Limitations
        A single level cannot represent trend: a steady ramp makes every
        residual small and a sudden level change invisible. ``alpha`` trades
        responsiveness against noise. Because the score is in raw units, a
        slowly drifting series will drift its own threshold.
    """

    model_type = "stat_ewma"
    deterministic = True

    def __init__(self, *, alpha: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"ewma alpha must lie in (0, 1], got {alpha}")
        self.alpha = alpha
        self.hyperparameters["alpha"] = alpha

    def _fit_impl(self, train: pl.DataFrame, period: Period) -> None:
        column = self.feature_columns[0]
        series = train[column].drop_nulls()
        if series.is_empty():
            raise ValueError(f"stat_ewma: feature {column!r} has no non-null training values")
        # Carry the level through the whole training period rather than starting
        # at the mean, so the state handed to the next period reflects the
        # series' actual recent level.
        level = float(series[0])
        # Direct Series iteration: `iter_rows` is a DataFrame method, and the
        # Series is already null-free here.
        for value in series:
            level = (1.0 - self.alpha) * level + self.alpha * float(value)
        self._level = level

    def _score_impl(self, data: pl.DataFrame) -> pl.Series:
        column = self.feature_columns[0]
        _require_finite(data[column], column, self.model_type)
        out: list[float] = []
        level = self._level
        alpha = self.alpha
        for raw in data[column]:
            if raw is None:
                out.append(0.0)
                continue
            value = float(raw)
            residual = abs(value - level)
            level = (1.0 - alpha) * level + alpha * value
            out.append(residual)
        self._level = level
        return pl.Series(out, dtype=pl.Float64)


class ChangePointDetector(Detector):
    """CUSUM-style two-sided change point, over a train-fitted reference scale.

    Input
        One numeric feature in causal order.
    Fitted state
        The train median and a train-fitted robust scale (MAD * 1.4826), used to
        make the statistic scale-free. Falling back to a fixed scale when the
        MAD is 0 would produce an arbitrary statistic.
    Output score
        The CUSUM statistic itself -- the largest accumulated evidence that the
        mean has shifted -- expressed in reference-scale units.
    Threshold semantics
        CUSUM units. The classic decision interval ``h`` is expressed in
        multiples of the reference scale. **Monotone, not a point score**: a
        change point raises the statistic for every subsequent row until the
        statistic is reset, so this detector reports "the process has been off
        baseline since some point", not "this row is anomalous". That is the
        intended semantics for a change point and is why its precision is not
        comparable to a per-event detector's.
    Complexity
        O(n) time, O(1) state.
    Leakage risks
        The reference scale comes from train only. The CUSUM itself advances
        causally and is never back-fitted.
    Limitations
        Cannot localise the change point in the output (only the accumulated
        evidence). Drifts slowly upward, so a long benign regime eventually
        crosses any threshold. Consecutive change points accumulate.
    """

    model_type = "stat_changepoint"
    deterministic = True

    def __init__(self, *, drift: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if drift < 0:
            raise ValueError(f"cusum drift allowance must be >= 0, got {drift}")
        self.drift = drift
        self.hyperparameters["drift"] = drift

    def _fit_impl(self, train: pl.DataFrame, period: Period) -> None:
        column = self.feature_columns[0]
        series = train[column].drop_nulls()
        if series.is_empty():
            raise ValueError(f"stat_changepoint: feature {column!r} has no non-null training values")
        self._centre = float(series.median())
        mad = float((series.abs() - self._centre).median())
        scale = mad * 1.4826
        if scale <= 0:
            # Same None-for-one-observation case as the z-score detector.
            raw_std = series.std(ddof=1)
            scale = float(raw_std) if raw_std is not None else 0.0
        self._scale = scale

    def _score_impl(self, data: pl.DataFrame) -> pl.Series:
        column = self.feature_columns[0]
        _require_finite(data[column], column, self.model_type)
        if self._scale <= 0:
            raise ValueError(
                f"stat_changepoint: feature {column!r} has zero variance in train, "
                "so no change point is definable"
            )
        centre, scale, drift = self._centre, self._scale, self.drift
        pos = neg = 0.0
        out: list[float] = []
        for value in data[column]:
            if value is None:
                out.append(0.0)
                continue
            z = (float(value) - centre) / scale
            pos = max(0.0, pos + z - drift)
            neg = max(0.0, neg - z - drift)
            out.append(max(pos, neg))
        return pl.Series(out, dtype=pl.Float64)


#: Registered detectors, in reporting order. Names are the ``--detector`` values
#: used by experiment configs.
STATISTICAL_DETECTORS: dict[str, type[Detector]] = {
    "zscore": ZScoreDetector,
    "robust_z": RobustZScoreDetector,
    "iqr": IQRFenceDetector,
    "ewma": EWMAResidualDetector,
    "changepoint": ChangePointDetector,
}
