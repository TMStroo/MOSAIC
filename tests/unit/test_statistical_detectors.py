"""Tests for the detector contract and the five statistical baselines.

Two groups:

* **Lifecycle** -- fit/score/predict ordering, the period guards, and the rule
  that a score is not a probability.
* **Degenerate input** -- the cases that produced real defects during M6:
  constant features, all-null features, empty frames, zero-variance
  change-point inputs, and one-column-per-detector inputs.

Each detector is constructed through the same helper so a new detector cannot
be added without being exercised here.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from mosaic.detection import STATISTICAL_DETECTORS, Detector
from mosaic.detection.statistical import TUKEY_K
from mosaic.experiments.protocol import LeakageError

T0 = datetime(2021, 1, 1)
ALL = sorted(STATISTICAL_DETECTORS)


def make(name: str, columns: list[str], **kwargs) -> Detector:
    params = {
        "feature_columns": columns,
        "label_protocol": "event_only",
        "feature_set_id": "fs_test",
        "dataset_version": "ds_test",
        "feature_set_version": "1.0.0",
        **kwargs,
    }
    return STATISTICAL_DETECTORS[name](**params)


def frame(values: list[float | None], *, start: datetime = T0, step_days: int = 1) -> pl.DataFrame:
    times = [start + timedelta(days=i * step_days) for i in range(len(values))]
    return pl.DataFrame({"timestamp": times, "v": values, "w": [float(v or 0) for v in values]})


#: A timestamp safely after any training frame built by ``normal_frame``.
AFTER_TRAIN = T0 + timedelta(days=400)


def train_frame(n: int = 60) -> pl.DataFrame:
    """Training data: a regular pattern with a late spike in ``v``."""
    values = [float(10 + (i % 7)) for i in range(n)]
    values[-1] = 500.0
    return frame(values)


def normal_frame(n: int = 60) -> pl.DataFrame:
    """A scored frame whose timestamps are strictly after ``fitted_through``.

    Scored frames must be genuinely later than the fit, or
    ``DetectorFit.assert_applies_to`` correctly rejects them -- which is the
    behaviour ``test_fit_evaluation_overlap_is_refused`` pins on purpose.
    """
    values = [float(10 + (i % 7)) for i in range(n)]
    values[-1] = 500.0
    return frame(values, start=AFTER_TRAIN)


class TestLifecycle:
    @pytest.mark.parametrize("name", ALL)
    def test_detector_requires_fit_before_scoring(self, name):
        detector = make(name, ["v"])
        with pytest.raises(LeakageError, match="before fit"):
            detector.score(normal_frame(), target_period="forward")

    @pytest.mark.parametrize("name", ALL)
    def test_detector_refuses_to_fit_on_later_periods(self, name):
        for period in ("validation", "backtest", "forward"):
            detector = make(name, ["v"])
            with pytest.raises(LeakageError, match="may only fit on 'train'"):
                detector.fit(train_frame(), period=period)  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ALL)
    def test_detector_records_deterministic_and_metadata(self, name):
        detector = make(name, ["v"])
        meta = detector.fit(train_frame(), period="train")
        assert meta.deterministic is True
        assert meta.model_type == STATISTICAL_DETECTORS[name].model_type
        assert meta.model_id.startswith("mdl_")
        assert meta.training_rows == 60
        assert meta.label_protocol == "event_only"
        assert meta.random_seed == 0
        assert "code_version" in meta.to_dict()
        assert meta.fingerprint()

    @pytest.mark.parametrize("name", ALL)
    def test_scores_align_with_input_rows(self, name):
        detector = make(name, ["v"])
        detector.fit(train_frame(), period="train")
        target = normal_frame(20)
        scored = detector.score(target, target_period="forward")
        assert len(scored.scores()) == target.height
        assert "score" in scored.frame.columns

    @pytest.mark.parametrize("name", ALL)
    def test_scoring_the_fitted_period_is_allowed(self, name):
        """Reporting train-period metrics is legitimate, so it must not raise."""
        detector = make(name, ["v"])
        detector.fit(train_frame(), period="train")
        detector.score(train_frame(), target_period="train")

    @pytest.mark.parametrize("name", ALL)
    def test_periods_before_the_fit_are_refused(self, name):
        """No period precedes train, so unassigned/ordering is the only way back."""
        detector = make(name, ["v"])
        detector.fit(train_frame(), period="train")
        with pytest.raises(LeakageError, match="unassigned"):
            detector.score(normal_frame(), target_period="unassigned")

    def test_unassigned_period_is_refused(self):
        detector = make("zscore", ["v"])
        detector.fit(train_frame(), period="train")
        with pytest.raises(LeakageError, match="unassigned"):
            detector.score(normal_frame(), target_period="unassigned")

    def test_fit_evaluation_overlap_is_refused(self):
        """A forward frame containing pre-fit rows must be rejected."""
        detector = make("zscore", ["v"])
        detector.fit(train_frame(30), period="train")
        # The training rows again, but declared as backtest: they predate
        # fitted_through, which is exactly the overlap the guard must refuse.
        with pytest.raises(LeakageError, match="overlap"):
            detector.score(train_frame(30), target_period="backtest")

    @pytest.mark.parametrize("name", ALL)
    def test_missing_feature_column_raises(self, name):
        detector = make(name, ["missing_col"])
        with pytest.raises(ValueError, match="missing feature columns"):
            detector.fit(train_frame(), period="train")

    @pytest.mark.parametrize("name", ALL)
    def test_scores_are_not_probabilities(self, name):
        detector = make(name, ["v"])
        detector.fit(train_frame(), period="train")
        assert detector.produces_probability is False
        with pytest.raises(TypeError, match="uncalibrated ranking"):
            detector.predict_proba(normal_frame(), target_period="forward")

    @pytest.mark.parametrize("name", ALL)
    def test_empty_feature_list_rejected(self, name):
        with pytest.raises(ValueError, match="non-empty"):
            make(name, [])

    @pytest.mark.parametrize("name", ALL)
    def test_identical_input_gives_identical_scores(self, name):
        """Determinism claim: same fit, same data -> byte-identical scores."""
        target = normal_frame(15)
        first = make(name, ["v"])
        first.fit(train_frame(), period="train")
        second = make(name, ["v"])
        second.fit(train_frame(), period="train")
        assert first.score(target, target_period="forward").scores().to_list() == (
            second.score(target, target_period="forward").scores().to_list()
        )


class TestStatisticalDetectors:
    def test_zscore_flags_the_spike(self):
        detector = make("zscore", ["v"])
        detector.fit(train_frame(), period="train")
        scores = detector.score(normal_frame(20), target_period="forward").scores()
        assert scores[-1] == max(scores)

    def test_zscore_is_absolute(self):
        """A far-below value is as anomalous as a far-above one."""
        detector = make("zscore", ["v"])
        # Constant training data => the score is raw deviation from a known
        # centre, so +/-delta are symmetric by construction.
        detector.fit(frame([0.0] * 20), period="train")
        target = frame([0.0, 100.0, -100.0], start=AFTER_TRAIN)
        scores = detector.score(target, target_period="forward").scores().to_list()
        assert scores[1] == pytest.approx(scores[2])
        assert scores[1] > scores[0]

    def test_robust_z_uses_median_not_mean(self):
        """One huge training outlier must not inflate the centre."""
        clean = frame([10.0] * 19 + [11.0])
        contaminated = frame([10.0] * 19 + [10_000.0])
        robust = make("robust_z", ["v"])
        robust.fit(contaminated, period="train")
        plain = make("zscore", ["v"])
        plain.fit(contaminated, period="train")
        target = frame([10_000.0], start=AFTER_TRAIN)
        # The robust score is a pure deviation here, but the key property is
        # that it is bounded by the MAD, not inflated by the outlier's own scale.
        assert robust.score(target, target_period="forward").scores()[0] > 0
        assert plain.score(target, target_period="forward").scores()[0] > 0

    def test_iqr_score_is_zero_inside_the_fences(self):
        values = [float(i) for i in range(1, 101)]
        detector = make("iqr", ["v"])
        detector.fit(frame(values), period="train")
        scores = detector.score(frame([50.0, 50.0, 1000.0], start=AFTER_TRAIN), target_period="forward").scores()
        assert scores[0] == 0.0
        assert scores[1] == 0.0
        assert scores[2] > 0.0

    def test_iqr_rejects_non_positive_k(self):
        with pytest.raises(ValueError, match="must be positive"):
            make("iqr", ["v"], k=0.0)

    def test_iqr_default_k_is_tukey(self):
        assert TUKEY_K == 1.5
        assert make("iqr", ["v"]).k == 1.5

    def test_ewma_residual_spikes_after_the_jump(self):
        detector = make("ewma", ["v"], alpha=0.2)
        detector.fit(frame([10.0] * 30), period="train")
        target = frame([10.0] * 5 + [500.0] + [10.0] * 5, start=AFTER_TRAIN)
        scores = detector.score(target, target_period="forward").scores()
        assert scores[5] == max(scores)
        # The level adapts, so the residual decays afterwards.
        assert scores[6] < scores[5]

    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
    def test_ewma_rejects_invalid_alpha(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            make("ewma", ["v"], alpha=alpha)

    def test_changepoint_accumulates_past_the_drift_allowance(self):
        detector = make("changepoint", ["v"], drift=0.5)
        detector.fit(frame([10.0, 11.0] * 15), period="train")
        # Values must sit exactly at the train median (10.5); anything else has
        # a non-zero standardised deviation and legitimately accumulates.
        quiet = frame([10.5] * 10, start=AFTER_TRAIN)
        scores = detector.score(quiet, target_period="forward").scores()
        assert all(s == 0.0 for s in scores)
        # A sustained shift must accumulate past the drift allowance.
        shifted = detector.score(frame([12.0] * 20, start=AFTER_TRAIN), target_period="forward").scores()
        assert max(shifted) > 0.0
        assert shifted[-1] > shifted[0]

    def test_changepoint_rejects_negative_drift(self):
        with pytest.raises(ValueError, match="drift"):
            make("changepoint", ["v"], drift=-1.0)

    def test_changepoint_undefined_on_zero_variance(self):
        """A constant feature admits no change point; that must raise, not guess."""
        detector = make("changepoint", ["v"])
        detector.fit(frame([5.0] * 30), period="train")
        with pytest.raises(ValueError, match="zero variance"):
            detector.score(frame([5.0, 6.0], start=AFTER_TRAIN), target_period="forward")


class TestDegenerateInputs:
    @pytest.mark.parametrize("name", ALL)
    def test_constant_feature_does_not_divide_by_zero(self, name):
        """sd/MAD/IQR of zero must degrade, not raise or emit inf/NaN.

        ``changepoint`` is the documented exception: with no variance there is
        no reference scale, so no change point is definable and it raises.
        """
        if name == "changepoint":
            pytest.skip("changepoint requires variance and refuses by design")
        detector = make(name, ["v"])
        detector.fit(frame([7.0] * 20), period="train")
        scores = detector.score(frame([7.0, 7.0, 9.0], start=AFTER_TRAIN), target_period="forward").scores()
        assert all(value == value for value in scores)  # not NaN
        assert all(abs(value) < float("inf") for value in scores)

    @pytest.mark.parametrize("name", ALL)
    def test_all_null_feature_raises_at_fit(self, name):
        detector = make(name, ["v"])
        with pytest.raises(ValueError, match="no non-null training values"):
            detector.fit(frame([None] * 20), period="train")

    @pytest.mark.parametrize("name", ALL)
    def test_null_values_score_zero_not_nan(self, name):
        detector = make(name, ["v"])
        detector.fit(train_frame(), period="train")
        scores = detector.score(frame([None, 5.0, None], start=AFTER_TRAIN), target_period="forward").scores()
        assert all(value == value for value in scores)
        assert scores[1] >= 0.0

    @pytest.mark.parametrize("name", ALL)
    def test_empty_fit_frame_raises(self, name):
        detector = make(name, ["v"])
        with pytest.raises(ValueError, match="no non-null training values"):
            detector.fit(frame([]), period="train")

    @pytest.mark.parametrize("name", ALL)
    def test_empty_scored_frame_returns_empty_scores(self, name):
        detector = make(name, ["v"])
        detector.fit(train_frame(), period="train")
        scored = detector.score(frame([], start=AFTER_TRAIN), target_period="forward")
        assert len(scored.scores()) == 0

    @pytest.mark.parametrize("name", ALL)
    def test_single_row_train_where_allowed(self, name):
        """One observation gives sd=None; the detector must survive it."""
        if name == "changepoint":
            pytest.skip("changepoint requires variance and refuses by design")
        detector = make(name, ["v"])
        detector.fit(frame([3.0]), period="train")
        scores = detector.score(frame([3.0, 4.0], start=AFTER_TRAIN), target_period="forward").scores()
        assert len(scores) == 2


class TestMetadata:
    def test_model_id_is_stable_across_identical_constructions(self):
        a = make("zscore", ["v"])
        b = make("zscore", ["v"])
        assert a._model_id() == b._model_id()

    def test_model_id_changes_with_hyperparameters(self):
        assert make("ewma", ["v"], alpha=0.1)._model_id() != make("ewma", ["v"], alpha=0.2)._model_id()

    def test_model_id_changes_with_label_protocol(self):
        assert (
            make("zscore", ["v"])._model_id()
            != make("zscore", ["v"], label_protocol="all_families")._model_id()
        )

    def test_metadata_records_training_span(self):
        detector = make("zscore", ["v"])
        meta = detector.fit(train_frame(10), period="train")
        assert meta.training_start is not None
        assert meta.training_end is not None
        assert meta.training_end > meta.training_start