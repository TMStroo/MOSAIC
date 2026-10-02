"""Tests for the ML baselines and the model contract they satisfy.

Three groups:

* **Lifecycle** -- the same guards the statistical detectors have: fit only on
  train, never score before fit, never apply a later threshold backwards.
* **Score semantics** -- higher means more anomalous everywhere; only genuine
  classifiers expose ``predict_proba``; unsupervised models never do.
* **Leakage boundaries** -- adversarial cases, asserted to raise.

The ``DummyClassifier`` tests pin the floor of the supervised comparison. It
must predict the majority class, so on any class-imbalanced frame its scores are
identically zero. If that ever stops being true, "learned nothing" no longer has
a reference point and every other number in E011 loses its meaning.
"""

from __future__ import annotations

import datetime
from typing import ClassVar

import numpy as np
import polars as pl
import pytest

from mosaic.detection import (
    ML_MODELS,
    SUPERVISED_MODELS,
    UNSUPERVISED_MODELS,
    Detector,
    model_specs,
)
from mosaic.detection.ml import _rank_scale
from mosaic.experiments.protocol import LeakageError

T0 = datetime.datetime(2021, 1, 1)
AFTER_TRAIN = datetime.datetime(2021, 3, 1)

ALL = sorted(ML_MODELS)
SUPERVISED = sorted(SUPERVISED_MODELS)
UNSUPERVISED = sorted(UNSUPERVISED_MODELS)

BASE = {
    "label_protocol": "event_only",
    "feature_set_id": "fs_test",
    "dataset_version": "test:0",
    "feature_set_version": "v1",
    "random_seed": 0,
}


def frame(values: list[float | None], *, start: datetime.datetime = T0) -> pl.DataFrame:
    times = [start + datetime.timedelta(hours=i) for i in range(len(values))]
    return pl.DataFrame(
        {
            "timestamp": times,
            "is_anomaly": [i % 20 == 0 for i in range(len(values))],
            "f_a": values,
            "f_b": [(v or 0.0) * 2.0 + 1.0 for v in values],
        }
    )


def varied_target(n: int = 40) -> pl.DataFrame:
    """A scored frame with spread, *within* the training feature range.

    Two properties are needed for a seed-sensitivity assertion to mean anything:

    * spread, or every row scores identically and no seed can change the result;
    * in-range values, because Isolation Forest gives any point far outside the
      training distribution a path length of 1 -- it is isolated immediately --
      so a wildly out-of-range target saturates to one score for every seed. That
      is a real property of the algorithm, not a seed effect.
    """
    times = [AFTER_TRAIN + datetime.timedelta(hours=i) for i in range(n)]
    values = 10.0 + 2.0 * np.random.default_rng(3).normal(size=n)
    return pl.DataFrame(
        {
            "timestamp": times,
            "is_anomaly": [i % 20 == 0 for i in range(n)],
            "f_a": values,
            "f_b": [v * 2.0 + 1.0 for v in values],
        }
    )


def make(name: str, **kwargs) -> Detector:
    params = {**BASE, "feature_columns": ["f_a", "f_b"], **kwargs}
    return ML_MODELS[name](**params)


class TestRegistry:
    def test_seven_models_in_two_families(self):
        assert len(SUPERVISED) == 5
        assert len(UNSUPERVISED) == 2
        assert set(SUPERVISED) | set(UNSUPERVISED) == set(ML_MODELS)

    def test_specs_declare_supervision_and_probability(self):
        specs = {s["name"]: s for s in model_specs()}
        assert specs["dummy"]["supervised"] is True
        assert specs["isolation_forest"]["supervised"] is False
        assert specs["isolation_forest"]["produces_probability"] is False
        assert specs["random_forest"]["produces_probability"] is True

    def test_trees_do_not_standardise_but_distance_models_do(self):
        specs = {s["name"]: s for s in model_specs()}
        assert specs["random_forest"]["standardize"] is False
        assert specs["lof"]["standardize"] is True

    def test_stochastic_models_are_not_marked_deterministic(self):
        specs = {s["name"]: s for s in model_specs()}
        for name in ("logistic", "random_forest", "gradient_boosting", "isolation_forest", "lof"):
            assert specs[name]["deterministic"] is False
        assert specs["dummy"]["deterministic"] is True
        assert specs["decision_tree"]["deterministic"] is True


class TestLifecycle:
    @pytest.mark.parametrize("name", ALL)
    def test_fit_then_score_preserves_row_order(self, name):
        model = make(name)
        model.fit(frame([1.0] * 200), period="train")
        target = frame([1.0] * 50, start=AFTER_TRAIN)
        scores = model.score(target, target_period="forward").scores()
        assert scores.len() == target.height
        assert scores.null_count() == 0

    @pytest.mark.parametrize("name", ALL)
    @pytest.mark.parametrize("period", ["validation", "backtest", "forward", "unassigned"])
    def test_fit_refuses_every_non_train_period(self, name, period):
        with pytest.raises(LeakageError, match="may only fit on 'train'"):
            make(name).fit(frame([1.0] * 50), period=period)

    @pytest.mark.parametrize("name", ALL)
    def test_score_before_fit_raises(self, name):
        with pytest.raises((RuntimeError, LeakageError)):
            make(name).score(frame([1.0] * 10, start=AFTER_TRAIN), target_period="forward")

    @pytest.mark.parametrize("name", ALL)
    def test_missing_feature_column_raises(self, name):
        model = make(name)
        bare = pl.DataFrame({"timestamp": [T0], "is_anomaly": [True]})
        with pytest.raises(ValueError, match="missing feature columns"):
            model.fit(bare, period="train")

    @pytest.mark.parametrize("name", ALL)
    def test_scoring_the_fitted_period_is_allowed(self, name):
        """Phase 2 rule: train-with-train-fitted is legitimate and required."""
        model = make(name)
        model.fit(frame([1.0] * 100), period="train")
        scores = model.score(frame([1.0] * 10), target_period="train").scores()
        assert scores.len() == 10

    @pytest.mark.parametrize("name", ALL)
    def test_later_period_containing_pre_fit_rows_raises(self, name):
        model = make(name)
        model.fit(frame([1.0] * 100, start=AFTER_TRAIN), period="train")
        # Rows from before the fit, presented as a later period.
        with pytest.raises(LeakageError, match="overlap"):
            model.score(frame([1.0] * 5, start=T0), target_period="forward")

    #: One hyperparameter each estimator actually accepts.
    VALID_HP: ClassVar = {
        "dummy": {"strategy": "most_frequent"},
        "logistic": {"max_iter": 50},
        "decision_tree": {"max_depth": 4},
        "random_forest": {"n_estimators": 7},
        "gradient_boosting": {"n_estimators": 5},
        "isolation_forest": {"n_estimators": 7},
        "lof": {"n_neighbors": 5},
    }

    @pytest.mark.parametrize("name", ALL)
    def test_metadata_records_the_required_fields(self, name):
        hyper = self.VALID_HP[name]
        model = make(name, hyperparameters=hyper)
        meta = model.fit(frame([1.0] * 120), period="train")
        payload = meta.to_dict()
        for key in (
            "model_id", "model_type", "feature_set_version", "dataset_version",
            "training_period", "random_seed", "hyperparameters", "code_version",
            "produces_probability", "deterministic", "fitted_features",
        ):
            assert key in payload, key
        assert payload["training_period"] == "train"
        assert payload["hyperparameters"] == hyper
        assert payload["code_version"] != "unknown"
        assert list(payload["fitted_features"]) == ["f_a", "f_b"]

    @pytest.mark.parametrize("name", ALL)
    def test_model_id_changes_with_seed(self, name):
        a = make(name, random_seed=0).fit(frame([1.0] * 60), period="train").model_id
        b = make(name, random_seed=1).fit(frame([1.0] * 60), period="train").model_id
        assert a != b

    def test_empty_feature_columns_refused(self):
        with pytest.raises(ValueError, match="non-empty"):
            ML_MODELS["dummy"](**{**BASE, "feature_columns": []})


class TestScoreSemantics:
    @staticmethod
    def _separable_frame(n: int, start: datetime.datetime = T0) -> pl.DataFrame:
        """A frame where high feature values genuinely mark the anomalies.

        Written out rather than reusing ``frame`` because the ordering claim
        needs the label and the feature to agree; a helper that derives labels
        from row position would make that agreement accidental.
        """
        times = [start + datetime.timedelta(hours=i) for i in range(n)]
        labels = [i % 20 == 0 for i in range(n)]
        values = [900.0 if label else 1.0 for label in labels]
        return pl.DataFrame(
            {
                "timestamp": times,
                "is_anomaly": labels,
                "f_a": values,
                "f_b": [v * 2.0 + 1.0 for v in values],
            }
        )

    @pytest.mark.parametrize("name", [n for n in ALL if n != "dummy"])
    def test_higher_score_means_more_anomalous(self, name):
        """A clearly extreme row must outrank a clearly ordinary one.

        ``dummy`` is excluded because it is flat by construction -- it cannot
        express the ordering at all, which is the whole point of including it.
        """
        model = make(name)
        model.fit(self._separable_frame(200), period="train")
        target = self._separable_frame(20, start=AFTER_TRAIN)
        scores = model.score(target, target_period="forward").scores()
        # Row 0 is positive in this fixture, row 1 is not.
        assert scores[0] > scores[1]

    @pytest.mark.parametrize("name", ALL)
    def test_scores_are_finite_even_with_nulls(self, name):
        """A null in a later period must not become NaN in the score.

        The unsupervised models fit an imputer on train; if that imputer is not
        also applied at score time, sklearn raises on the null. This test is the
        regression for that.
        """
        model = make(name)
        model.fit(frame([1.0, 2.0] * 100), period="train")
        scores = model.score(
            frame([1.0, None, 3.0, None], start=AFTER_TRAIN), target_period="forward"
        ).scores()
        assert scores.len() == 4
        assert not any(np.isnan(v) for v in scores)

    @pytest.mark.parametrize("name", UNSUPERVISED)
    def test_unsupervised_refuses_predict_proba(self, name):
        model = make(name)
        model.fit(frame([1.0] * 100), period="train")
        with pytest.raises(TypeError, match="does not produce probabilities"):
            model.predict_proba(frame([1.0] * 5, start=AFTER_TRAIN))

    @pytest.mark.parametrize("name", SUPERVISED)
    def test_supervised_predict_proba_is_a_probability(self, name):
        model = make(name)
        model.fit(frame([1.0] * 120), period="train")
        proba = model.predict_proba(
            frame([1.0] * 20, start=AFTER_TRAIN), target_period="forward"
        ).scores()
        assert float(proba.min()) >= 0.0
        assert float(proba.max()) <= 1.0

    def test_rank_scale_is_order_preserving_and_bounded(self):
        values = np.array([5.0, -1.0, 3.0, 3.0, 0.0])
        scaled = _rank_scale(values)
        assert scaled.min() > 0.0 and scaled.max() <= 1.0
        # Ties share a rank, so their order against each other is arbitrary but
        # their value is identical.
        assert scaled[2] == scaled[3]
        assert scaled[1] < scaled[2] < scaled[0]

    def test_rank_scale_handles_empty(self):
        assert _rank_scale(np.array([])).size == 0


class TestDummyIsTheFloor:
    """The dummy defines "learned nothing"; it must stay exactly there."""

    def test_dummy_scores_are_identically_zero_when_minority(self):
        model = make("dummy")
        model.fit(frame([1.0] * 200), period="train")  # 10 positives of 200
        scores = model.score(frame([1.0] * 20, start=AFTER_TRAIN), target_period="forward").scores()
        assert all(v == 0.0 for v in scores)

    def test_dummy_never_predicts_positive_on_imbalanced_data(self):
        model = make("dummy")
        model.fit(frame([1.0] * 500), period="train")
        predicted = model.score(frame([1.0] * 50, start=AFTER_TRAIN), target_period="forward")
        assert not any(predicted.predictions(0.5))

    def test_dummy_precision_equals_prevalence_or_is_undefined(self):
        """The statistical Phase 2 result, reproduced as a model baseline.

        Flagging nothing makes precision undefined; the point is that the dummy
        can never exceed the base rate, which is the floor every other model
        must clear.
        """
        model = make("dummy")
        model.fit(frame([1.0] * 200), period="train")
        scores = model.score(frame([1.0] * 100, start=AFTER_TRAIN), target_period="forward").scores()
        assert float(scores.max()) == 0.0


class TestPreprocessingIsCausal:
    @pytest.mark.parametrize("name", ALL)
    def test_imputer_is_fitted_on_train_only(self, name):
        """Fitting then scoring must not refit anything.

        Verified behaviourally: scoring the same frame twice gives identical
        results, which a refitting imputer would not guarantee once the data
        changed.
        """
        model = make(name)
        model.fit(frame([1.0] * 150), period="train")
        a = frame([1.0, None, 2.0] * 5, start=AFTER_TRAIN)
        b = frame([9.0, 8.0, 7.0] * 5, start=AFTER_TRAIN)
        first = model.score(a, target_period="forward").scores().to_list()
        model.score(b, target_period="backtest")
        again = model.score(a, target_period="forward").scores().to_list()
        assert len(first) == len(again)
        # A refitting imputer would change these materially; float-reduction
        # noise in a parallel forest does not, hence the tolerance.
        assert np.allclose(first, again, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("name", ALL)
    def test_rejects_non_numeric_feature(self, name):
        model = make(name)
        text = pl.DataFrame(
            {
                "timestamp": [T0 + datetime.timedelta(hours=i) for i in range(20)],
                "is_anomaly": [i == 0 for i in range(20)],
                "f_a": ["x"] * 20,
                "f_b": ["y"] * 20,
            }
        )
        with pytest.raises((TypeError, ValueError, pl.exceptions.PolarsError)):
            model.fit(text, period="train")


class TestSeeds:
    @staticmethod
    def _continuous_train(n: int = 200) -> pl.DataFrame:
        """Training data with genuinely varied features.

        Isolation Forest draws its randomness from subsampling rows. If the
        training frame holds only a couple of distinct feature vectors, every
        subsample is identical and the seed cannot change the fit at all -- so a
        seed-sensitivity test on such a frame asserts nothing.
        """
        rng = np.random.default_rng(7)
        times = [T0 + datetime.timedelta(hours=i) for i in range(n)]
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        labels = [i % 20 == 0 for i in range(n)]
        # Anomalies are a *shift* of the same distribution, not a single extreme
        # outlier: one 900-sigma point would put every in-distribution test row
        # in the same far leaf and make all scores coincide.
        values = [v + 6.0 if label else v for label, v in zip(labels, base, strict=True)]
        return pl.DataFrame(
            {
                "timestamp": times,
                "is_anomaly": labels,
                "f_a": values,
                "f_b": [v * 2.0 + 1.0 for v in values],
            }
        )

    @pytest.mark.parametrize("name", ["random_forest", "isolation_forest"])
    def test_same_seed_reproduces_scores(self, name):
        target = varied_target()
        a = make(name, random_seed=11)
        a.fit(self._continuous_train(), period="train")
        b = make(name, random_seed=11)
        b.fit(self._continuous_train(), period="train")
        first = a.score(target, target_period="forward").scores().to_list()
        second = b.score(target, target_period="forward").scores().to_list()
        assert np.allclose(first, second, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("name", ["random_forest", "isolation_forest"])
    def test_different_seeds_change_scores(self, name):
        target = varied_target(120)
        a = make(name, random_seed=1)
        a.fit(self._continuous_train(), period="train")
        b = make(name, random_seed=2)
        b.fit(self._continuous_train(), period="train")
        first = a.score(target, target_period="forward").scores().to_list()
        second = b.score(target, target_period="forward").scores().to_list()
        assert not np.allclose(first, second, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("name", ["dummy", "decision_tree"])
    def test_deterministic_models_ignore_the_seed(self, name):
        target = varied_target()
        a = make(name, random_seed=1)
        a.fit(frame([1.0, 2.0] * 75), period="train")
        b = make(name, random_seed=99)
        b.fit(frame([1.0, 2.0] * 75), period="train")
        assert np.allclose(
            a.score(target, target_period="forward").scores().to_list(),
            b.score(target, target_period="forward").scores().to_list(),
            rtol=1e-12, atol=1e-15,
        )
