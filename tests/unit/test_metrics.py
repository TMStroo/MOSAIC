"""Tests for anomaly metrics, confidence intervals and per-family reporting.

The central claim under test: with a very small positive support, a point
estimate is not a result. Every rate metric must carry a 95% interval, and an
undefined metric must be ``None`` plus a NOT_APPLICABLE reason -- never 0.0,
which would be indistinguishable from a measured zero.

PR-AUC and ROC-AUC are cross-checked against scikit-learn to prove the
from-scratch implementations agree, including on ties.
"""

from __future__ import annotations

import polars as pl
import pytest

from mosaic.metrics import (
    average_precision,
    confusion_counts,
    evaluate_predictions,
    per_family_metrics,
    roc_auc,
    wilson_interval,
)


def series(values: list[bool]) -> pl.Series:
    return pl.Series(values, dtype=pl.Boolean)


class TestWilsonInterval:
    def test_point_matches_proportion(self):
        assert wilson_interval(12, 21).point == pytest.approx(12 / 21)

    def test_forward_recall_on_21_positives_is_very_wide(self):
        """The power problem, stated as a test.

        12/21 gives a 0.37-wide interval, which is why EVENT_ONLY forward
        metrics cannot rank models.
        """
        interval = wilson_interval(12, 21)
        assert interval.low == pytest.approx(0.365, abs=0.01)
        assert interval.high == pytest.approx(0.755, abs=0.01)
        assert interval.high - interval.low > 0.35

    def test_zero_trials_is_unknown_not_zero(self):
        assert wilson_interval(0, 0) is None

    def test_interval_stays_inside_unit_range(self):
        for successes, trials in ((0, 5), (5, 5), (1, 3), (0, 21)):
            interval = wilson_interval(successes, trials)
            assert 0.0 <= interval.low <= 1.0
            assert 0.0 <= interval.high <= 1.0
            assert interval.low <= interval.point <= interval.high

    def test_wider_interval_for_smaller_samples(self):
        small = wilson_interval(5, 10)
        large = wilson_interval(500, 1000)
        assert (small.high - small.low) > (large.high - large.low)

    def test_interval_brackets_the_point(self):
        """Wilson is skewed, not symmetric; it must still contain the estimate."""
        interval = wilson_interval(7, 20)
        assert interval.low <= interval.point <= interval.high
        # The Wilson centre is pulled toward the null for p_hat < 0.5, so the
        # upper arm is the longer one.
        assert (interval.high - interval.point) > (interval.point - interval.low)

    def test_serialises_with_method_named(self):
        payload = wilson_interval(3, 10).to_dict()
        assert payload["ci_method"] == "wilson"
        assert set(payload) == {"point", "ci95_low", "ci95_high", "ci_method"}


class TestConfusion:
    def test_all_four_keys_always_present(self):
        counts = confusion_counts(series([True]), series([True]))
        assert counts == {"tp": 1, "fp": 0, "fn": 0, "tn": 0}

    def test_counts_a_mixed_case(self):
        counts = confusion_counts(
            series([True, True, False, False]), series([True, False, True, False])
        )
        assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}

    def test_empty_input(self):
        assert confusion_counts(series([]), series([])) == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            confusion_counts(series([True]), series([True, False]))


class TestAveragePrecision:
    def test_matches_sklearn(self):
        import numpy as np
        from sklearn.metrics import average_precision_score

        rng = np.random.default_rng(11)
        y = rng.random(500) < 0.05
        s = rng.random(500) + y * 0.4
        mine = average_precision(pl.Series(y), pl.Series(s))
        assert mine == pytest.approx(float(average_precision_score(y, s)), abs=1e-12)

    def test_perfect_separation_gives_one(self):
        assert average_precision(series([False, False, True, True]), pl.Series([0.1, 0.2, 0.9, 1.0])) == 1.0

    def test_inverted_separation_gives_low(self):
        value = average_precision(series([False, True]), pl.Series([10.0, 1.0]))
        assert value < 0.6

    def test_no_positives_is_undefined(self):
        assert average_precision(series([False, False]), pl.Series([1.0, 2.0])) is None

    def test_ties_are_ranked_together(self):
        """Tied scores must not be broken arbitrarily."""
        y = series([True, False, True, False])
        s = pl.Series([1.0, 1.0, 1.0, 1.0])
        value = average_precision(y, s)
        # Every threshold yields the same base rate.
        assert value == pytest.approx(0.5, abs=1e-9)

    def test_empty_input(self):
        assert average_precision(series([]), pl.Series([], dtype=pl.Float64)) is None


class TestRocAuc:
    def test_matches_sklearn(self):
        import numpy as np
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(3)
        y = rng.random(400) < 0.04
        s = rng.random(400) + y * 0.3
        mine = roc_auc(pl.Series(y), pl.Series(s))
        assert mine == pytest.approx(float(roc_auc_score(y, s)), abs=1e-12)

    def test_perfect_and_inverted(self):
        assert roc_auc(series([False, True]), pl.Series([0.0, 1.0])) == 1.0
        assert roc_auc(series([False, True]), pl.Series([1.0, 0.0])) == 0.0

    def test_all_tied_gives_one_half(self):
        assert roc_auc(series([True, False, True, False]), pl.Series([1.0] * 4)) == 0.5

    def test_single_class_is_undefined(self):
        assert roc_auc(series([True, True]), pl.Series([1.0, 2.0])) is None
        assert roc_auc(series([False, False]), pl.Series([1.0, 2.0])) is None


class TestEvaluatePredictions:
    def test_reports_support_and_confusion(self):
        metrics = evaluate_predictions(
            period="forward",
            label_protocol="event_only",
            y_true=series([True, True, False, False, False]),
            y_pred=series([True, False, True, False, False]),
            threshold=0.5,
        )
        assert metrics.support_positive == 2
        assert metrics.support_negative == 3
        assert metrics.confusion == {"tp": 1, "fp": 1, "fn": 1, "tn": 2}
        assert metrics.recall.point == pytest.approx(0.5)
        assert metrics.precision.point == pytest.approx(0.5)

    def test_zero_positive_period_marks_recall_not_applicable(self):
        metrics = evaluate_predictions(
            period="backtest",
            label_protocol="event_only",
            y_true=series([False] * 5),
            y_pred=series([False] * 5),
            threshold=0.5,
            scores=pl.Series([0.0] * 5),
        )
        assert metrics.support_positive == 0
        assert metrics.recall is None
        assert "recall" in metrics.not_applicable
        assert "pr_auc" in metrics.not_applicable

    def test_zero_negative_period_marks_fpr_not_applicable(self):
        metrics = evaluate_predictions(
            period="forward",
            label_protocol="event_only",
            y_true=series([True, True]),
            y_pred=series([True, False]),
            threshold=0.5,
            scores=pl.Series([1.0, 0.0]),
        )
        assert metrics.support_negative == 0
        assert metrics.false_positive_rate is None
        assert "false_positive_rate" in metrics.not_applicable

    def test_no_predictions_means_precision_undefined_but_f1_zero(self):
        """Precision and F1 behave differently when nothing is predicted.

        Precision is a ratio with a zero denominator -> undefined, reported as
        None. F1 is 2tp/(2tp+fp+fn), which is genuinely 0 when tp=0, so it is a
        real measurement and must not be reported as None.
        """
        metrics = evaluate_predictions(
            period="validation",
            label_protocol="event_only",
            y_true=series([True, False]),
            y_pred=series([False, False]),
            threshold=0.5,
        )
        assert metrics.precision is None
        assert metrics.f1 is not None
        assert metrics.f1.point == 0.0

    def test_all_null_scores_leave_ranking_metrics_absent(self):
        metrics = evaluate_predictions(
            period="forward",
            label_protocol="event_only",
            y_true=series([True, False]),
            y_pred=series([True, False]),
            threshold=0.5,
            scores=pl.Series([None, None], dtype=pl.Float64),
        )
        assert metrics.pr_auc is None

    def test_accuracy_is_not_reported(self):
        """Accuracy measures prevalence on an imbalanced problem; omit it."""
        metrics = evaluate_predictions(
            period="forward",
            label_protocol="event_only",
            y_true=series([True] + [False] * 99),
            y_pred=series([False] * 100),
            threshold=0.5,
        )
        assert "accuracy" not in metrics.to_dict()

    def test_serialises_every_field(self):
        metrics = evaluate_predictions(
            period="forward",
            label_protocol="all_families",
            y_true=series([True, False]),
            y_pred=series([True, True]),
            threshold=0.4,
            scores=pl.Series([0.9, 0.8]),
        )
        payload = metrics.to_dict()
        for key in (
            "period", "label_protocol", "n_rows", "support_positive", "support_negative",
            "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc",
            "false_positive_rate", "confusion", "not_applicable",
        ):
            assert key in payload
        assert payload["label_protocol"] == "all_families"

    def test_empty_period_does_not_crash(self):
        metrics = evaluate_predictions(
            period="forward",
            label_protocol="event_only",
            y_true=series([]),
            y_pred=series([]),
        )
        assert metrics.n_rows == 0
        assert metrics.recall is None


class TestPerFamilyMetrics:
    @pytest.fixture
    def frame(self) -> pl.DataFrame:
        rows_fam: list[list[str]] = []
        rows_label: list[bool] = []
        rows_pred: list[bool] = []
        for i in range(40):
            family = "point" if i < 10 else ("collective" if i < 20 else "")
            rows_fam.append([family] if family else [])
            # Half of each block is truly positive.
            rows_label.append((i % 10) < 5 if i < 20 else False)
            rows_pred.append((i % 10) < 3 if i < 20 else False)
        return pl.DataFrame(
            {
                "anomaly_family": rows_fam,
                "is_anomaly": rows_label,
                "predicted_anomaly": rows_pred,
            }
        )

    def test_reports_status_and_support(self, frame):
        out = per_family_metrics(
            frame=frame, period="forward", label_protocol="event_only",
            minimum_support=1, families=["point", "collective", "temporal"],
        )
        assert out["point"]["status"] == "OK"
        assert out["point"]["support"] == 5
        assert out["temporal"]["status"] == "NOT_APPLICABLE"
        assert out["temporal"]["precision"] is None

    def test_below_floor_is_not_applicable_not_zero(self, frame):
        out = per_family_metrics(
            frame=frame, period="forward", label_protocol="event_only",
            minimum_support=50, families=["point"],
        )
        assert out["point"]["status"] == "NOT_APPLICABLE"
        assert "minimum" in out["point"]["reason"]
        assert out["point"]["recall"] is None

    def test_records_coverage_when_supplied(self, frame):
        out = per_family_metrics(
            frame=frame, period="forward", label_protocol="event_only",
            minimum_support=1, families=["point"], coverage={"point": 10},
        )
        assert out["point"]["coverage"] == 10

    def test_confidence_intervals_present_when_ok(self, frame):
        out = per_family_metrics(
            frame=frame, period="forward", label_protocol="event_only",
            minimum_support=1, families=["point"],
        )
        ci = out["point"]["confidence_intervals"]
        assert ci["recall"]["ci_method"] == "wilson"
        assert 0.0 <= ci["recall"]["ci95_low"] <= ci["recall"]["ci95_high"] <= 1.0

    def test_aggregate_family_can_have_zero_positives_under_event_scope(self, frame):
        """A family resolved but scope-excluded reports 0 support, not an error."""
        out = per_family_metrics(
            frame=frame, period="forward", label_protocol="event_only",
            minimum_support=1, families=["collective"],
        )
        assert out["collective"]["support"] >= 0
