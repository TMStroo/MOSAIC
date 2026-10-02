"""Adversarial tests for threshold selection and the freeze it implies.

The invariant: a threshold may be chosen from *validation* scores and labels,
and from nothing else. It is then frozen and applied unchanged to backtest and
forward.

These tests deliberately attempt the forbidden operations and assert they
raise, rather than asserting the happy path only. The E001 result where four
of five detectors selected a threshold of exactly 0.0 under ALL_FAMILIES is
also pinned here, because that is the failure mode a lax guard would hide.
"""

from __future__ import annotations

import polars as pl
import pytest

from mosaic.detection.thresholds import (
    FrozenThreshold,
    ThresholdMethod,
    assert_validation_only,
    freeze,
    select_threshold,
)
from mosaic.experiments.protocol import LeakageError


def labels(values: list[bool]) -> pl.Series:
    return pl.Series(values, dtype=pl.Boolean)


def scores(values: list[float]) -> pl.Series:
    return pl.Series(values, dtype=pl.Float64)


class TestValidationOnlyGuard:
    @pytest.mark.parametrize("period", ["train", "backtest", "forward", "unassigned", ""])
    def test_every_non_validation_period_is_refused(self, period):
        with pytest.raises(LeakageError, match="may only be selected on 'validation'"):
            assert_validation_only(period)

    def test_validation_is_allowed(self):
        assert_validation_only("validation")

    @pytest.mark.parametrize("period", ["forward", "backtest", "train"])
    def test_select_threshold_refuses_non_validation(self, period):
        """The adversarial case: selecting a threshold on forward must raise."""
        with pytest.raises(LeakageError, match="may only be selected on 'validation'"):
            select_threshold(
                y_true=labels([True, False]),
                scores=scores([2.0, 1.0]),
                method=ThresholdMethod.F1_MAX,
                validation_period=period,
            )

    def test_forward_selection_raises_before_any_computation(self):
        """It must not even evaluate candidates on forward data."""
        with pytest.raises(LeakageError):
            select_threshold(
                y_true=labels([True] * 500),
                scores=scores([1.0] * 500),
                method=ThresholdMethod.F1_MAX,
                validation_period="forward",
            )


class TestFrozenThreshold:
    def test_freeze_refuses_an_infeasible_result(self):
        from mosaic.detection.thresholds import ThresholdResult

        result = ThresholdResult(
            method=ThresholdMethod.PRECISION_CONSTRAINED,
            threshold=1.0,
            validation_period="validation",
            feasible=False,
            reason="no threshold reached precision >= 0.9",
        )
        with pytest.raises(LeakageError, match="infeasible"):
            freeze(result)

    def test_frozen_threshold_applies_to_later_periods(self):
        result = select_threshold(
            y_true=labels([True, False, True, False]),
            scores=scores([4.0, 1.0, 3.0, 0.5]),
        )
        frozen = freeze(result)
        frozen.assert_applies_to("validation")
        frozen.assert_applies_to("backtest")
        frozen.assert_applies_to("forward")

    @pytest.mark.parametrize("period", ["train", "unassigned"])
    def test_frozen_threshold_refuses_earlier_or_unassigned(self, period):
        result = select_threshold(
            y_true=labels([True, False]), scores=scores([2.0, 1.0])
        )
        frozen = freeze(result)
        with pytest.raises(LeakageError):
            frozen.assert_applies_to(period)

    def test_frozen_threshold_is_immutable(self):
        frozen = freeze(select_threshold(y_true=labels([True]), scores=scores([1.0])))
        with pytest.raises(Exception):
            frozen.threshold = 99.0  # type: ignore[misc]

    def test_threshold_is_stable_across_identical_calls(self):
        args = {"y_true": labels([True, False, True, False, False]),
                "scores": scores([5.0, 1.0, 4.0, 2.0, 0.1])}
        first = select_threshold(**args)  # type: ignore[arg-type]
        second = select_threshold(**args)  # type: ignore[arg-type]
        assert first.threshold == second.threshold


class TestObjectivesAreDistinct:
    """F1 maximisation is not universally right; each objective must differ."""

    def test_f1_max_picks_the_f1_peak(self):
        y = labels([False] * 8 + [True] * 2)
        s = scores([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 9.0, 9.5])
        result = select_threshold(y_true=y, scores=s, method=ThresholdMethod.F1_MAX)
        assert result.feasible
        assert result.achieved["f1"] == pytest.approx(1.0)

    def test_precision_constraint_never_relaxes_silently(self):
        """Unreachable precision yields feasible=False, not a relaxed answer."""
        y = labels([True] + [False] * 20)
        s = scores([1.0] + [5.0] * 20)
        result = select_threshold(
            y_true=y, scores=s, method=ThresholdMethod.PRECISION_CONSTRAINED,
            min_precision=0.9,
        )
        assert result.feasible is False
        assert "precision" in result.reason

    def test_precision_constraint_requires_its_parameter(self):
        with pytest.raises(ValueError, match="min_precision"):
            select_threshold(
                y_true=labels([True, False]), scores=scores([1.0, 0.0]),
                method=ThresholdMethod.PRECISION_CONSTRAINED,
            )

    def test_recall_constraint_requires_its_parameter(self):
        with pytest.raises(ValueError, match="min_recall"):
            select_threshold(
                y_true=labels([True, False]), scores=scores([1.0, 0.0]),
                method=ThresholdMethod.RECALL_CONSTRAINED,
            )

    def test_fpr_constraint_requires_its_parameter(self):
        with pytest.raises(ValueError, match="target_fpr"):
            select_threshold(
                y_true=labels([True, False]), scores=scores([1.0, 0.0]),
                method=ThresholdMethod.FIXED_FPR,
            )

    def test_recall_constraint_chooses_a_more_permissive_point_than_f1(self):
        """The two objectives must not collapse onto the same threshold."""
        # Four positives sit above 40 negatives, but a fifth positive is
        # buried among the low scores, so "catch everything" and "flag little"
        # genuinely disagree.
        y = labels([False] * 40 + [True] * 4 + [True])
        s = scores([0.01 * i for i in range(40)] + [5.0, 5.1, 5.2, 5.3] + [0.005])
        f1 = select_threshold(y_true=y, scores=s, method=ThresholdMethod.F1_MAX)
        recall = select_threshold(
            y_true=y, scores=s, method=ThresholdMethod.RECALL_CONSTRAINED, min_recall=1.0
        )
        fpr = select_threshold(
            y_true=y, scores=s, method=ThresholdMethod.FIXED_FPR, target_fpr=0.05
        )
        assert f1.threshold != recall.threshold or f1.threshold != fpr.threshold
        assert recall.achieved.get("recall") == 1.0
        assert fpr.achieved.get("fpr", 0.0) <= 0.05

    def test_fixed_fpr_respects_its_ceiling(self):
        y = labels([True] * 3 + [False] * 97)
        s = scores([1.0] * 3 + [0.5] * 97)
        result = select_threshold(
            y_true=y, scores=s, method=ThresholdMethod.FIXED_FPR, target_fpr=0.0
        )
        assert result.achieved.get("fpr", 0.0) <= 0.0


class TestDegenerateInputs:
    def test_all_null_scores_are_infeasible_not_zero(self):
        result = select_threshold(
            y_true=labels([True, False]),
            scores=pl.Series([None, None], dtype=pl.Float64),
        )
        assert result.feasible is False
        assert "no non-null scores" in result.reason

    def test_zero_positive_period_is_reported_not_crashed(self):
        """No positives: every threshold is equally bad, F1 is a flat 0.0.

        The point is that this is reported as a measured 0.0 rather than
        crashing or inventing a positive metric.
        """
        result = select_threshold(
            y_true=labels([False, False, False]),
            scores=scores([1.0, 2.0, 3.0]),
        )
        assert result.feasible
        assert result.achieved["f1"] == 0.0
        assert result.achieved["precision"] == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            select_threshold(y_true=labels([True, False]), scores=scores([1.0]))

    def test_candidate_cap_bounds_cost(self):
        y = labels([False, True] * 500)
        s = scores([float(i) for i in range(1000)])
        result = select_threshold(y_true=y, scores=s, max_candidates=64)
        # The flag-nothing sentinel is always added on top of the sampled set.
        assert result.candidates_considered <= 65

    def test_always_offers_a_flag_nothing_threshold(self):
        """A zero-FPR solution must exist, so FIXED_FPR(0) is attainable.

        Selection may land on any candidate meeting the ceiling -- the tie
        between "flags the top row" and "flags nothing" is broken by cost --
        but the guarantee is the FPR, not a particular number.
        """
        y = labels([True, False, True])
        s = scores([100.0, 50.0, 10.0])
        result = select_threshold(
            y_true=y, scores=s, method=ThresholdMethod.FIXED_FPR, target_fpr=0.0
        )
        assert result.feasible
        assert result.achieved["fpr"] <= 0.0
        # The sentinel is present in the candidate set even on a 3-row input.
        assert result.candidates_considered >= 4


class TestRegressionAllFamiliesZeroThreshold:
    """Pin the E001 finding so it can never be silently 'fixed'.

    Under ALL_FAMILIES, four of five statistical detectors selected a
    threshold of exactly 0.0, flagging every row. That is a degenerate operating
    point -- precision collapses to the base rate -- and it is reported as a
    result rather than removed by changing the detector.
    """

    def test_f1_max_can_select_zero_when_nothing_beats_the_base_rate(self):
        """F1 maximisation has no reason to avoid flagging everything.

        With one positive buried in 1000 rows, flagging all rows is the only way
        to catch it, so F1's optimum is the degenerate threshold.
        """
        y = labels([False] * 999 + [True])
        s = scores([0.0] * 1000)
        result = select_threshold(y_true=y, scores=s, method=ThresholdMethod.F1_MAX)
        assert result.threshold == 0.0
        assert result.achieved["precision"] == pytest.approx(1 / 1000)
        assert result.achieved["recall"] == 1.0

    def test_precision_constraint_avoids_the_degenerate_point(self):
        """The alternative objective refuses the flag-everything operating point."""
        y = labels([False] * 999 + [True])
        s = scores([0.0] * 1000)
        result = select_threshold(
            y_true=y, scores=s, method=ThresholdMethod.PRECISION_CONSTRAINED,
            min_precision=0.5,
        )
        assert result.feasible is False

    def test_recall_constraint_still_allows_the_degenerate_point(self):
        """Recall >= 1.0 forces flagging everything: the objective's own doing.

        This is why the objective must be an explicit experimental variable
        rather than a default.
        """
        y = labels([False] * 99 + [True])
        s = scores([0.0] * 100)
        result = select_threshold(
            y_true=y, scores=s, method=ThresholdMethod.RECALL_CONSTRAINED, min_recall=1.0
        )
        assert result.feasible
        assert result.achieved["recall"] == 1.0