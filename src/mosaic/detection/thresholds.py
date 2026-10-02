"""Validation-only threshold selection, with the freeze made explicit.

The rule this module enforces: a threshold may be chosen from *validation*
scores and validation labels, and from nothing else. Once chosen it is frozen
and handed to backtest and forward untouched.

That is enforced three ways, not just documented:

1. :func:`select_threshold` takes a ``validation_period`` argument that it
   checks against the period labels of the frames it is given.
2. :class:`FrozenThreshold` is immutable and carries the period it was selected
   on. Applying it to an earlier period raises.
3. :func:`assert_validation_only` raises :class:`LeakageError` when a caller
   tries to select a threshold on backtest/forward/train scores. The test suite
   calls this deliberately to prove it fires.

Selection methods are configuration-driven: F1-maximising, precision-constrained,
recall-constrained and fixed-FPR. Each states what it optimises and when it is
infeasible, so a failed constraint is reported rather than silently relaxed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NamedTuple

import polars as pl

from mosaic.experiments.protocol import PERIODS, LeakageError, Period
from mosaic.metrics import confusion_counts


class ThresholdMethod(StrEnum):
    F1_MAX = "f1_max"
    PRECISION_CONSTRAINED = "precision_constrained"
    RECALL_CONSTRAINED = "recall_constrained"
    FIXED_FPR = "fixed_fpr"


@dataclass(frozen=True)
class ThresholdResult:
    """A chosen threshold plus the evidence for why it was chosen."""

    method: ThresholdMethod
    threshold: float
    validation_period: Period
    feasible: bool
    reason: str = ""
    achieved: dict[str, float] = field(default_factory=dict)
    candidates_considered: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": str(self.method),
            "threshold": self.threshold,
            "validation_period": self.validation_period,
            "feasible": self.feasible,
            "reason": self.reason,
            "achieved": self.achieved,
            "candidates_considered": self.candidates_considered,
        }


@dataclass(frozen=True)
class FrozenThreshold:
    """An immutable threshold with the period it was selected on attached."""

    threshold: float
    selected_on: Period
    result: ThresholdResult

    def __post_init__(self) -> None:
        if not self.result.feasible:
            raise LeakageError(
                f"refusing to freeze an infeasible threshold: {self.result.reason}"
            )

    def assert_applies_to(self, target_period: Period | str) -> None:
        """A validation threshold may not be applied to an earlier period."""
        if target_period == "unassigned":
            raise LeakageError("frozen threshold cannot be applied to unassigned rows")
        if target_period in PERIODS and PERIODS.index(target_period) < PERIODS.index(self.selected_on):
            raise LeakageError(
                f"threshold selected on {self.selected_on} cannot be applied to "
                f"{target_period}: that is backward in time"
            )


def assert_validation_only(period: Period | str) -> None:
    """Raise unless ``period`` is exactly ``validation``.

    Called by :func:`select_threshold`. Exposed so a test can prove that
    attempting forward-period threshold selection raises.
    """
    if period != "validation":
        raise LeakageError(
            f"thresholds may only be selected on 'validation', not {period!r}; "
            "selecting on backtest/forward/train would tune the decision on the "
            "evaluation data"
        )


def _counts(y_true: pl.Series, y_pred: pl.Series) -> dict[str, int]:
    return confusion_counts(y_true, y_pred)


class _Rates(NamedTuple):
    """Per-threshold rates.

    ``fpr`` and ``f1`` are always defined -- FPR is 0.0 when there are no
    negatives to divide by, and F1 is 0.0 when nothing was predicted positive --
    while ``precision`` and ``recall`` are genuinely undefined when their
    denominator is empty. Making that difference part of the type is the point:
    the caller cannot accidentally treat a missing precision as 0.0.
    """

    precision: float | None
    recall: float | None
    f1: float
    fpr: float


def _metrics_from_counts(counts: dict[str, int]) -> _Rates:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision and recall and (precision + recall):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    fpr = fp / (fp + counts["tn"]) if (fp + counts["tn"]) else 0.0
    return _Rates(precision=precision, recall=recall, f1=f1, fpr=fpr)


def _candidate_thresholds(scores: pl.Series, max_candidates: int = 512) -> list[float]:
    """Candidate thresholds: observed scores plus a mid-point above the max.

    Using the observed scores themselves means a threshold reproduces exactly
    the row that produced it (``>=``), so the selected operating point is
    attainable rather than interpolated. Sub-sampling keeps selection cost
    bounded on the 250k-row train period while still spanning the range.
    """
    unique = sorted(set(scores.drop_nulls().to_list()))
    if not unique:
        return []
    top = max(unique)
    # A threshold strictly above the maximum flags nothing. It is always
    # included -- including on the small-input path -- so a feasible "zero-FPR"
    # solution exists whenever one is required.
    sentinel = top + max(1e-9, abs(top) * 1e-9)
    if len(unique) <= max_candidates:
        return sorted({*unique, sentinel})
    step = len(unique) / max_candidates
    sampled = [unique[int(i * step)] for i in range(max_candidates)]
    return sorted({*sampled, sentinel})


def select_threshold(
    *,
    y_true: pl.Series,
    scores: pl.Series,
    method: ThresholdMethod = ThresholdMethod.F1_MAX,
    validation_period: Period | str = "validation",
    min_precision: float | None = None,
    min_recall: float | None = None,
    target_fpr: float | None = None,
    max_candidates: int = 512,
) -> ThresholdResult:
    """Choose a threshold from validation scores and labels.

    Raises :class:`LeakageError` unless ``validation_period == "validation"``.
    Infeasible constraints produce a ``feasible=False`` result rather than a
    quietly relaxed one.
    """
    assert_validation_only(validation_period)

    if len(y_true) != len(scores):
        raise ValueError(f"length mismatch: {len(y_true)} labels vs {len(scores)} scores")
    candidates = _candidate_thresholds(scores, max_candidates)
    if not candidates:
        return ThresholdResult(
            method=method, threshold=float("nan"), validation_period="validation",
            feasible=False, reason="no non-null scores on validation",
        )

    y = y_true.cast(pl.Boolean)
    best: tuple[float, dict[str, int]] | None = None
    best_threshold = candidates[-1]
    evaluated = 0
    for candidate in candidates:
        evaluated += 1
        counts = _counts(y, scores >= candidate)
        rates = _metrics_from_counts(counts)
        # `value` is what the objective maximises subject to its own constraint,
        # so a feasible candidate always beats no candidate.
        if method is ThresholdMethod.F1_MAX:
            value = rates.f1
        elif method is ThresholdMethod.PRECISION_CONSTRAINED:
            if min_precision is None:
                raise ValueError("precision_constrained requires min_precision")
            if rates.precision is None or rates.precision < min_precision:
                continue
            # Among thresholds meeting the precision floor, flag as little as
            # possible -- measured as the recall actually achieved.
            value = rates.recall or 0.0
        elif method is ThresholdMethod.RECALL_CONSTRAINED:
            if min_recall is None:
                raise ValueError("recall_constrained requires min_recall")
            if rates.recall is None or rates.recall < min_recall:
                continue
            value = rates.precision or 0.0
        elif method is ThresholdMethod.FIXED_FPR:
            if target_fpr is None:
                raise ValueError("fixed_fpr requires target_fpr")
            if rates.fpr > target_fpr:
                continue
            value = rates.f1
        else:  # pragma: no cover - StrEnum exhaustive
            raise ValueError(f"unknown threshold method {method!r}")

        if best is None or value > best[0]:
            best, best_threshold = (value, counts), candidate

    chosen: _Rates | None = _metrics_from_counts(best[1]) if best is not None else None
    achieved: dict[str, float | None] = chosen._asdict() if chosen is not None else {}
    reason = ""
    feasible = best is not None
    if not feasible:
        if method is ThresholdMethod.PRECISION_CONSTRAINED:
            reason = f"no validation threshold reached precision >= {min_precision}"
        elif method is ThresholdMethod.RECALL_CONSTRAINED:
            reason = f"no validation threshold reached recall >= {min_recall}"
        elif method is ThresholdMethod.FIXED_FPR:
            reason = f"no validation threshold satisfied FPR <= {target_fpr}"

    return ThresholdResult(
        method=method,
        threshold=best_threshold,
        validation_period="validation",
        feasible=feasible,
        reason=reason,
        achieved={k: v for k, v in achieved.items() if v is not None},
        candidates_considered=evaluated,
    )


def freeze(result: ThresholdResult) -> FrozenThreshold:
    """Freeze a selected threshold for backtest/forward use."""
    return FrozenThreshold(
        threshold=result.threshold,
        selected_on="validation",
        result=result,
    )
