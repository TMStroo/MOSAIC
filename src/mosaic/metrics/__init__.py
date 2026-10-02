"""Anomaly-detection metrics, with the uncertainty that small support demands.

Accuracy is deliberately not implemented. On a 0.03% positive rate a trivial
always-negative classifier scores 99.97% accuracy, so accuracy measures the
prevalence, not the detector.

Every rate metric returns a point estimate *and* a 95% interval. With EVENT_ONLY
the forward period holds 21 positives, where recall of 0.60 and recall of 0.30
are barely distinguishable; a point estimate alone would invite exactly the
model ranking that 21 samples cannot support.

Intervals are Wilson score intervals for proportions, which stay inside [0, 1]
and remain sensible at the extreme rates and tiny counts here -- unlike the
normal approximation, whose bounds leave the valid range when ``p_hat`` is 0 or
1. No SciPy dependency is required.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import polars as pl

#: z for a two-sided 95% interval.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    """A point estimate with a 95% confidence interval."""

    point: float
    low: float
    high: float
    method: str = "wilson"

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "ci95_low": self.low,
            "ci95_high": self.high,
            "ci_method": self.method,
        }


@dataclass
class PeriodMetrics:
    """Metrics for one detector on one period, under one label protocol."""

    period: str
    label_protocol: str
    n_rows: int
    support_positive: int
    support_negative: int
    precision: Interval | None = None
    recall: Interval | None = None
    f1: Interval | None = None
    pr_auc: float | None = None
    roc_auc: float | None = None
    false_positive_rate: Interval | None = None
    threshold: float | None = None
    n_predicted_positive: int | None = None
    confusion: dict[str, int] = field(default_factory=dict)
    not_applicable: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "period": self.period,
            "label_protocol": self.label_protocol,
            "n_rows": self.n_rows,
            "support_positive": self.support_positive,
            "support_negative": self.support_negative,
            "threshold": self.threshold,
            "n_predicted_positive": self.n_predicted_positive,
            "confusion": self.confusion,
            "not_applicable": self.not_applicable,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
        }
        for name in ("precision", "recall", "f1", "false_positive_rate"):
            value = getattr(self, name)
            out[name] = value.to_dict() if value else None
        return out


def wilson_interval(successes: int, trials: int, *, z: float = Z_95) -> Interval | None:
    """Wilson score interval for a proportion.

    Returns ``None`` when ``trials`` is 0 rather than 0.0 or 1.0: with no
    observations the rate is unknown, and reporting 0.0 would assert "this
    detector found nothing" when nothing was measured.
    """
    if trials <= 0:
        return None
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return Interval(point=p, low=max(0.0, centre - margin), high=min(1.0, centre + margin))


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def confusion_counts(
    y_true: Sequence[bool] | pl.Series, y_pred: Sequence[bool] | pl.Series
) -> dict[str, int]:
    """Confusion matrix counts. Always returns all four keys, even when zero."""
    true_s = y_true if isinstance(y_true, pl.Series) else pl.Series("t", list(y_true), dtype=pl.Boolean)
    pred_s = y_pred if isinstance(y_pred, pl.Series) else pl.Series("p", list(y_pred), dtype=pl.Boolean)
    if len(true_s) != len(pred_s):
        raise ValueError(f"length mismatch: {len(true_s)} true vs {len(pred_s)} predicted")
    frame = pl.DataFrame({"t": true_s, "p": pred_s})
    counts = frame.group_by(["t", "p"]).len().to_dicts()
    out = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for row in counts:
        key = ("tp" if row["p"] else "fn") if row["t"] else ("fp" if row["p"] else "tn")
        out[key] = int(row["len"])
    return out


def _f1_from_counts(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else None


def evaluate_predictions(
    *,
    period: str,
    label_protocol: str,
    y_true: pl.Series,
    y_pred: pl.Series,
    threshold: float | None = None,
    scores: pl.Series | None = None,
) -> PeriodMetrics:
    """Full metric block for one period, with NOT_APPLICABLE marking.

    Metrics that are undefined for the observed support are recorded as
    ``not_applicable`` and left ``None`` -- never filled with 0.0, which would be
    indistinguishable from a genuine zero.
    """
    counts = confusion_counts(y_true, y_pred)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    support_pos = tp + fn
    support_neg = fp + tn
    metrics = PeriodMetrics(
        period=period,
        label_protocol=label_protocol,
        n_rows=len(y_true),
        support_positive=support_pos,
        support_negative=support_neg,
        threshold=threshold,
        n_predicted_positive=tp + fp,
        confusion=counts,
    )

    if support_pos == 0:
        metrics.not_applicable.append("recall")
        metrics.not_applicable.append("pr_auc")
    if support_neg == 0:
        metrics.not_applicable.append("false_positive_rate")
    if support_pos == 0 and tp + fp == 0:
        metrics.not_applicable.append("precision")

    metrics.precision = wilson_interval(tp, tp + fp) if (tp + fp) > 0 else None
    if support_pos:
        metrics.recall = wilson_interval(tp, support_pos)
    f1 = _f1_from_counts(tp, fp, fn)
    if f1 is not None:
        metrics.f1 = Interval(point=f1, low=0.0, high=1.0, method="none")
    if support_neg:
        metrics.false_positive_rate = wilson_interval(fp, support_neg)

    if scores is not None and support_pos and support_neg:
        metrics.pr_auc = average_precision(y_true, scores)
        metrics.roc_auc = roc_auc(y_true, scores)
    elif scores is not None and support_pos:
        metrics.not_applicable.append("pr_auc: no negatives in period")
        metrics.not_applicable.append("roc_auc: no negatives in period")
    elif scores is not None and support_neg:
        metrics.not_applicable.append("pr_auc: no positives in period")
        metrics.not_applicable.append("roc_auc: no positives in period")
    return metrics


def average_precision(y_true: pl.Series, scores: pl.Series) -> float | None:
    """Average precision (area under the precision-recall curve).

    Implemented from the precision-recall curve rather than delegating to
    sklearn so the metric is defined identically on the tiny supports here and
    so its behaviour on ties is explicit: tied scores are ranked together.
    Returns ``None`` if there are no positives, or if every score is null --
    a null score carries no ranking information, so reporting a value would
    invent one.
    """
    n_pos = int(y_true.sum() or 0)
    if n_pos == 0:
        return None
    if scores.drop_nulls().is_empty():
        return None
    frame = pl.DataFrame({"t": y_true, "s": scores}).sort("s", descending=True)
    tp = frame.group_by("s", maintain_order=True).agg(pl.col("t").sum().alias("tp"), pl.len().alias("n"))
    tp_cum = tp["tp"].cum_sum().cast(pl.Float64)
    k_cum = tp["n"].cum_sum().cast(pl.Float64)
    precision = tp_cum / k_cum
    recall = tp_cum / n_pos
    # Step-function area: recall increments times the precision in force there.
    recall_deltas = recall.diff().fill_null(recall[0])
    return float((precision * recall_deltas).sum())


def roc_auc(y_true: pl.Series, scores: pl.Series) -> float | None:
    """ROC-AUC via the rank/Mann-Whitney identity, tie-aware.

    Secondary metric by design: on a heavily imbalanced anomaly problem a high
    ROC-AUC can coexist with useless precision, so it is reported but never used
    to select a model. Returns ``None`` when all scores are null.
    """
    n_pos = int(y_true.sum() or 0)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    if scores.drop_nulls().is_empty():
        return None
    frame = pl.DataFrame({"t": y_true, "s": scores})
    ranks = frame.select(
        pl.col("s").rank(method="average", descending=False).alias("r")
    )["r"]
    positive_rank_sum = float(ranks.filter(frame["t"]).sum())
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def per_family_metrics(
    *,
    frame: pl.DataFrame,
    period: str,
    label_protocol: str,
    family_column: str = "anomaly_family",
    label_column: str = "is_anomaly",
    pred_column: str = "predicted_anomaly",
    minimum_support: int,
    families: Sequence[str],
    coverage: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-family support / precision / recall / F1, with NOT_APPLICABLE.

    A family's metric is computed on rows carrying that family. Rows carrying no
    family serve as its negatives -- which is the honest framing: can the
    detector distinguish this family from ordinary behaviour? Under ALL_FAMILIES
    a family that is also covered by another family's window would otherwise
    have its own positives counted as negatives.
    """
    out: dict[str, dict[str, Any]] = {}
    positive_mask = frame[label_column].cast(pl.Boolean)
    predicted = frame[pred_column].cast(pl.Boolean)
    has_family = frame[family_column].list.len() > 0

    for family in families:
        entry: dict[str, Any] = {
            "family": family,
            "label_protocol": label_protocol,
            "period": period,
            "coverage": int((coverage or {}).get(family, 0)),
        }
        mask = frame[family_column].list.eval(pl.element() == family).list.any()
        support = int((mask & positive_mask).sum() or 0)
        entry["support"] = support
        if support < minimum_support:
            entry["status"] = "NOT_APPLICABLE"
            entry["reason"] = f"support {support} < minimum {minimum_support}"
            for key in ("precision", "recall", "f1", "false_positive_rate"):
                entry[key] = None
            entry["confidence_intervals"] = {}
            out[family] = entry
            continue

        counts = confusion_counts(positive_mask.filter(mask | ~mask).alias(None), predicted) if False else _masked_confusion(
            mask, positive_mask, predicted
        )
        tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        entry["status"] = "OK"
        entry["confusion"] = counts
        entry["precision"] = wilson_interval(tp, tp + fp)
        entry["recall"] = wilson_interval(tp, tp + fn)
        entry["f1"] = wilson_interval(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) else None
        entry["false_positive_rate"] = wilson_interval(fp, fp + tn)
        entry["confidence_intervals"] = {
            "precision": entry["precision"].to_dict() if entry["precision"] else None,
            "recall": entry["recall"].to_dict() if entry["recall"] else None,
        }
        out[family] = entry
    return out


def _masked_confusion(
    mask: pl.Series, positive: pl.Series, predicted: pl.Series
) -> dict[str, int]:
    """Confusion restricted to ``mask`` rows, with ``~mask`` rows as negatives."""
    sub_true = pl.Series([bool(p) and bool(m) for p, m in zip(positive, mask)])
    sub_pred = pl.Series([bool(p) and bool(m) for p, m in zip(predicted, mask)])
    # Rows outside the family are negatives for this family's task.
    sub_true = pl.Series([bool(t) for t in sub_true])
    sub_pred = pl.Series([bool(p) for p in sub_pred])
    return confusion_counts(sub_true, sub_pred)
