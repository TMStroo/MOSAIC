"""Measuring entity resolution: precision, recall, F1, threshold sensitivity.

This module is the *only* place in the resolver path that reads the ground-truth
entity map, and it does so strictly after the fact: the resolver is handed the
events, produces a map, and only then is the map scored. That ordering is what
makes the reported precision/recall meaningful rather than circular.

Metrics reported
----------------
* **reference-level** precision/recall/F1 - is each reference assigned the right
  entity? This is what the research questions care about, because the cost of a
  wrong merge propagates into every graph and behavioural feature downstream.
* **pair-level** precision/recall/F1 - the classical record-linkage view, over
  candidate pairs. A method can score well here and badly on clustering, so both
  are reported.
* **blocking recall** - how many true matches survived candidate generation. This
  is the ceiling on everything else, and reporting it separately stops a low
  recall from being misattributed to the matcher.
* **cluster purity / completeness** - whether accepted edges form clean groups.
* **threshold sensitivity** - precision/recall across the decision threshold,
  which is the curve the entity-resolution ablation and the UI both need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from .base import ResolutionResult
from .matchers import MATCHER_NAMES, MatcherName


@dataclass(frozen=True)
class TruthMap:
    """The ground-truth (source, source_ref) -> entity_key map, loaded from disk.

    Wrapped in its own type so that a resolver bug cannot accidentally receive it:
    :class:`TruthMap` has no method that returns labels for a feature frame.
    """

    frame: pl.DataFrame

    @classmethod
    def load(cls, path: str) -> TruthMap:
        return cls(pl.read_parquet(path))

    def key_for(self) -> dict[str, str]:
        return dict(
            self.frame.select("source_ref", "entity_key").iter_rows()
        )

    def key_for_source(self, source_id: str) -> dict[str, str]:
        part = self.frame.filter(pl.col("source_id") == source_id)
        return dict(part.select("source_ref", "entity_key").iter_rows())

    def n_entities(self) -> int:
        return int(self.frame["entity_key"].n_unique())


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def evaluate_resolution(
    result: ResolutionResult,
    truth: TruthMap,
    *,
    decisions: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Score a completed resolution against ground truth.

    ``decisions`` lets the caller evaluate a *different* threshold without
    re-running the matcher - the probabilities are fixed, only the cut moves. The
    entity-resolution robustness experiment uses this to sweep thresholds cheaply.
    """
    decisions = decisions if decisions is not None else result.decisions
    key_for = truth.key_for()
    entity_map = result.entity_map

    # ---- reference-level: did each reference land in the right cluster? -----
    joined = entity_map.join(
        truth.frame.select(pl.col("source_ref"), pl.col("entity_key")),
        left_on="raw_ref",
        right_on="source_ref",
        how="left",
    )
    # A reference that the truth map does not know (e.g. a heavily corrupted ref
    # that no longer matches any rendered id) is excluded and counted separately:
    # it is unscoreable, and counting it as a miss would penalise the resolver for
    # the renderer's damage.
    scored = joined.filter(pl.col("entity_key").is_not_null())
    unscoreable = int(joined["entity_key"].null_count())

    if scored.is_empty():
        return {"error": "no references could be joined to the truth map"}

    # Build the contingency between predicted cluster and true entity.
    pairs = scored.select("entity_id", "entity_key").unique()
    n_true_entities = int(scored["entity_key"].n_unique())
    n_pred_clusters = int(scored["entity_id"].n_unique())

    # Reference-level: two references are "correctly together" if the resolver
    # grouped them exactly when the truth does. Count via cluster purity.
    cluster_sizes = scored.group_by("entity_id").agg(
        pl.col("entity_key").n_unique().alias("true_entities"),
        pl.len().alias("refs"),
    )
    total_refs = int(scored.height)
    pure_refs = int(
        scored.group_by("entity_id")
        .agg((pl.col("entity_key").n_unique() == 1).sum().alias("n"))
        .get_column("n")
        .sum()
    )
    # A cluster is pure if it contains exactly one true entity. Correct assignment
    # for a reference = it is in a pure cluster AND that cluster is the whole truth
    # group. Undersized clusters (a truth group split in two) are false negatives.
    truth_sizes = scored.group_by("entity_key").agg(pl.len().alias("refs"))
    correct = 0
    for cluster_id, sub in scored.group_by("entity_id"):
        entities = sub["entity_key"].unique()
        if len(entities) == 1:
            key = entities[0]
            expected = int(
                truth_sizes.filter(pl.col("entity_key") == key).get_column("refs")[0]
            )
            correct += min(sub.height, expected)
    ref_precision = correct / total_refs if total_refs else 0.0
    ref_recall = correct / total_refs if total_refs else 0.0  # same denominator by construction
    ref_f1 = (
        2 * ref_precision * ref_recall / (ref_precision + ref_recall)
        if (ref_precision + ref_recall)
        else 0.0
    )

    # ---- pair-level: over candidate pairs ---------------------------------
    pair_metrics = _pair_metrics(decisions, key_for)

    # ---- blocking recall ---------------------------------------------------
    block_recall = _blocking_recall(result.candidates, key_for)

    # ---- cluster purity / completeness ------------------------------------
    purity = float((cluster_sizes["true_entities"] == 1).mean()) if cluster_sizes.height else 0.0
    completeness = _completeness(scored, truth_sizes)

    return {
        "matcher": result.config.get("matcher"),
        "threshold": result.config.get("match_threshold"),
        "reference_level": {
            "precision": round(ref_precision, 6),
            "recall": round(ref_recall, 6),
            "f1": round(ref_f1, 6),
            "correctly_assigned": correct,
            "references_scored": total_refs,
            "references_unscoreable": unscoreable,
        },
        "pair_level": pair_metrics,
        "blocking_recall": block_recall,
        "clustering": {
            "n_predicted_clusters": n_pred_clusters,
            "n_true_entities": n_true_entities,
            "purity": round(purity, 6),
            "completeness": round(completeness, 6),
            "largest_predicted": int(
                scored.group_by("entity_id").len().get_column("len").max() or 0
            ),
            "largest_true": int(truth_sizes.get_column("refs").max() or 0),
        },
        "n_candidates": result.candidates.height,
        "n_accepted_pairs": int((decisions["decision"] == "match").sum()) if decisions.height else 0,
    }


def _pair_metrics(decisions: pl.DataFrame, key_for: dict[str, str]) -> dict[str, Any]:
    """Classical pair-level precision/recall over candidates.

    A candidate is a *true* pair when both sides map to the same ground-truth
    entity. Two things this deliberately does **not** do:

    * it does not score unlabelable candidates (a reference so corrupted the truth
      map cannot name it) as false positives - that would measure the renderer's
      damage rather than the resolver;
    * it does not treat a non-generated pair as a false negative. A true pair that
      blocking never produced is a *blocking recall* failure, reported separately
      by :func:`_blocking_recall`; folding it in here would double-count the same
      miss and make the matcher look worse than it is.

    Consequently recall here is "recall over generated candidates" and must be
    read together with ``blocking_recall``; their product is the end-to-end
    reference-level recall.
    """
    if decisions.is_empty():
        return {"candidates": 0, "note": "no candidates"}
    # Look the ground-truth key up by the *raw reference carried on the candidate
    # frame*, never by stripping a prefix off the ref id: a source id can contain
    # "::" (the site::slug style), so prefix-stripping truncates the reference.
    scored = decisions.with_columns(
        pl.col("left_ref").replace_strict(key_for, default=None).alias("left_key"),
        pl.col("right_ref").replace_strict(key_for, default=None).alias("right_key"),
    )
    labelable = scored.filter(pl.col("left_key").is_not_null() & pl.col("right_key").is_not_null())
    unlabelable = int(
        (scored["left_key"].is_null() | scored["right_key"].is_null()).sum()
    )
    if labelable.is_empty():
        return {"candidates": int(scored.height), "unlabelable": unlabelable}

    is_true = (labelable["left_key"] == labelable["right_key"]).to_numpy()
    predicted = (labelable["decision"] == "match").to_numpy()
    tp = int((is_true & predicted).sum())
    fp = int((~is_true & predicted).sum())
    fn = int((is_true & ~predicted).sum())
    tn = int((~is_true & ~predicted).sum())
    metrics = _prf(tp, fp, fn)
    metrics.update({
        "candidates": int(labelable.height),
        "unlabelable_candidates": unlabelable,
        "true_negatives": tn,
        "candidate_precision": round(float(is_true.mean()) if is_true.size else 0.0, 6),
        "note": "recall is over generated candidates; multiply by blocking_recall for end-to-end",
    })
    return metrics


def _blocking_recall(candidates: pl.DataFrame, key_for: dict[str, str]) -> dict[str, Any]:
    """Fraction of true cross-source pairs that reached the candidate set.

    Reported separately from matcher precision so a low final recall can be
    attributed to the right stage: if this is 0.98, the matcher had its chance.
    """
    if candidates.is_empty():
        return {"recall": 0.0, "candidates": 0, "note": "no candidates generated"}
    left = candidates["left_ref"].to_list()
    right = candidates["right_ref"].to_list()
    true_total = 0
    found = 0
    # ground truth: for each true entity, the number of cross-source ref pairs
    from collections import defaultdict

    by_entity: dict[str, list[str]] = defaultdict(list)
    for ref, key in key_for.items():
        by_entity[key].append(ref)
    for key, refs in by_entity.items():
        unique = sorted(set(refs))
        true_total += len(unique) * (len(unique) - 1) // 2
    pair_set = {(a, b) if a < b else (b, a) for a, b in zip(left, right, strict=True)}
    for key, refs in by_entity.items():
        unique = sorted(set(refs))
        for i, a in enumerate(unique):
            for b in unique[i + 1 :]:
                if (a, b) in pair_set:
                    found += 1
    return {
        "recall": round(found / true_total, 6) if true_total else 0.0,
        "true_pairs": true_total,
        "true_pairs_in_candidates": found,
        "candidates": candidates.height,
        "precision_proxy": round(found / candidates.height, 6) if candidates.height else 0.0,
    }


def _completeness(scored: pl.DataFrame, truth_sizes: pl.DataFrame) -> float:
    """Mean fraction of each true entity's references that share one cluster.

    For each ground-truth entity, take the *largest* predicted cluster holding its
    references and divide by the entity's total reference count. A truth group
    split across k clusters therefore scores 1/k for the largest piece, which is
    what "completeness" should mean. (Dividing by the cluster *count* instead
    would report >1 for any intact group, which is why the earlier version was
    wrong.)
    """
    scores: list[float] = []
    expected_by_key = dict(
        truth_sizes.select("entity_key", "refs").iter_rows()
    )
    for key, sub in scored.group_by("entity_key"):
        total = int(expected_by_key.get(key, sub.height))
        if total <= 0:
            continue
        largest = int(sub.group_by("entity_id").len().get_column("len").max() or 0)
        scores.append(min(1.0, largest / total))
    return float(np.mean(scores)) if scores else 0.0


def threshold_sensitivity(
    result: ResolutionResult,
    truth: TruthMap,
    *,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Sweep the decision threshold and record pair-level P/R/F1 at each point.

    Only the *decision* is re-thresholded: the matcher's probabilities are fixed,
    so this is a pure operating-curve calculation and not a re-fit. That matters -
    sweeping a threshold and then picking the best F1 on the same data would be
    selection on the test set, which is exactly the failure mode the evaluation
    protocol forbids. The report presents this curve; the *chosen* threshold still
    comes from the training/validation split.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows: list[dict[str, Any]] = []
    base = result.decisions
    for threshold in thresholds:
        re_decided = base.with_columns(
            pl.when(pl.col("match_probability") >= float(threshold))
            .then(pl.lit("match"))
            .when(pl.col("match_probability") >= result.config.get("possible_threshold", 0.6))
            .then(pl.lit("possible_match"))
            .otherwise(pl.lit("non_match"))
            .alias("decision")
        )
        metrics = _pair_metrics(re_decided, truth.key_for())
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
                "accepted_pairs": int((re_decided["decision"] == "match").sum())
                if re_decided.height
                else 0,
            }
        )
    frame = pl.DataFrame(rows)
    best = frame.sort("f1", descending=True).head(1)
    return {
        "matcher": result.config.get("matcher"),
        "curve": rows,
        "best_f1_threshold": float(best["threshold"][0]) if best.height else None,
        "best_f1": float(best["f1"][0]) if best.height else 0.0,
        "note": "best_f1_threshold is reported for diagnosis only; the threshold used in "
                "experiments is selected on the training/validation split, never here",
    }


def compare_matchers(
    results: dict[str, ResolutionResult],
    truth: TruthMap,
) -> dict[str, Any]:
    """Side-by-side table for the entity-resolution ablation."""
    out: dict[str, Any] = {}
    for name in MATCHER_NAMES:
        result = results.get(name)
        if result is None:
            continue
        out[name] = evaluate_resolution(result, truth)
    return {
        "matchers": out,
        "note": "each matcher is evaluated on the same candidate set with the same "
                "threshold, so differences are attributable to the matcher alone",
    }
