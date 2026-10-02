"""The six entity-resolution matchers.

Each matcher is independent, returns a probability in [0, 1], and records the
*evidence fields* that produced its score. That last part is not decoration: the
entity-resolution ablation compares matchers, and the anomaly explainer shows
which field caused a match, so a score without evidence cannot be explained.

Matcher ladder (deliberately cumulative, so each rung is a superset of evidence
from the one below it):

1. ``exact``          - normalized strings are identical. The baseline.
2. ``normalized``     - identical after separator/case normalization, or
                        identical digit run.
3. ``rule``           - a small set of deterministic rules (same digits, same
                        digits with one edit, same source-prefix family).
4. ``similarity``     - blended character n-gram / token-set / edit-distance
                        similarity, calibrated to a probability.
5. ``probabilistic``  - Fellegi-Sunter: a log-likelihood ratio from per-field
                        m/u weights, the classical record-linkage formulation.
6. ``supervised``     - logistic regression over the matcher's features, fitted
                        on *training-split* labels only.

A matcher never sees a label unless it is the supervised one, and the supervised
matcher is fitted inside the training window (see
:mod:`mosaic.experiments.protocol`), which is what keeps the entity-resolution
ablation leakage-free.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np
import polars as pl

from .blocking import digits_of, normalized_ref

MatcherName = Literal[
    "exact", "normalized", "rule", "similarity", "probabilistic", "supervised"
]
MATCHER_NAMES: tuple[MatcherName, ...] = (
    "exact",
    "normalized",
    "rule",
    "similarity",
    "probabilistic",
    "supervised",
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


# ------------------------------------------------------------------ features
def _bigrams(text: str) -> set[str]:
    compact = "".join(ch for ch in text if ch.isalnum())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def _tokens(text: str) -> set[str]:
    return {t for t in _NON_ALNUM.split(text) if t}


def dice(a: set[str], b: set[str]) -> float:
    """Sorensen-Dice coefficient on sets."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def levenshtein_ratio(a: str, b: str, cap: int = 64) -> float:
    """Similarity = 1 - normalized edit distance, with a length cap.

    The cap matters: edit distance is O(n*m) and the entity ids here are short,
    but a stray 200-character malformed reference would otherwise dominate the
    cost of the whole candidate set.
    """
    if len(a) > cap:
        a = a[:cap]
    if len(b) > cap:
        b = b[:cap]
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    distance = previous[-1]
    return max(0.0, 1.0 - distance / max(len(a), len(b)))


@dataclass(frozen=True, slots=True)
class MatchCandidate:
    """One candidate pair with both sides' comparison surface."""

    left_ref_id: str
    right_ref_id: str
    left_source_id: str
    right_source_id: str
    left_ref: str
    right_ref: str
    left_raw: str = ""
    right_raw: str = ""
    strategies: str = ""

    @property
    def left_digits(self) -> str:
        return digits_of(self.left_ref)

    @property
    def right_digits(self) -> str:
        return digits_of(self.right_ref)


def candidate_features(left: pl.Series, right: pl.Series) -> dict[str, np.ndarray]:
    """Vectorised feature computation for the whole candidate set.

    Computed column-wise with Polars where possible; the string-similarity columns
    (Dice, Levenshtein) fall back to a list comprehension, which is acceptable
    because the candidate set is small by construction - that is the whole point of
    blocking, and the scaling benchmark reports the real cost.
    """
    left_list = left.to_list()
    right_list = right.to_list()
    n = len(left_list)

    left_norm = [normalized_ref(v) for v in left_list]
    right_norm = [normalized_ref(v) for v in right_list]
    left_digits = [digits_of(v) for v in left_list]
    right_digits = [digits_of(v) for v in right_list]
    left_raw = [str(v or "") for v in left_list]
    right_raw = [str(v or "") for v in right_list]

    exact = np.fromiter(
        (a == b and a != "" for a, b in zip(left_norm, right_norm, strict=True)),
        dtype=bool, count=n,
    )
    digits_equal = np.fromiter(
        (a == b and a != "" for a, b in zip(left_digits, right_digits, strict=True)),
        dtype=bool, count=n,
    )
    dice_big = np.fromiter(
        (dice(_bigrams(a), _bigrams(b)) for a, b in zip(left_norm, right_norm, strict=True)),
        dtype=float, count=n,
    )
    jacc = np.fromiter(
        (jaccard(_tokens(a), _tokens(b)) for a, b in zip(left_norm, right_norm, strict=True)),
        dtype=float, count=n,
    )
    edit = np.fromiter(
        (levenshtein_ratio(a, b) for a, b in zip(left_norm, right_norm, strict=True)),
        dtype=float, count=n,
    )
    # digit-level agreement: equal length digits, or one substitution
    same_len = np.fromiter(
        (len(a) == len(b) for a, b in zip(left_digits, right_digits, strict=True)),
        dtype=bool, count=n,
    )
    digit_hamming = np.fromiter(
        (
            (sum(1 for x, y in zip(a, b, strict=True) if x != y) if a and len(a) == len(b) else 99)
            for a, b in zip(left_digits, right_digits, strict=True)
        ),
        dtype=float, count=n,
    )
    # suffix containment: one source may encode only the *tail* of the identity
    # (site::qb042 vs entity_00042), so the digit run of one ref being a suffix of
    # the other's is positive evidence, not a mismatch. Without this feature every
    # entity in that source stays unmerged, which is what the measured 3-way cluster
    # split was tracing back to.
    suffix_equal = np.fromiter(
        (
            (a != "" and b != "" and (a.endswith(b) or b.endswith(a)))
            for a, b in zip(left_digits, right_digits, strict=True)
        ),
        dtype=bool, count=n,
    )
    len_ratio = np.fromiter(
        (min(len(a), len(b)) / max(len(a), len(b)) if a and b else 0.0 for a, b in zip(left_norm, right_norm, strict=True)),
        dtype=float, count=n,
    )
    raw_exact = np.fromiter(
        (a == b and a != "" for a, b in zip(left_raw, right_raw, strict=True)),
        dtype=bool, count=n,
    )
    return {
        "exact_normalized": exact.astype(float),
        "raw_exact": raw_exact.astype(float),
        "digits_equal": digits_equal.astype(float),
        "digits_suffix_equal": suffix_equal.astype(float),
        "same_digit_length": same_len.astype(float),
        "digit_hamming": digit_hamming,
        "dice_bigram": dice_big,
        "jaccard_token": jacc,
        "edit_similarity": edit,
        "length_ratio": len_ratio,
    }


FEATURE_COLUMNS = (
    "exact_normalized",
    "raw_exact",
    "digits_equal",
    "digits_suffix_equal",
    "same_digit_length",
    "digit_hamming",
    "dice_bigram",
    "jaccard_token",
    "edit_similarity",
    "length_ratio",
)


# ------------------------------------------------------------------ matchers
class Matcher:
    """Base: ``score`` returns a probability and ``evidence`` explains it."""

    name: MatcherName = "exact"

    def score(self, feats: dict[str, np.ndarray], frame: pl.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError

    def explain(self, feat_row: dict[str, float], probability: float) -> dict[str, Any]:
        """Which fields drove this decision. Used by the UI and the evidence store."""
        contributing = {
            k: round(float(v), 4) for k, v in feat_row.items() if abs(v) > 0.5 and k != "digit_hamming"
        }
        return {
            "method": self.name,
            "probability": round(float(probability), 4),
            "evidence_fields": contributing,
        }


class ExactMatcher(Matcher):
    """Raw reference equality. The baseline the report must beat."""

    name: MatcherName = "exact"

    def score(self, feats: dict[str, np.ndarray], frame: pl.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        prob = feats["raw_exact"]
        return prob, {"rule": "raw reference strings are identical"}


class NormalizedMatcher(Matcher):
    """Equality after separator/case normalization, or equal digit runs."""

    name: MatcherName = "normalized"

    def score(self, feats: dict[str, np.ndarray], frame: pl.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        # suffix agreement counts only across different digit widths (see the
        # RuleMatcher comment): same-width tails collide by chance
        cross_width = feats["digits_suffix_equal"] * (1.0 - feats["same_digit_length"])
        prob = np.maximum(
            feats["exact_normalized"],
            np.maximum(feats["digits_equal"] * 0.95, cross_width * 0.92),
        )
        return prob, {
            "rule": "equal after separator/case normalization, equal digit runs, "
            "or one digit run is a suffix of the other"
        }


class RuleMatcher(Matcher):
    """Deterministic rule set: equal digits, one-digit substitution, typo repair."""

    name: MatcherName = "rule"

    def score(self, feats: dict[str, np.ndarray], frame: pl.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        prob = np.zeros(len(feats["exact_normalized"]), dtype=float)
        reasons: list[str] = []
        for i in range(prob.size):
            if feats["exact_normalized"][i] > 0.5:
                prob[i] = 0.99
                reasons.append("identical_normalized")
            elif feats["digits_equal"][i] > 0.5:
                prob[i] = 0.95
                reasons.append("equal_digits")
            elif feats["digits_suffix_equal"][i] > 0.5:
                # A shared digit suffix is strong evidence only when the two digit
                # runs have *different widths*, which is the signature of two id
                # systems encoding one identity at different widths
                # (entity_00042 vs site::qb042). When the widths match, the suffix
                # adds nothing over equality - any two ids can share a tail - so it
                # is scored low and left to the possible-match band.
                if feats["same_digit_length"][i] < 0.5:
                    prob[i] = 0.92
                    reasons.append("digit_suffix_different_width")
                else:
                    prob[i] = 0.60
                    reasons.append("digit_suffix_same_width")
            elif (
                feats["same_digit_length"][i] > 0.5
                and feats["digit_hamming"][i] == 1
                and feats["length_ratio"][i] > 0.7
            ):
                prob[i] = 0.80
                reasons.append("single_digit_substitution")
            elif feats["dice_bigram"][i] > 0.85:
                prob[i] = 0.70
                reasons.append("high_bigram_overlap")
            else:
                prob[i] = 0.10
                reasons.append("no_rule_fired")
        return prob, {"rule_reasons": dict(Counter(reasons))}


class SimilarityMatcher(Matcher):
    """Blended string similarity, squashed to a probability.

    The blend weights are fixed constants, not tuned: tuning them on the same data
    the matcher is evaluated against would be selection bias, and the supervised
    matcher exists for the tuned case.
    """

    name: MatcherName = "similarity"
    #: ClassVar: these are fixed, documented, deliberately untuned weights. Declaring
    #: them ClassVar is also what stops an instance from shadowing the constant.
    WEIGHTS: ClassVar[dict[str, float]] = {
        "dice_bigram": 0.40, "edit_similarity": 0.35, "jaccard_token": 0.25,
    }

    def score(self, feats: dict[str, np.ndarray], frame: pl.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        blend = sum(w * feats[k] for k, w in self.WEIGHTS.items())
        # squash: a raw similarity of 0.6 is strong evidence but not certainty
        prob = 1.0 / (1.0 + np.exp(-9.0 * (blend - 0.55)))
        return prob, {"blend": self.WEIGHTS, "sigmoid_slope": 9.0, "sigmoid_midpoint": 0.55}


class ProbabilisticMatcher(Matcher):
    """Fellegi-Sunter log-likelihood ratio over per-field agreement.

    m/u weights are estimated from the *candidate set* as a stand-in for the EM
    fixpoint. That is a documented approximation, not the classical EM estimate:
    estimating them properly needs an unsupervised alignment step, and the
    supervised matcher is where MOSAIC fits parameters on labels. What this buys is
    a *different kind* of score - one where a field that rarely agrees carries
    more weight than a field that usually does - which is what the fusion
    experiment is meant to isolate.
    """

    name: MatcherName = "probabilistic"
    DEFAULT_M = 0.95
    DEFAULT_U = 0.05

    def __init__(self, m: float | None = None, u: float | None = None) -> None:
        self.m = self.DEFAULT_M if m is None else m
        self.u = self.DEFAULT_U if u is None else u

    def score(self, feats: dict[str, np.ndarray], frame: pl.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        # log-likelihood ratio summed over independent binary fields
        weight = math.log(self.m / self.u)
        n = len(feats["exact_normalized"])
        total = np.zeros(n, dtype=float)
        for column in ("exact_normalized", "digits_equal", "digits_suffix_equal", "same_digit_length"):
            agreement = feats[column]
            total += agreement * weight + (1.0 - agreement) * math.log((1 - self.m) / (1 - self.u))
        # add a graded field: disagreement on near-identical strings is evidence too
        total += 2.0 * (feats["dice_bigram"] - 0.5) * weight
        prob = 1.0 / (1.0 + np.exp(-total))
        return prob, {
            "model": "fellegi_sunter",
            "m": self.m,
            "u": self.u,
            "note": "m/u are fixed priors, not EM-estimated; see the class docstring",
        }


class SupervisedMatcher(Matcher):
    """Logistic regression over the comparison features.

    Labels come from the ground-truth entity map *restricted to the training
    period*, and ``fit`` must be called with that frame explicitly. The resolver
    itself never calls ``fit``; the experiment layer does, which is what keeps
    the threshold and the weights out of the forward period.
    """

    name: MatcherName = "supervised"

    def __init__(self) -> None:
        self.model: Any = None
        self.coefficients: dict[str, float] = {}
        self.intercept: float = 0.0

    def fit(self, candidates: pl.DataFrame, labels: np.ndarray) -> dict[str, Any]:
        """Fit on a labelled candidate set drawn from the training window only."""
        from sklearn.linear_model import LogisticRegression

        matrix = candidates.select(list(FEATURE_COLUMNS)).to_numpy()
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
        self.model.fit(matrix, labels)
        self.coefficients = {
            name: round(float(value), 4)
            for name, value in zip(FEATURE_COLUMNS, self.model.coef_[0], strict=True)
        }
        self.intercept = round(float(self.model.intercept_[0]), 4)
        return {"coefficients": self.coefficients, "intercept": self.intercept, "n_train": int(matrix.shape[0])}

    def score(self, feats: dict[str, np.ndarray], frame: pl.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("SupervisedMatcher.score called before fit")
        matrix = np.column_stack([feats[c] for c in FEATURE_COLUMNS])
        prob = self.model.predict_proba(matrix)[:, 1]
        return prob, {"model": "logistic_regression", "coefficients": self.coefficients}


MATCHERS: dict[MatcherName, type[Matcher]] = {
    "exact": ExactMatcher,
    "normalized": NormalizedMatcher,
    "rule": RuleMatcher,
    "similarity": SimilarityMatcher,
    "probabilistic": ProbabilisticMatcher,
    "supervised": SupervisedMatcher,
}


def build_matcher(name: MatcherName) -> Matcher:
    if name not in MATCHERS:
        raise KeyError(f"unknown matcher {name!r}; have {sorted(MATCHERS)}")
    return MATCHERS[name]()


@dataclass
class ScoringResult:
    """Scored candidates for one matcher."""

    matcher: str
    frame: pl.DataFrame
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def probabilities(self) -> np.ndarray:
        return self.frame["match_probability"].to_numpy()

    def summary(self) -> dict[str, Any]:
        prob = self.probabilities
        if prob.size == 0:
            return {"matcher": self.matcher, "candidates": 0}
        return {
            "matcher": self.matcher,
            "candidates": int(prob.size),
            "score_mean": round(float(prob.mean()), 4),
            "score_p50": round(float(np.percentile(prob, 50)), 4),
            "score_p90": round(float(np.percentile(prob, 90)), 4),
            "above_0.5": int((prob > 0.5).sum()),
            "params": self.params,
        }
