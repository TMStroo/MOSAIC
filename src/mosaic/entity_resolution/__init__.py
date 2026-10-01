"""Entity resolution: connect the same underlying entity across sources.

The problem
-----------
Four sources name the same actor incompatibly and imperfectly::

    transit_feed   entity_00042
    ledger_api     ENT-00042
    incident_logs  00042
    sensor_net     site::cp042

...plus ~3% of references carry a typo, and a configurable share of latent
entities are *genuinely* merged in the source data (so the correct answer is
sometimes "these two really are the same actor" and sometimes unresolvable).
Exact string equality is therefore not a solution, it is the baseline.

Approach
--------
1. **Reference extraction** - every (source, raw ref) pair becomes a reference row
   with a normalized form and provenance.
2. **Blocking** - candidate pairs come from several cheap keys so the Cartesian
   product is never formed. Every strategy is reported with its yield.
3. **Scoring** - six matchers, each returning a probability plus the *evidence
   fields* that produced it. Each is independently evaluable, because the ablation
   needs exact / normalized / rule / similarity / probabilistic / supervised.
4. **Decision** - a threshold, fixed on training data, produces
   match / possible_match / non_match decisions.
5. **Clustering** - accepted pairs are unioned into canonical entity ids via
   connected components, with a size guard against merging a whole community.
6. **Measurement** - precision, recall, F1 against the ground-truth entity map,
   plus a threshold-sensitivity curve.

Nothing in this module reads ground truth. The truth map is passed in by the
*evaluator*, not by the resolver.
"""

from __future__ import annotations

from .base import EntityResolver, ResolutionConfig, ResolutionResult
from .blocking import BlockKey, blocking_strategies, generate_candidates
from .matchers import (
    ExactMatcher,
    MatcherName,
    NormalizedMatcher,
    ProbabilisticMatcher,
    RuleMatcher,
    SimilarityMatcher,
    SupervisedMatcher,
    build_matcher,
)
from .evaluate import TruthMap, compare_matchers, evaluate_resolution, threshold_sensitivity

__all__ = [
    "BlockKey",
    "EntityResolver",
    "ExactMatcher",
    "MatcherName",
    "NormalizedMatcher",
    "ProbabilisticMatcher",
    "ResolutionConfig",
    "ResolutionResult",
    "TruthMap",
    "RuleMatcher",
    "SimilarityMatcher",
    "SupervisedMatcher",
    "blocking_strategies",
    "build_matcher",
    "compare_matchers",
    "evaluate_resolution",
    "generate_candidates",
    "threshold_sensitivity",
]
