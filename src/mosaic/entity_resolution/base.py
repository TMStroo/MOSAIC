"""The resolver: references -> candidates -> scores -> decisions -> entity ids.

Pipeline
--------
1. ``build_references`` - one row per (source, raw ref) with a normalized form and
   an occurrence count. Occurrence count matters: a reference seen 5000 times is
   evidence, a reference seen once is not.
2. ``score_candidates`` - run a matcher over the candidate set.
3. ``decide`` - threshold into match / possible_match / non_match.
4. ``cluster`` - connected components over accepted pairs, with a size guard.
5. ``assign_canonical_ids`` - every reference, matched or not, gets an entity id.

The size guard deserves a note. Connected components will happily merge an entire
community into one node if the data contains a chain of false-positive links. The
guard refuses to grow a component past ``max_component_size`` and records the
attempted edge, so the failure is visible in the report rather than silently
corrupting the graph the anomaly features are built on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from .blocking import STRATEGIES, Strategy, blocking_strategies, generate_candidates
from .matchers import (
    FEATURE_COLUMNS,
    MatcherName,
    build_matcher,
    candidate_features,
)

LOGGER = logging.getLogger(__name__)

REFERENCE_SCHEMA = {
    "ref_id": pl.Utf8,
    "source_id": pl.Utf8,
    "raw_ref": pl.Utf8,
    "entity_ref_norm": pl.Utf8,
    "occurrence_count": pl.Int64,
    "first_seen": pl.Datetime("us"),
    "last_seen": pl.Datetime("us"),
}


@dataclass
class ResolutionConfig:
    """Every resolution decision, in one configurable object."""

    name: str = "default"
    matcher: MatcherName = "probabilistic"
    match_threshold: float = 0.90
    possible_threshold: float = 0.60
    #: Defaults to every strategy in :data:`mosaic.entity_resolution.blocking.STRATEGIES`.
    #: Re-declared here rather than hard-coded so the config and the candidate
    #: generator cannot drift apart - a strategy present in one and absent from the
    #: other silently caps blocking recall.
    strategies: tuple[Strategy, ...] = STRATEGIES
    #: A block larger than this is dropped: a 3-digit suffix shared by hundreds of
    #: references carries no identity information and would flood the candidate set
    #: (measured: 300-wide blocks for a 500-entity population, which then collided
    #: with the component size guard and dropped true edges).
    max_block_size: int = 400
    #: True entity population size is unknown at inference time, so the guard is
    #: expressed as a *fraction* of the reference table. Set from the reference
    #: count by :meth:`EntityResolver._effective_max_block_size`.
    max_block_fraction: float = 0.02
    max_component_size: int = 12
    min_occurrences: int = 1
    #: When true, a reference seen only once can still be matched. Off by default:
    #: a single-observation reference is usually a malformed value, and matching
    #: it is how entity resolution invents entities.
    allow_singleton_match: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "matcher": self.matcher,
            "match_threshold": self.match_threshold,
            "possible_threshold": self.possible_threshold,
            "strategies": list(self.strategies),
            "max_block_size": self.max_block_size,
            "max_component_size": self.max_component_size,
            "min_occurrences": self.min_occurrences,
            "allow_singleton_match": self.allow_singleton_match,
        }


@dataclass
class ResolutionResult:
    """Everything the evaluator, the report and the API need."""

    config: dict[str, Any]
    references: pl.DataFrame
    candidates: pl.DataFrame
    decisions: pl.DataFrame
    entity_map: pl.DataFrame
    blocking_stats: dict[str, Any] = field(default_factory=dict)
    scoring: dict[str, Any] = field(default_factory=dict)
    clustering: dict[str, Any] = field(default_factory=dict)

    @property
    def n_entities(self) -> int:
        return int(self.entity_map["entity_id"].n_unique()) if self.entity_map.height else 0

    def accepted_pairs(self) -> pl.DataFrame:
        if self.decisions.is_empty():
            return self.decisions
        return self.decisions.filter(pl.col("decision") == "match")

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "blocking": self.blocking_stats,
            "scoring": self.scoring,
            "clustering": self.clustering,
            "n_references": self.references.height,
            "n_candidates": self.candidates.height,
            "n_decisions": self.decisions.height,
            "n_matched": self.accepted_pairs().height,
            "n_entities": self.n_entities,
        }


class EntityResolver:
    """Stateless-ish resolver driven entirely by a :class:`ResolutionConfig`."""

    def __init__(self, config: ResolutionConfig | None = None) -> None:
        self.config = config or ResolutionConfig()
        self.matcher = build_matcher(self.config.matcher)

    # -- step 1: references -------------------------------------------------
    def build_references(self, events: pl.DataFrame) -> pl.DataFrame:
        """Collapse event rows into one reference row per (source, raw ref)."""
        if events.is_empty():
            return pl.DataFrame(schema=REFERENCE_SCHEMA)
        part = events.select(
            pl.col("source_id"),
            pl.col("entity_ref_source").alias("raw_ref"),
            pl.col("entity_ref_norm"),
            pl.col("timestamp"),
        ).filter(pl.col("raw_ref").is_not_null() & (pl.col("entity_ref_norm") != ""))
        if self.config.min_occurrences > 1:
            part = part.group_by("source_id", "raw_ref", "entity_ref_norm")
            part = part.agg(
                pl.len().alias("occurrence_count"),
                pl.col("timestamp").min().alias("first_seen"),
                pl.col("timestamp").max().alias("last_seen"),
            )
        else:
            part = part.group_by("source_id", "raw_ref", "entity_ref_norm").agg(
                pl.len().alias("occurrence_count"),
                pl.col("timestamp").min().alias("first_seen"),
                pl.col("timestamp").max().alias("last_seen"),
            )
        out = part.with_columns(
            pl.concat_str([pl.col("source_id"), pl.lit("::"), pl.col("raw_ref")]).alias("ref_id")
        )
        return out.sort("ref_id")

    # -- step 2: candidates + scoring ---------------------------------------
    def _effective_max_block_size(self, references: pl.DataFrame) -> int:
        """Block cap = min(absolute, fraction of references).

        A fixed cap cannot be right for both a 5k-reference and a 1M-reference
        dataset, and the failure mode is quiet: oversized blocks are dropped, so
        recall falls without any error. The fraction makes the cap scale with the
        data while the absolute value still bounds pathological blocks.
        """
        fractional = max(20, int(references.height * self.config.max_block_fraction))
        return int(min(self.config.max_block_size, fractional))

    def generate_candidates(self, references: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
        cap = self._effective_max_block_size(references)
        return generate_candidates(
            references,
            ref_col="entity_ref_norm",
            id_col="ref_id",
            strategies=self.config.strategies,
            max_block_size=cap,
        )

    def score_candidates(
        self, candidates: pl.DataFrame, references: pl.DataFrame
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Attach features and a probability to every candidate."""
        if candidates.is_empty():
            return candidates, {"matcher": self.config.matcher, "candidates": 0}
        # The candidate frame already carries both sides' normalized references, so
        # the features are read straight off it - no id parsing, no lookup table.
        feats = candidate_features(candidates["left_ref"], candidates["right_ref"])
        scored = candidates.with_columns(
            *[pl.Series(name, feats[name]) for name in FEATURE_COLUMNS]
        )
        prob, params = self.matcher.score(feats, scored)
        scored = scored.with_columns(pl.Series("match_probability", prob, dtype=pl.Float64))
        return scored, {"matcher": self.config.matcher, "candidates": scored.height, **params}

    # -- step 3: decisions ---------------------------------------------------
    def decide(self, scored: pl.DataFrame, references: pl.DataFrame) -> pl.DataFrame:
        """Threshold the probabilities into decisions.

        A low-occurrence candidate is demoted to ``non_match`` unless the config
        allows singletons: matching a one-off reference is the most common way to
        invent entities that do not exist.
        """
        if scored.is_empty():
            return scored
        counts = dict(references.select("ref_id", "occurrence_count").iter_rows())
        min_count = 1 if self.config.allow_singleton_match else self.config.min_occurrences
        out = scored.with_columns(
            pl.min_horizontal(
                pl.col("left_ref_id").replace_strict(counts, default=0),
                pl.col("right_ref_id").replace_strict(counts, default=0),
            ).alias("min_occurrence"),
        )
        eligible = out["min_occurrence"] >= min_count
        out = out.with_columns(
            pl.when(~eligible)
            .then(pl.lit("non_match"))
            .when(pl.col("match_probability") >= self.config.match_threshold)
            .then(pl.lit("match"))
            .when(pl.col("match_probability") >= self.config.possible_threshold)
            .then(pl.lit("possible_match"))
            .otherwise(pl.lit("non_match"))
            .alias("decision"),
            pl.col("match_probability").round(6).alias("match_probability"),
        )
        return (
            out.with_columns(
                pl.lit(self.config.match_threshold, dtype=pl.Float64).alias("decision_threshold"),
                pl.col("strategies").alias("blocking_strategies"),
                pl.lit(self.config.matcher, dtype=pl.Utf8).alias("matching_method"),
            )
            .select(
                "left_ref_id",
                "right_ref_id",
                "left_source_id",
                "right_source_id",
                "left_ref",
                "right_ref",
                "match_probability",
                "matching_method",
                "decision",
                "decision_threshold",
                "blocking_strategies",
                "min_occurrence",
                *FEATURE_COLUMNS,
            )
        )

    # -- step 4: clustering --------------------------------------------------
    def cluster(self, decisions: pl.DataFrame, references: pl.DataFrame) -> tuple[dict[str, str], dict[str, Any]]:
        """Union accepted pairs into components, with a working size guard.

        Implementation notes that matter at scale:

        * union-find with path compression and union-by-size, so clustering is
          near-linear in the number of accepted edges. The earlier version
          recounted component membership per edge, which is O(n^2) and was the
          dominant cost on large candidate sets.
        * the size guard compares against the *survivor's* size (union by size), so
          refusing an edge no longer depends on which side happened to be the root.
        * edges are applied in descending probability order. Accepted edges are
          not all equally trustworthy, and merging the strongest links first is
          what makes the guard bite on the weakest ones.
        """
        accepted = decisions.filter(pl.col("decision") == "match")
        parent: dict[str, str] = {r: r for r in references["ref_id"].to_list()}
        size: dict[str, int] = dict.fromkeys(parent, 1)

        def find(node: str) -> str:
            root = node
            while parent[root] != root:
                root = parent[root]
            while parent[node] != root:
                parent[node], node = root, parent[node]
            return root

        blocked = 0
        if accepted.height:
            ordered = accepted.sort("match_probability", descending=True)
            for left, right in ordered.select("left_ref_id", "right_ref_id").iter_rows():
                if left not in parent or right not in parent:
                    continue
                ra, rb = find(left), find(right)
                if ra == rb:
                    continue
                if size[ra] + size[rb] > self.config.max_component_size:
                    blocked += 1
                    continue
                # union by size: the larger component becomes the root
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                size[ra] += size[rb]

        assignment = {ref_id: find(ref_id) for ref_id in parent}
        roots = set(assignment.values())
        component_sizes = [size[r] for r in roots]
        stats = {
            "components": len(roots),
            "accepted_pairs": accepted.height,
            "possible_pairs": int((decisions["decision"] == "possible_match").sum())
            if decisions.height
            else 0,
            "blocked_by_size_guard": blocked,
            "max_component_size": self.config.max_component_size,
            "largest_component": max(component_sizes, default=0),
            "mean_component_size": round(
                sum(component_sizes) / len(component_sizes), 4
            )
            if component_sizes
            else 0.0,
        }
        if blocked:
            LOGGER.warning(
                "entity resolution: %d accepted edges refused by the size guard "
                "(max_component_size=%d); the weakest links were dropped",
                blocked,
                self.config.max_component_size,
            )
        return assignment, stats

    # -- orchestration -------------------------------------------------------
    def run(self, events: pl.DataFrame) -> ResolutionResult:
        references = self.build_references(events)
        candidates, block_stats = self.generate_candidates(references)
        scored, scoring_stats = self.score_candidates(candidates, references)
        decisions = self.decide(scored, references)
        assignment, cluster_stats = self.cluster(decisions, references)
        entity_map = self._entity_map(references, assignment)
        return ResolutionResult(
            config=self.config.to_dict(),
            references=references,
            candidates=candidates,
            decisions=decisions,
            entity_map=entity_map,
            blocking_stats={**block_stats, "strategies": blocking_strategies()},
            scoring=scoring_stats,
            clustering=cluster_stats,
        )

    def _entity_map(self, references: pl.DataFrame, assignment: dict[str, str]) -> pl.DataFrame:
        """Reference -> canonical entity id, via the schema's stable hash."""
        from mosaic.schema.ids import entity_uid

        rows = [
            {
                "ref_id": ref_id,
                "source_id": source_id,
                "raw_ref": raw_ref,
                "entity_ref_norm": norm,
                "entity_id": entity_uid("actor", f"grp_{root}"),
                "cluster_root": f"grp_{root}",
                "occurrence_count": count,
            }
            for ref_id, (source_id, raw_ref, norm, count) in zip(
                references["ref_id"].to_list(),
                references.select("source_id", "raw_ref", "entity_ref_norm", "occurrence_count")
                .iter_rows(),
                strict=True,
            )
            for root in (assignment[ref_id],)
        ]
        return pl.DataFrame(rows, schema={
            "ref_id": pl.Utf8,
            "source_id": pl.Utf8,
            "raw_ref": pl.Utf8,
            "entity_ref_norm": pl.Utf8,
            "entity_id": pl.Utf8,
            "cluster_root": pl.Utf8,
            "occurrence_count": pl.Int64,
        })
