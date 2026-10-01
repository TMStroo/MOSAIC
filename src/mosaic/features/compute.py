"""Feature computation: eight families, all causally scoped.

The central guarantee
--------------------
Every feature at timestamp ``t`` is computed from rows with ``timestamp <= t``,
and every *fitted* statistic comes from a caller-supplied training frame that
predates the rows being transformed. :func:`assert_causal` re-derives this
invariant on the output and the leakage test suite runs it on real data.

Implementation approach
------------------------
Causal windowed aggregations use ``cum_sum`` over a sorted frame rather than a
groupby-shift: a shift by *n* rows is not a shift by *n* seconds, and the feature
semantics here are defined in time, not in rows. Where a window must exclude the
current row (to avoid trivial self-inclusion), the aggregation is taken over the
window *strictly before* ``t`` and the current row is added separately where
appropriate - stated per feature in its docstring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import polars as pl

from mosaic.features.registry import FeatureFamily, FeatureRegistry, FeatureSpec, LeakageClass
from mosaic.utils.timing import StageMetrics, stage_timer

LOGGER = logging.getLogger(__name__)

SECOND = 1
MINUTE = 60
HOUR = 3600
DAY = 86_400
WEEK = 604_800

#: Default windows, in seconds, per feature family. The ablation varies these.
DEFAULT_WINDOWS: dict[str, list[int]] = {
    "short": [6 * HOUR, DAY],
    "medium": [DAY, 7 * DAY],
    "long": [7 * DAY, 30 * DAY],
}


@dataclass
class FeatureConfig:
    """Which features to compute, with which windows, at which cost."""

    name: str = "feature_set_v1"
    families: tuple[FeatureFamily, ...] = tuple(FeatureFamily)
    windows: dict[str, int] = field(default_factory=lambda: {"short": 6 * HOUR, "medium": 7 * DAY, "long": 30 * DAY})
    rolling_hours: tuple[int, ...] = (1, 6, 24, 168)
    include_sequence: bool = True
    include_cross_source: bool = True
    include_graph: bool = True
    #: Fraction of training rows sampled when estimating fitted statistics. Below
    #: 1.0 the estimate is noisier but the *scope* is unchanged (still train-only).
    fit_sample_fraction: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "families": [f.value for f in self.families],
            "windows": dict(self.windows),
            "rolling_hours": list(self.rolling_hours),
            "include_sequence": self.include_sequence,
            "include_cross_source": self.include_cross_source,
            "include_graph": self.include_graph,
            "fit_sample_fraction": self.fit_sample_fraction,
        }


@dataclass
class FitState:
    """Statistics estimated on a training period, reused for later periods.

    Passing this explicitly is what makes the train/forward boundary enforceable:
    a transform call that does not receive a ``FitState`` cannot silently use
    forward data, because it has no access to any.
    """

    period: str
    fitted_through: Any
    n_rows: int
    entity_baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    global_stats: dict[str, float] = field(default_factory=dict)
    location_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    source_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    type_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    hour_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    #: (prev_type, cur_type) -> {probability, count}, keyed by tuple to avoid
    #: delimiter ambiguity in category names.
    sequence_stats: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    cross_source_stats: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "fitted_through": str(self.fitted_through),
            "n_rows": self.n_rows,
            "entities": len(self.entity_baselines),
            "locations": len(self.location_stats),
            "sources": len(self.source_stats),
        }


# --------------------------------------------------------------------- fitting
def fit_state(
    train: pl.DataFrame,
    *,
    entity_column: str = "entity_id",
    value_column: str = "event_value",
    time_column: str = "timestamp",
    period: str = "train",
) -> FitState:
    """Estimate every fitted statistic from the training frame only.

    Quantiles, not just means: a mean is not robust to the injected point
    anomalies, so a mean-based baseline would be contaminated by exactly the events
    the detector is meant to find. Using the median/IQR keeps the baseline a
    description of *normal* behaviour.
    """
    if train.is_empty():
        raise ValueError("cannot fit feature state on an empty training frame")

    state = FitState(
        period=period,
        fitted_through=train[time_column].max(),
        n_rows=train.height,
    )

    # per-entity behavioural baseline: typical event rate and typical magnitude
    per_entity = train.group_by(entity_column).agg(
        pl.len().alias("n"),
        (pl.col(time_column).max() - pl.col(time_column).min()).dt.total_seconds().alias("span"),
        pl.col(value_column).median().alias("value_median"),
        pl.col(value_column).quantile(0.25).alias("value_q25"),
        pl.col(value_column).quantile(0.75).alias("value_q75"),
        pl.col(value_column).std().alias("value_std"),
    )
    span = (train[time_column].max() - train[time_column].min()).total_seconds() or 1.0
    for entity, n, ent_span, median, q25, q75, std in per_entity.iter_rows():
        state.entity_baselines[entity or "unknown"] = {
            "events_per_day": float(n) / max(1.0, float(ent_span or span)) * DAY,
            "value_median": float(median) if median is not None else 0.0,
            "value_iqr": float((q75 or 0.0) - (q25 or 0.0)),
            "value_std": float(std) if std is not None else 0.0,
            "n_events": float(n),
        }

    state.global_stats = {
        "value_median": float(train[value_column].median() or 0.0),
        "value_q25": float(train[value_column].quantile(0.25) or 0.0),
        "value_q75": float(train[value_column].quantile(0.75) or 0.0),
        "value_iqr": float(
            (train[value_column].quantile(0.75) or 0.0) - (train[value_column].quantile(0.25) or 0.0)
        ),
        "value_mean": float(train[value_column].mean() or 0.0),
        "value_std": float(train[value_column].std() or 0.0),
        "events_per_day": float(train.height) / max(1.0, span) * DAY,
    }

    for key, column in (("location_stats", "location_id"), ("source_stats", "source_id"), ("type_stats", "event_type")):
        if column not in train.columns:
            continue
        target: dict[str, dict[str, float]] = getattr(state, key)
        grouped = train.group_by(column).agg(
            pl.len().alias("n"),
            pl.col(value_column).median().alias("median"),
        )
        total = max(1, train.height)
        for name, n, median in grouped.iter_rows():
            target[str(name)] = {
                "share": float(n) / total,
                "value_median": float(median) if median is not None else 0.0,
                "n_events": float(n),
            }

    hourly = train.with_columns(pl.col(time_column).dt.hour().alias("hour")).group_by("hour").agg(
        pl.len().alias("n")
    )
    total = max(1, train.height)
    for hour, n in hourly.iter_rows():
        state.hour_stats[int(hour)] = {"share": float(n) / total, "n_events": float(n)}

    if "source_event_type" in train.columns:
        transitions = train.sort(time_column).with_columns(
            pl.col("source_event_type").shift(1).over(entity_column).alias("prev")
        )
        # Keyed by a (prev, cur) tuple, never by a delimited string. A string key
        # would be ambiguous as soon as a category contained the delimiter, and it
        # would have to be re-parsed to look up, which is where this goes wrong.
        counts: dict[tuple[str, str], float] = {}
        for prev, cur in transitions.select("prev", "source_event_type").iter_rows():
            if prev is None or cur is None:
                continue
            key = (str(prev), str(cur))
            counts[key] = counts.get(key, 0.0) + 1.0
        total_transitions = max(1.0, sum(counts.values()))
        state.sequence_stats = {
            key: {"probability": value / total_transitions, "count": value}
            for key, value in counts.items()
        }
        state.cross_source_stats = {
            "mean_transition_probability": float(
                np.mean([v["probability"] for v in state.sequence_stats.values()])
            )
            if state.sequence_stats
            else 0.0,
            "n_transitions": total_transitions,
        }
    return state


# ------------------------------------------------------------------- features
def _safe_div(numerator: pl.Expr, denominator: pl.Expr | float) -> pl.Expr:
    """Divide, guarding the denominator against zero.

    The floor matters: an entity with no baseline rate would otherwise produce
    inf, which propagates silently into the model matrix. The result is always
    float, so ``rate_*h`` is a rate in events per day rather than an integer count.
    """
    if isinstance(denominator, pl.Expr):
        guard = pl.max_horizontal(denominator.cast(pl.Float64), pl.lit(1e-9))
    else:
        guard = pl.lit(max(1e-9, float(denominator)))
    return (numerator.cast(pl.Float64) / guard).cast(pl.Float64)


def compute_features(
    events: pl.DataFrame,
    state: FitState,
    *,
    config: FeatureConfig | None = None,
    relations: pl.DataFrame | None = None,
    entity_column: str = "entity_id",
    time_column: str = "timestamp",
    value_column: str = "event_value",
) -> tuple[pl.DataFrame, FeatureRegistry, list[StageMetrics]]:
    """Compute every enabled feature family, causally.

    ``events`` must be sorted by ``(entity_id, timestamp)``; the function sorts it
    and returns the sorted frame, so downstream code cannot accidentally rely on
    the original order.
    """
    config = config or FeatureConfig()
    metrics: list[StageMetrics] = []
    registry = _registry_for(config)

    if events.is_empty():
        return events, registry, metrics

    frame = events.sort([entity_column, time_column, "event_id"], nulls_last=True)
    if entity_column not in frame.columns or frame[entity_column].null_count() == frame.height:
        raise ValueError(
            f"column {entity_column!r} is missing or entirely null; resolve entities before features"
        )

    with stage_timer("features:frequency", metrics):
        frame = _frequency_features(frame, state, config, entity_column, time_column)
    with stage_timer("features:statistical", metrics):
        frame = _statistical_features(frame, state, config, entity_column, time_column, value_column)
    with stage_timer("features:temporal", metrics):
        frame = _temporal_features(frame, state, config, entity_column, time_column)
    with stage_timer("features:behavioral", metrics):
        frame = _behavioral_features(frame, state, config, entity_column, time_column, value_column)
    with stage_timer("features:data_quality", metrics):
        frame = _data_quality_features(frame, state, config, entity_column, time_column)
    if config.include_sequence:
        with stage_timer("features:sequence", metrics):
            frame = _sequence_features(frame, state, config, entity_column, time_column)
    if config.include_cross_source:
        with stage_timer("features:cross_source", metrics):
            frame = _cross_source_features(frame, state, config, entity_column, time_column)
    if config.include_graph and relations is not None and relations.height:
        from mosaic.graph.features import add_graph_features

        with stage_timer("features:relational", metrics):
            frame = add_graph_features(frame, relations, entity_column=entity_column, time_column=time_column)

    frame = frame.sort([entity_column, time_column, "event_id"], nulls_last=True)
    present = set(frame.columns)
    feature_names = [n for n in registry.names() if n in present]
    return frame, registry, metrics


def _frequency_features(
    frame: pl.DataFrame, state: FitState, config: FeatureConfig,
    entity: str, time_col: str,
) -> pl.DataFrame:
    """Event counts in trailing windows.

    Counts use a *trailing* window, inclusive of the current row, so a rate
    detector sees the burst that is happening rather than one that already ended.
    """
    out = frame.with_columns(
        pl.len().over(entity).cast(pl.Float64).alias("entity_events_so_far")
    )
    for hours in config.rolling_hours:
        window = hours * HOUR
        out = out.with_columns(
            _trailing_count(frame, time_col, entity, window).alias(f"count_{hours}h")
        )
        out = out.with_columns(
            _safe_div(pl.col(f"count_{hours}h"), float(window) / DAY).alias(f"rate_{hours}h")
        )
    return out


def _trailing_count(
    frame: pl.DataFrame, time_col: str, entity: str, window: int
) -> pl.Expr:
    """Per-entity count of rows in the trailing ``window`` seconds, inclusive of t.

    Counts rows with ``t - window < ts <= t``. Two details are load-bearing and both
    were bugs before this version was written against a brute-force reference:

    * **Ties.** ``(entity, timestamp)`` is not unique - the real data has thousands
      of exact ties, including a near-duplicate pair the cleaner did not collapse.
      Every tied row must see *all* tied rows in its count, so the upper bound uses
      ``side="right"``. The naive ``index - left + 1`` silently gave tied rows
      different counts depending on their position in the sort.
    * **Block containment.** Entities are shifted into disjoint stride intervals so
      one ``searchsorted`` can serve the whole frame, but when a window is wider
      than an entity's own time span the lower bound can land in a *previous*
      block. Clamping it to the block start is what makes the count per-entity.

    Returns a Series-alias expression aligned to the frame's current row order.
    """
    seconds = frame[time_col].dt.epoch("s").to_numpy().astype(np.int64)
    entities = frame[entity].to_numpy()
    _, codes = np.unique(entities, return_inverse=True)
    index = np.arange(seconds.size)
    # first row index of each entity's contiguous block (frame is sorted by entity)
    first_occurrence = np.full(int(codes.max()) + 1, seconds.size, dtype=np.int64)
    first_occurrence[codes[::-1]] = index[::-1]
    block_start = first_occurrence[codes]
    span = int(seconds.max() - seconds.min()) + 1 if seconds.size else 1
    stride = np.int64(max(span, window + 1))
    composite = codes.astype(np.int64) * stride + (seconds - seconds.min())
    upper = np.searchsorted(composite, composite, side="right")
    lower = np.maximum(np.searchsorted(composite, composite - int(window), side="right"), block_start)
    counts = (upper - lower).astype(np.int64)
    return pl.Series(counts).alias(f"__trailing_{window}")


def _statistical_features(
    frame: pl.DataFrame, state: FitState, config: FeatureConfig,
    entity: str, time_col: str, value_col: str,
) -> pl.DataFrame:
    """Trailing descriptive statistics of the magnitude, causal.

    Computed from cumulative moments *within each entity* so that a row's value
    depends only on that entity's own rows at or before it. The cumulative sums are
    grouped by entity: a plain ``cum_sum()`` accumulates across the whole frame, which
    would leak entity A's magnitudes into entity B's running mean.

    The window statistics deliberately **exclude the current row**. A "is this event
    unusual for the recent past" feature must not include the event being judged, or
    a single huge magnitude inflates the mean that is supposed to flag it. The
    frequency features (which measure volume, not magnitude) do include it.
    """
    x = pl.col(value_col).cast(pl.Float64)
    out = frame.with_columns(
        x.fill_null(0.0).alias("_x"),
        (x.fill_null(0.0) ** 2).alias("_x2"),
    )
    for suffix, w in (("24h", DAY), ("7d", 7 * DAY)):
        n_incl = _trailing_count(frame, time_col, entity, w).cast(pl.Float64)
        # number of *prior* rows in the window = inclusive count minus this row
        prior_n = (n_incl - 1.0).clip(0.0)
        prior_sum_x = pl.col("_x").cum_sum().over(entity) - pl.col("_x")
        prior_sum_x2 = pl.col("_x2").cum_sum().over(entity) - pl.col("_x2")
        safe_n = pl.max_horizontal(prior_n, pl.lit(1.0))
        mean = (prior_sum_x / safe_n).alias(f"mean_value_{suffix}")
        # E[x^2] - E[x]^2, floored at 0 because the identity is only
        # algebraically non-negative and float error can push it slightly negative
        variance = (prior_sum_x2 / safe_n - mean**2).clip(0.0, None)
        out = out.with_columns(mean, variance.sqrt().alias(f"std_value_{suffix}"))
    # EWMA over this entity's own history. adjust=False gives the recursive form
    # s_t = alpha*x_t + (1-alpha)*s_{t-1}, seeded at the first observation, which is
    # causal: s_t never references a row after t.
    alpha = 2.0 / (config.windows["medium"] / HOUR + 1.0)
    ewma = pl.col("_x").ewm_mean(alpha=alpha, adjust=False).over(entity)
    out = out.with_columns(
        ewma.alias("ewma_value"),
        (pl.col("_x") - ewma).abs().alias("ewma_abs_residual"),
    )
    return out.drop("_x", "_x2")


def _temporal_features(
    frame: pl.DataFrame, state: FitState, config: FeatureConfig,
    entity: str, time_col: str,
) -> pl.DataFrame:
    """Calendar and inter-arrival features. All are functions of ``t`` alone or of
    strictly earlier events for the same entity."""
    out = frame.with_columns(
        pl.col(time_col).dt.hour().cast(pl.Float64).alias("hour_of_day"),
        pl.col(time_col).dt.weekday().cast(pl.Float64).alias("day_of_week"),
        pl.col(time_col).dt.ordinal_day().cast(pl.Float64).alias("day_of_year"),
        (pl.col(time_col).dt.hour() * 60 + pl.col(time_col).dt.minute()).cast(pl.Float64).alias("minute_of_day"),
        pl.col(time_col).dt.is_leap_year().cast(pl.Float64).alias("is_leap_year"),
    )
    # time since this entity's previous event: shift over the sorted stream, which
    # is causal by construction because the frame is sorted by (entity, time)
    out = out.with_columns(
        (pl.col(time_col) - pl.col(time_col).shift(1).over(entity))
        .dt.total_seconds()
        .alias("seconds_since_prev")
    )
    out = out.with_columns(
        pl.col("seconds_since_prev").log1p().alias("log_gap_since_prev"),
        pl.when(pl.col("seconds_since_prev") < 60)
        .then(1.0)
        .otherwise(0.0)
        .alias("is_rapid_succession"),
    )
    # gap compared with the entity's own typical gap (fitted on train)
    baselines = pl.DataFrame(
        {
            entity: list(state.entity_baselines.keys()),
            "_base_rate": [v["events_per_day"] for v in state.entity_baselines.values()],
        }
    )
    out = out.join(baselines, on=entity, how="left").with_columns(
        pl.col("_base_rate").fill_null(state.global_stats.get("events_per_day", 1.0)),
        (pl.col("seconds_since_prev") / pl.lit(DAY)).alias("gap_in_days"),
    )
    expected_gap = pl.lit(DAY) / pl.max_horizontal(pl.col("_base_rate"), pl.lit(1e-6))
    out = out.with_columns(
        (pl.col("seconds_since_prev") / expected_gap).log1p().alias("gap_vs_expected"),
        (pl.col("_base_rate") / pl.max_horizontal(
            pl.col("count_24h") / pl.lit(1.0), pl.lit(1e-6)
        )).log1p().alias("rate_24h_vs_baseline"),
    ).drop("_base_rate")
    return out


def _behavioral_features(
    frame: pl.DataFrame, state: FitState, config: FeatureConfig,
    entity: str, time_col: str, value_col: str,
) -> pl.DataFrame:
    """Deviation from the entity's *training-period* baseline.

    These are the ``FITTED`` features: the baseline is a train-period statistic, and
    applying it to a forward row is exactly the intended use. Computing the baseline
    from the row's own trailing window instead would make the detector blind to a
    *sustained* change - the definition of a behavioural anomaly - because the
    baseline would drift along with the behaviour.
    """
    baselines = pl.DataFrame(
        {
            entity: list(state.entity_baselines.keys()),
            "_base_median": [v["value_median"] for v in state.entity_baselines.values()],
            "_base_iqr": [v["value_iqr"] for v in state.entity_baselines.values()],
            "_base_rate": [v["events_per_day"] for v in state.entity_baselines.values()],
        }
    )
    global_median = state.global_stats.get("value_median", 0.0)
    global_iqr = max(1e-9, state.global_stats.get("value_iqr", 1.0))
    out = frame.join(baselines, on=entity, how="left").with_columns(
        pl.col("_base_median").fill_null(global_median),
        pl.col("_base_iqr").fill_null(global_iqr).clip(1e-9, None),
        pl.col("_base_rate").fill_null(state.global_stats.get("events_per_day", 1.0)),
    )
    value = pl.col(value_col).cast(pl.Float64)
    out = out.with_columns(
        # robust z: median/MAD rather than mean/std, so injected point anomalies do
        # not inflate the scale that is supposed to detect them
        ((value - pl.col("_base_median")) / (1.4826 * pl.col("_base_iqr")))
        .alias("value_robust_z"),
        ((value - pl.col("_base_median")).abs() / pl.col("_base_iqr")).alias("value_iqr_ratio"),
        (value > pl.col("_base_median") + 3.0 * pl.col("_base_iqr")).cast(pl.Float64).alias("value_above_3iqr"),
    )
    out = out.with_columns(
            # Parenthesised: `.alias()` binds tighter than `/`, so without these the
            # name would attach to the clipped baseline instead of the ratio.
            (pl.col("rate_24h") / pl.max_horizontal(pl.col("_base_rate"), pl.lit(1e-6))).alias("rate_vs_base"),
            ((pl.col("rate_24h") - pl.col("_base_rate")).abs()).alias("rate_abs_deviation"),
    )
    return out.drop("_base_median", "_base_iqr", "_base_rate")


def _data_quality_features(
    frame: pl.DataFrame, state: FitState, config: FeatureConfig,
    entity: str, time_col: str,
) -> pl.DataFrame:
    """How trustworthy is this row, from source-level statistics fitted on train.

    A detector should be allowed to weigh a row from a source that is behaving
    oddly, and these features are the measured basis for that decision rather than
    a hand-assigned source weight.
    """
    out = frame
    if "source_id" in frame.columns:
        shares = pl.DataFrame(
            {
                "source_id": list(state.source_stats.keys()),
                "_source_share": [v["share"] for v in state.source_stats.values()],
            }
        )
        out = out.join(shares, on="source_id", how="left").with_columns(
            # Two steps, not one: within a single with_columns call every
            # expression sees the *input* frame, so a column aliased in that same
            # call does not exist yet to be referenced by the next expression.
            pl.col("_source_share").fill_null(0.0).alias("source_share_train"),
        ).with_columns(
            (pl.col("source_share_train") <= 0.0).cast(pl.Float64).alias("source_unseen_in_train"),
        ).drop("_source_share")
    out = out.with_columns(
        (pl.col("event_value").is_null()).cast(pl.Float64).alias("value_missing"),
        (pl.col("location_id").is_null()).cast(pl.Float64).alias("location_missing"),
    )
    if "source_confidence" in frame.columns:
        out = out.with_columns(
            pl.col("source_confidence").cast(pl.Float64).fill_null(1.0).alias("record_confidence")
        )
    else:
        out = out.with_columns(pl.lit(1.0, dtype=pl.Float64).alias("record_confidence"))
    return out


def _sequence_features(
    frame: pl.DataFrame, state: FitState, config: FeatureConfig,
    entity: str, time_col: str,
) -> pl.DataFrame:
    """Event-transition features against the *training-period* transition table.

    Uses the fitted transition probabilities rather than a rolling count, so a
    repeated sequence that is globally common but rare for this entity is visible.
    """
    if "source_event_type" not in frame.columns:
        return frame
    out = frame.with_columns(
        pl.col("source_event_type").shift(1).over(entity).alias("_prev_type")
    )
    # Join the fitted transition table rather than building one when/then branch per
    # observed pair: the chain cost O(n_categories^2) expressions and was the single
    # slowest step in the pipeline. The join is O(n) and keeps the same semantics.
    table = pl.DataFrame(
        {
            "_prev_type": [k[0] for k in state.sequence_stats],
            "source_event_type": [k[1] for k in state.sequence_stats],
            "_transition_probability": [v["probability"] for v in state.sequence_stats.values()],
        },
        schema={"_prev_type": pl.Utf8, "source_event_type": pl.Utf8, "_transition_probability": pl.Float64},
    )
    # Unseen transitions and the first event of a stream (null previous type) get the
    # training-mean probability as a neutral prior rather than 0, so "unseen" is a
    # separate, explicit flag instead of being conflated with "average".
    prior = state.cross_source_stats.get("mean_transition_probability", 0.0)
    out = out.join(table, on=["_prev_type", "source_event_type"], how="left").with_columns(
        pl.col("_transition_probability").fill_null(float(prior)).alias("transition_probability"),
    ).with_columns(
        (pl.col("transition_probability") <= 0.0).cast(pl.Float64).alias("transition_unseen"),
    ).drop("_prev_type", "_transition_probability")
    return out


def _cross_source_features(
    frame: pl.DataFrame, state: FitState, config: FeatureConfig,
    entity: str, time_col: str,
) -> pl.DataFrame:
    """How many sources corroborate this entity, and how much they disagree.

    The interesting signal is *disagreement*: an event corroborated by three sources
    is not surprising, while the same event contradicted by a fourth is. Source
    support is a trailing count (causal); the agreement score uses the fitted
    per-source type distribution.
    """
    out = frame
    if "source_id" in frame.columns:
        out = out.with_columns(
            pl.col("source_id").n_unique().over([entity, pl.col(time_col).dt.truncate("1h")])
            .cast(pl.Float64)
            .alias("sources_in_hour"),
        )
        # per-entity source diversity over a trailing window
        out = out.with_columns(
            pl.col("source_id").n_unique().over(entity).cast(pl.Float64).alias("sources_for_entity")
        )
    if "source_event_type" in frame.columns:
        type_shares = pl.DataFrame(
            {
                "source_event_type": list(state.type_stats.keys()),
                "_type_share": [v["share"] for v in state.type_stats.values()],
            }
        )
        out = out.join(type_shares, on="source_event_type", how="left").with_columns(
            pl.col("_type_share").fill_null(0.0).alias("event_type_share_train"),
        ).with_columns(
            (pl.col("event_type_share_train") <= 0.0).cast(pl.Float64).alias("event_type_unseen_in_train"),
        ).drop("_type_share")
    return out


# ------------------------------------------------------------------- registry
def _registry_for(config: FeatureConfig) -> FeatureRegistry:
    """Declare the specs for the enabled families.

    Declared in the same module that computes them, so a feature cannot exist
    without a spec describing its window and leakage class.
    """
    registry = FeatureRegistry(version=config.name)

    def add(name: str, family: FeatureFamily, description: str, *, window: str = "trailing",
            columns: tuple[str, ...] = ("event_id",), fit: bool = False,
            dtype: str = "float64", cost: str = "low", notes: str = "") -> None:
        registry.register(
            FeatureSpec(
                name=name,
                family=family,
                description=description,
                dtype=dtype,
                window=window,
                source_columns=columns,
                requires_fit=fit,
                leakage=LeakageClass.FITTED if fit else LeakageClass.CAUSAL,
                cost=cost,
                notes=notes,
            )
        )

    if FeatureFamily.FREQUENCY in config.families:
        for hours in config.rolling_hours:
            add(f"count_{hours}h", FeatureFamily.FREQUENCY,
                f"events for this entity in the trailing {hours}h (inclusive of t)",
                window=f"trailing_{hours}h", columns=("timestamp",), cost="medium")
            add(f"rate_{hours}h", FeatureFamily.FREQUENCY,
                f"trailing {hours}h count normalised to events per day",
                window=f"trailing_{hours}h", columns=("timestamp",), cost="medium")
        add("entity_events_so_far", FeatureFamily.FREQUENCY,
            "index of this event within its entity's stream", columns=("timestamp",))

    if FeatureFamily.STATISTICAL in config.families:
        for suffix in ("24h", "7d"):
            add(f"mean_value_{suffix}", FeatureFamily.STATISTICAL,
                f"mean magnitude over the trailing {suffix} *excluding* the current row",
                window=f"trailing_{suffix}", columns=("event_value",), cost="medium")
            add(f"std_value_{suffix}", FeatureFamily.STATISTICAL,
                f"std of magnitude over the trailing {suffix} excluding the current row",
                window=f"trailing_{suffix}", columns=("event_value",), cost="medium")
        add("ewma_value", FeatureFamily.STATISTICAL,
            "exponentially weighted magnitude for this entity", columns=("event_value",), cost="medium")
        add("ewma_abs_residual", FeatureFamily.STATISTICAL,
            "absolute deviation of the current magnitude from its EWMA", columns=("event_value",), cost="medium")

    if FeatureFamily.TEMPORAL in config.families:
        add("hour_of_day", FeatureFamily.TEMPORAL, "hour of day (UTC)", window="point", columns=("timestamp",))
        add("day_of_week", FeatureFamily.TEMPORAL, "day of week (0=Mon)", window="point", columns=("timestamp",))
        add("day_of_year", FeatureFamily.TEMPORAL, "ordinal day of year", window="point", columns=("timestamp",))
        add("seconds_since_prev", FeatureFamily.TEMPORAL,
            "seconds since this entity's previous event", window="point",
            columns=("timestamp",), dtype="float64", cost="medium")
        add("log_gap_since_prev", FeatureFamily.TEMPORAL, "log1p of the inter-arrival gap",
            window="point", columns=("timestamp",), cost="medium")
        add("is_rapid_succession", FeatureFamily.TEMPORAL,
            "1 if the previous event was under a minute ago", window="point", columns=("timestamp",))
        add("gap_vs_expected", FeatureFamily.TEMPORAL,
            "log ratio of the observed gap to the entity's baseline gap (FITTED baseline)",
            window="point", columns=("timestamp",), fit=True, cost="medium")
        add("rate_24h_vs_baseline", FeatureFamily.TEMPORAL,
            "log ratio of the trailing 24h rate to the entity's baseline rate (FITTED)",
            window="point", columns=("timestamp",), fit=True, cost="medium")

    if FeatureFamily.BEHAVIORAL in config.families:
        add("value_robust_z", FeatureFamily.BEHAVIORAL,
            "robust z-score of magnitude vs the entity's TRAIN-period median/IQR (FITTED)",
            window="trailing", columns=("event_value",), fit=True, cost="medium",
            notes="baseline is train-only: a rolling baseline would drift with a sustained "
                   "behavioural change and hide exactly this anomaly")
        add("value_iqr_ratio", FeatureFamily.BEHAVIORAL,
            "absolute deviation from the train-period median in IQR units (FITTED)",
            window="trailing", columns=("event_value",), fit=True, cost="medium")
        add("value_above_3iqr", FeatureFamily.BEHAVIORAL,
            "1 if magnitude exceeds median + 3*IQR of the train period (FITTED)",
            window="trailing", columns=("event_value",), fit=True, cost="low")
        add("rate_vs_base", FeatureFamily.BEHAVIORAL,
            "trailing 24h rate divided by the entity's baseline rate (FITTED)",
            window="trailing", columns=("timestamp",), fit=True, cost="medium")
        add("rate_abs_deviation", FeatureFamily.BEHAVIORAL,
            "absolute difference between trailing 24h rate and the baseline rate (FITTED)",
            window="trailing", columns=("timestamp",), fit=True, cost="medium")

    if FeatureFamily.DATA_QUALITY in config.families:
        add("source_share_train", FeatureFamily.DATA_QUALITY,
            "this source's share of training rows (FITTED)", window="fit", columns=("source_id",), fit=True)
        add("source_unseen_in_train", FeatureFamily.DATA_QUALITY,
            "1 if the source contributed no training rows (FITTED)", window="fit",
            columns=("source_id",), fit=True, notes="detects a source that appears only at the end of the span")
        add("value_missing", FeatureFamily.DATA_QUALITY, "1 if the magnitude is null", window="point",
            columns=("event_value",))
        add("location_missing", FeatureFamily.DATA_QUALITY, "1 if the location is null", window="point",
            columns=("location_id",))
        add("record_confidence", FeatureFamily.DATA_QUALITY,
            "adapter-reported confidence for this record", window="point", columns=("source_confidence",))

    if FeatureFamily.SEQUENCE in config.families and config.include_sequence:
        add("transition_probability", FeatureFamily.SEQUENCE,
            "probability of this type transition under the TRAIN-period chain (FITTED)",
            window="fit", columns=("source_event_type",), fit=True, cost="medium")
        add("transition_unseen", FeatureFamily.SEQUENCE,
            "1 if the transition never occurred in training (FITTED)", window="fit",
            columns=("source_event_type",), fit=True, cost="low")

    if FeatureFamily.CROSS_SOURCE in config.families and config.include_cross_source:
        add("sources_in_hour", FeatureFamily.CROSS_SOURCE,
            "distinct sources reporting this entity in the same clock hour",
            window="current_hour", columns=("source_id",), cost="medium",
            notes="current-hour only: a trailing multi-hour window would mix in unrelated events")
        add("sources_for_entity", FeatureFamily.CROSS_SOURCE,
            "distinct sources ever seen for this entity", window="fit", columns=("source_id",), cost="medium")
        add("event_type_share_train", FeatureFamily.CROSS_SOURCE,
            "share of training rows with this event type (FITTED)", window="fit",
            columns=("event_type",), fit=True)
        add("event_type_unseen_in_train", FeatureFamily.CROSS_SOURCE,
            "1 if this event type is absent from training (FITTED)", window="fit",
            columns=("event_type",), fit=True)

    if FeatureFamily.RELATIONAL in config.families and config.include_graph:
        add("graph_degree", FeatureFamily.RELATIONAL,
            "total degree at time t, counting only edges first seen before t",
            window="causal_snapshot", columns=("related_entity_refs",), cost="high")
        add("graph_in_degree", FeatureFamily.RELATIONAL, "causal in-degree at t", window="causal_snapshot",
            columns=("related_entity_refs",), cost="high")
        add("graph_out_degree", FeatureFamily.RELATIONAL, "causal out-degree at t", window="causal_snapshot",
            columns=("related_entity_refs",), cost="high")
        add("graph_new_neighbors", FeatureFamily.RELATIONAL,
            "neighbors first seen within the trailing window", window="trailing",
            columns=("related_entity_refs",), cost="high")
        add("graph_pagerank", FeatureFamily.RELATIONAL, "PageRank of the entity in the causal snapshot",
            window="causal_snapshot", columns=("related_entity_refs",), cost="high")
        add("graph_clustering", FeatureFamily.RELATIONAL, "local clustering coefficient, causal snapshot",
            window="causal_snapshot", columns=("related_entity_refs",), cost="high")
    return registry


def assert_causal(
    features: pl.DataFrame,
    events: pl.DataFrame,
    *,
    feature: str = "count_24h",
    window_hours: int = 24,
    entity_column: str = "entity_id",
    time_column: str = "timestamp",
    sample_rows: int = 2000,
    seed: int = 0,
) -> None:
    """Independently recompute a trailing count and assert the pipeline agrees.

    This is the runtime twin of the leakage test suite: it re-derives the expected
    value straight from the event stream with a deliberately naive per-row filter -
    no vectorisation, no shared code with :func:`_trailing_count` - and compares.
    Sampling is spread across the frame rather than taken from the head, so a bug
    that only appears late in the time series is still caught.

    Raises ``AssertionError`` naming the offending row on any mismatch.
    """
    if feature not in features.columns:
        raise AssertionError(f"column {feature!r} not present in the feature frame")
    n = features.height
    if n == 0:
        return
    take = min(sample_rows, n)
    offsets = np.linspace(0, n - 1, take).astype(np.int64)
    sample = features[offsets]
    window = window_hours * HOUR
    for event_id, entity, when, value in sample.select(
        "event_id", entity_column, time_column, feature
    ).iter_rows():
        expected = events.filter(
            (pl.col(entity_column) == entity)
            & (pl.col(time_column) <= when)
            & (pl.col(time_column) > when - pl.duration(seconds=window))
        ).height
        if value is None:
            raise AssertionError(f"leakage: {feature} is null at {event_id}")
        if abs(float(value) - expected) > 1e-6:
            raise AssertionError(
                f"leakage: {feature} at {event_id} is {value} but the trailing "
                f"{window_hours}h window contains {expected} events for {entity}"
            )
