"""Evaluation protocol: chronological splits and leakage guards.

This is the single most important file in MOSAIC for research validity. Every
experiment obtains its splits here, and every fitted artifact is scoped to a
split by the ``FittedOn`` container below.

The protocol
------------
Four periods, in order, cut at *observed* data boundaries (never invented dates)::

    train     earliest segment          -> fit models, scalers, ER thresholds
    validation next segment             -> select thresholds and hyperparameters
    backtest  next segment              -> evaluate a frozen configuration
    forward   final segment             -> the only truly out-of-time result

Plus optional rolling-origin evaluation: several train->validate->test origins,
which is how model stability across time is measured rather than assumed.

What "no leakage" means here, concretely
-----------------------------------------
* Rows are assigned to a period by ``timestamp`` only. There is no shuffling path.
* Any object that is *fitted* (scaler, imputer, encoder, feature-selection cut,
  entity-resolution threshold, detector threshold) carries the period it was fitted
  on, and :meth:`FittedOn.assert_compatible` refuses to use it on an earlier period
  than it was fitted on.
* Rolling features are computed causally by the feature layer, and the leakage test
  suite asserts that directly on the computed values.
* Graph features come from snapshots built from edges with ``first_seen <= t``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

import polars as pl

LOGGER = logging.getLogger(__name__)

Period = Literal["train", "validation", "backtest", "forward", "unassigned"]
PERIODS: tuple[Period, ...] = ("train", "validation", "backtest", "forward")

#: Default fraction of the observed span per period. Fractions rather than fixed
#: dates, because the actual span comes from the data and inventing dates is how
#: temporal evaluation quietly stops being out-of-time.
DEFAULT_FRACTIONS: dict[str, float] = {
    "train": 0.50,
    "validation": 0.15,
    "backtest": 0.20,
    "forward": 0.15,
}


class LeakageError(AssertionError):
    """Raised when a fitted artifact or a split would use future information."""


@dataclass(frozen=True)
class TimeBound:
    name: Period
    start: datetime | None
    end: datetime | None

    def contains(self, when: datetime) -> bool:
        if self.start is not None and when < self.start:
            return False
        if self.end is not None and when >= self.end:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.name,
            "start": str(self.start) if self.start else None,
            "end": str(self.end) if self.end else None,
        }


@dataclass
class SplitPlan:
    """Explicit, inspectable time boundaries plus the period assignment function."""

    bounds: dict[Period, TimeBound]
    observed_start: datetime
    observed_end: datetime
    fractions: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FRACTIONS))
    strategy: str = "fraction_of_observed_span"
    rolling_origins: list[dict[str, Any]] = field(default_factory=list)

    # -- assignment ----------------------------------------------------------
    def period_of(self, when: datetime) -> Period:
        for name in PERIODS:
            if self.bounds[name].contains(when):
                return name
        return "unassigned"

    def assign(self, frame: pl.DataFrame, *, timestamp_column: str = "timestamp") -> pl.DataFrame:
        """Attach a ``period`` column by timestamp. Never shuffles, never reorders."""
        if frame.is_empty():
            return frame.with_columns(pl.lit("unassigned", dtype=pl.Utf8).alias("period"))
        cuts = self.cut_points()
        expr = pl.lit("unassigned", dtype=pl.Utf8)
        # descending so the earliest non-null match wins
        for name in reversed(PERIODS):
            bound = self.bounds[name]
            conditions = []
            if bound.start is not None:
                conditions.append(pl.col(timestamp_column) >= bound.start)
            if bound.end is not None:
                conditions.append(pl.col(timestamp_column) < bound.end)
            if conditions:
                combined = conditions[0]
                for extra in conditions[1:]:
                    combined = combined & extra
                expr = pl.when(combined).then(pl.lit(name, dtype=pl.Utf8)).otherwise(expr)
        out = frame.with_columns(expr.alias("period"))
        unassigned = out.filter(pl.col("period") == "unassigned")
        if unassigned.height:
            # Only possible for rows outside [observed_start, observed_end), which
            # should be empty. Reported rather than repaired: if this fires, the
            # frame contains timestamps the split plan was not built from.
            LOGGER.warning(
                "split: %d of %d rows unassigned; observed span is %s..%s, unassigned rows span %s..%s",
                unassigned.height, frame.height,
                self.observed_start, self.observed_end,
                unassigned[timestamp_column].min(), unassigned[timestamp_column].max(),
            )
        del cuts
        return out

    def cut_points(self) -> list[datetime]:
        return [b.start for b in self.bounds.values() if b.start is not None]

    # -- reporting -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "fractions": self.fractions,
            "observed_start": str(self.observed_start),
            "observed_end": str(self.observed_end),
            "bounds": {name: bound.to_dict() for name, bound in self.bounds.items()},
            "rolling_origins": self.rolling_origins,
            "invariant": "rows are assigned by timestamp only; no shuffle path exists",
        }

    def counts(self, frame: pl.DataFrame) -> dict[str, int]:
        if "period" not in frame.columns:
            return {}
        return {
            row["period"]: row["len"]
            for row in frame.group_by("period").len().sort("period").to_dicts()
        }


def observed_range(frame: pl.DataFrame, *, timestamp_column: str = "timestamp") -> tuple[datetime, datetime]:
    """First and last observed timestamp. Raises if the data is empty."""
    if frame.is_empty():
        raise ValueError("cannot build a split plan from an empty frame")
    series = frame[timestamp_column].drop_nulls()
    if series.is_empty():
        raise ValueError(f"column {timestamp_column!r} has no non-null values")
    return series.min(), series.max()


def build_split_plan(
    frame: pl.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    fractions: dict[str, float] | None = None,
) -> SplitPlan:
    """Cut the observed span into train / validation / backtest / forward.

    The boundaries are computed from the data's own min and max, and the resulting
    bounds are returned in :attr:`SplitPlan.bounds` for inspection - the point is
    that no experiment can use a period it has not been told the dates of.
    """
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    total = sum(fractions.get(name, 0.0) for name in PERIODS)
    if not 0.5 < total <= 1.0 + 1e-9:
        raise ValueError(
            f"period fractions must sum to (0.5, 1.0], got {total:.3f} from {fractions}"
        )
    start, end = observed_range(frame, timestamp_column=timestamp_column)
    span_seconds = (end - start).total_seconds()

    bounds: dict[Period, TimeBound] = {}
    cursor = start
    for index, name in enumerate(PERIODS):
        share = fractions.get(name, 0.0)
        if index == len(PERIODS) - 1:
            bounds[name] = TimeBound(name, cursor, None)  # forward runs to the end
        else:
            next_cursor = cursor + timedelta(seconds=span_seconds * share)
            bounds[name] = TimeBound(name, cursor, next_cursor)
            cursor = next_cursor
    return SplitPlan(
        bounds=bounds,
        observed_start=start,
        observed_end=end,
        fractions=fractions,
        strategy="fraction_of_observed_span",
    )


def rolling_origins(
    frame: pl.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    n_origins: int = 3,
    train_share: float = 0.5,
    validate_share: float = 0.2,
) -> list[dict[str, Any]]:
    """Rolling-origin windows over the observed span.

    Each origin is train -> validate -> test, all strictly increasing in time. Used
    to measure *stability* across periods: one split can flatter a model by chance,
    several cannot.
    """
    start, end = observed_range(frame, timestamp_column=timestamp_column)
    span = (end - start).total_seconds()
    if n_origins < 1:
        raise ValueError("n_origins must be >= 1")
    step = span * (1.0 - (train_share + validate_share)) / max(1, n_origins)
    window = span * (train_share + validate_share)
    origins: list[dict[str, Any]] = []
    for i in range(n_origins):
            origin_start = start + timedelta(seconds=i * step)
            train_end = origin_start + timedelta(seconds=span * train_share)
            validate_end = origin_start + timedelta(seconds=window)
            if validate_end >= end:
                break
            test_end = min(end, validate_end + timedelta(seconds=step))
            origins.append(
                {
                    "origin": i,
                    "train": [str(origin_start), str(train_end)],
                    "validation": [str(train_end), str(validate_end)],
                    "test": [str(validate_end), str(test_end)],
                }
            )
    return origins


@dataclass
class FittedOn:
    """A fitted artifact, permanently stamped with the data period it saw.

    This is the mechanism that makes "fit on train, apply to forward" a type-level
    rule rather than a convention. A scaler fitted through the forward period and
    then used to standardize training features raises :class:`LeakageError` instead
    of quietly producing optimistic numbers.
    """

    name: str
    fitted_period: Period
    fitted_through: datetime
    n_rows: int
    meta: dict[str, Any] = field(default_factory=dict)

    def assert_compatible(self, target_period: Period, target_frame: pl.DataFrame | None = None) -> None:
        """Refuse to apply this artifact to a period at or before its fitting data."""
        if target_period == "unassigned":
            raise LeakageError(f"{self.name}: cannot apply to unassigned rows")
        if PERIODS.index(target_period) < PERIODS.index(self.fitted_period):
            raise LeakageError(
                f"{self.name}: fitted on {self.fitted_period} (through {self.fitted_through}) "
                f"but asked to transform {target_period}; that is backward in time"
            )
        if target_frame is not None and "timestamp" in target_frame.columns and not target_frame.is_empty():
            earliest = target_frame["timestamp"].min()
            if earliest < self.fitted_through:
                raise LeakageError(
                    f"{self.name}: fitted through {self.fitted_through} but {target_period} "
                    f"contains rows from {earliest}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fitted_period": self.fitted_period,
            "fitted_through": str(self.fitted_through),
            "n_rows": self.n_rows,
            "meta": self.meta,
        }


def assert_no_forward_rows(
    train: pl.DataFrame, evaluation: pl.DataFrame, *, timestamp_column: str = "timestamp"
) -> None:
    """Explicit invariant: evaluation rows never precede training rows."""
    if train.is_empty() or evaluation.is_empty():
        return
    if evaluation[timestamp_column].min() < train[timestamp_column].max():
        raise LeakageError(
            "evaluation period overlaps training: "
            f"evaluation starts {evaluation[timestamp_column].min()} but training runs to "
            f"{train[timestamp_column].max()}"
        )


def period_frames(frame: pl.DataFrame, period: Period) -> pl.DataFrame:
    if "period" not in frame.columns:
        raise LeakageError("frame has no 'period' column; call SplitPlan.assign first")
    return frame.filter(pl.col("period") == period)


def label_window(
    frame: pl.DataFrame, *, start: datetime, end: datetime
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split rows into (inside window, outside window) by timestamp."""
    inside = frame.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
    outside = frame.filter(~((pl.col("timestamp") >= start) & (pl.col("timestamp") < end)))
    return inside, outside


def summarize_splits(plan: SplitPlan, frame: pl.DataFrame) -> dict[str, Any]:
    """Per-period row counts, time ranges and label rates, for the report."""
    out: dict[str, Any] = {}
    for name in PERIODS:
        part = period_frames(frame, name)
        entry: dict[str, Any] = {"rows": part.height}
        if part.height:
            entry["start"] = str(part["timestamp"].min())
            entry["end"] = str(part["timestamp"].max())
            entry["days"] = round(
                (part["timestamp"].max() - part["timestamp"].min()).total_seconds() / 86400, 2
            )
            if "is_anomaly" in part.columns:
                rate = float(part["is_anomaly"].mean() or 0.0)
                entry["anomaly_rate"] = round(rate, 6)
                entry["positives"] = int(part["is_anomaly"].sum())
        out[name] = entry
    out["_plan"] = plan.to_dict()
    return out


def periods_in_order() -> Sequence[Period]:
    return PERIODS
