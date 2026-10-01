"""Narrow typing boundary for Polars reductions.

Why
---
Polars' type stubs declare ``Series.min()`` and friends as returning the *whole*
``dtype`` union - ``int | float | Decimal | date | time | timedelta | str | bytes |
ndarray | list | None`` - rather than the concrete type the column actually holds.
Every comparison or subtraction of a reduction therefore type-checks as an
unsupported operand combination, which produces hundreds of errors on code that is
in fact completely well typed.

That is a third-party stub limitation, not a defect in this codebase, and the fix
is deliberately *local*: these helpers state the concrete return type at the one
place the information Polars already knows is being discarded. Each one re-checks
the value at runtime, so a frame that does not match its annotation raises rather
than silently returning a wrong type.

This is not a global ``ignore_errors`` on Polars. Nothing outside this module is
exempt, and the runtime checks mean the annotation is enforced, not just declared.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl


def datetime_min(series: pl.Series) -> datetime:
    """The earliest non-null value of a datetime series, as a ``datetime``."""
    value = series.min()
    if not isinstance(value, datetime):
        raise TypeError(
            f"expected a datetime column, got {series.dtype!r} "
            f"(min() returned {type(value).__name__})"
        )
    return value


def datetime_max(series: pl.Series) -> datetime:
    """The latest non-null value of a datetime series, as a ``datetime``."""
    value = series.max()
    if not isinstance(value, datetime):
        raise TypeError(
            f"expected a datetime column, got {series.dtype!r} "
            f"(max() returned {type(value).__name__})"
        )
    return value


def float_mean(series: pl.Series) -> float:
    """The mean of a numeric series, as a ``float``.

    An all-null column has no mean; Polars returns ``None`` and this raises rather
    than substituting 0.0, because a silent 0.0 is indistinguishable from a real
    zero rate and would make a report claim a period had no anomalies when in fact
    nothing was known about it. Callers that legitimately have an empty or
    all-null column must decide the value themselves.
    """
    value = series.mean()
    if value is None:
        raise ValueError("cannot take the mean of an empty or all-null series")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(
            f"expected a numeric column, got {series.dtype!r} "
            f"(mean() returned {type(value).__name__})"
        )
    return float(value)


def float_sum(series: pl.Series) -> int:
    """The sum of a boolean/numeric series, as an ``int``.

    A boolean column's sum is the count of True values, which is what a positive
    count means. Same null policy as :func:`float_mean`: raises rather than
    defaulting.
    """
    value = series.sum()
    if value is None:
        raise ValueError("cannot sum an empty or all-null series")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(
            f"expected a numeric or boolean column, got {series.dtype!r} "
            f"(sum() returned {type(value).__name__})"
        )
    return int(value)


def as_any(value: object) -> Any:
    """Escape hatch for genuinely dynamic values (e.g. a Polars scalar)."""
    return value
