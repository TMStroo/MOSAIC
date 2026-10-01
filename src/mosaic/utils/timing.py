"""Timing + memory instrumentation. Every research stage records both.

Peak RSS is read from ``resource`` where available (POSIX), otherwise from
``psutil`` if installed, otherwise the field is reported as ``None`` rather than
guessed. Hardware capture lives in :mod:`mosaic.utils.provenance`.
"""

from __future__ import annotations

import os
import time
import tracemalloc
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


def peak_rss_mb() -> float | None:
    """Peak resident set size in MiB, or ``None`` if the platform won't say."""
    try:  # POSIX
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(usage / 1024, 2)  # KiB -> MiB on Linux
    except Exception:  # pragma: no cover - Windows path
        try:
            import psutil  # type: ignore[import-not-found]

            return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
        except Exception:
            return None


def current_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return None


@dataclass
class StageMetrics:
    """Measured cost of one pipeline stage."""

    stage: str
    duration_s: float
    rows: int | None = None
    peak_rss_mb: float | None = None
    python_alloc_mb: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "duration_s": round(self.duration_s, 6),
            "rows": self.rows,
            "peak_rss_mb": self.peak_rss_mb,
            "python_alloc_mb": self.python_alloc_mb,
            **({"extra": self.extra} if self.extra else {}),
        }


class Stopwatch:
    """Accumulating timer usable as a context manager or manually.

    >>> sw = Stopwatch("features")
    >>> with sw:
    ...     pass
    >>> sw.metrics.duration_s >= 0
    True
    """

    def __init__(self, stage: str, *, track_python_alloc: bool = False) -> None:
        self.stage = stage
        self.track_python_alloc = track_python_alloc
        self.metrics: StageMetrics | None = None
        self._start = 0.0
        self._alloc_start = 0

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        if self.track_python_alloc:
            tracemalloc.start()
            self._alloc_start = tracemalloc.get_traced_memory()[1]
        return self

    def __exit__(self, *exc: object) -> None:
        duration = time.perf_counter() - self._start
        alloc = None
        if self.track_python_alloc:
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            alloc = round((peak - self._alloc_start) / (1024 * 1024), 4)
        self.metrics = StageMetrics(
            stage=self.stage,
            duration_s=duration,
            peak_rss_mb=peak_rss_mb(),
            python_alloc_mb=alloc,
        )
        if self.metrics.extra:
            pass

    def set_rows(self, rows: int) -> None:
        if self.metrics is not None:
            self.metrics.rows = rows


@contextmanager
def stage_timer(stage: str, sink: list[StageMetrics] | None = None) -> Iterator[Stopwatch]:
    """Time a block and optionally append its :class:`StageMetrics` to ``sink``."""
    sw = Stopwatch(stage)
    with sw:
        yield sw
    assert sw.metrics is not None
    if sink is not None:
        sink.append(sw.metrics)


def cpu_count() -> int:
    return os.cpu_count() or 1
