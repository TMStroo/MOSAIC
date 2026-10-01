"""Validation subsystem: structured, countable, non-destructive checks.

Every check returns a :class:`CheckResult` rather than raising or silently
dropping rows. A dataset that is 40% malformed is a research finding, not an
error to be swallowed — so the report records it, and cleaning decides what to do
under an explicit, configured policy.

Status vocabulary is fixed: ``PASS`` / ``WARN`` / ``FAIL``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import polars as pl

#: Status ordering used when collapsing a report into a single verdict.
SEVERITY_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


class Status(str, enum.Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    """One validation finding, in the shape the report and API both consume."""

    source_id: str
    check: str
    expected: Any
    observed: Any
    status: Status
    severity: str = "error"
    details: dict[str, Any] = field(default_factory=dict)
    rows_affected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "check": self.check,
            "expected": self.expected,
            "observed": self.observed,
            "status": self.status.value,
            "severity": self.severity,
            "rows_affected": self.rows_affected,
            "details": self.details,
        }


class ValidationReport:
    """Accumulator for a whole dataset's checks."""

    def __init__(self, dataset: str, dataset_version: str) -> None:
        self.dataset = dataset
        self.dataset_version = dataset_version
        self.results: list[CheckResult] = []

    def add(
        self,
        source_id: str,
        check: str,
        *,
        expected: Any,
        observed: Any,
        status: Status,
        severity: str = "error",
        rows_affected: int = 0,
        **details: Any,
    ) -> CheckResult:
        result = CheckResult(
            source_id=source_id,
            check=check,
            expected=expected,
            observed=observed,
            status=status,
            severity=severity,
            rows_affected=rows_affected,
            details=details,
        )
        self.results.append(result)
        return result

    def extend(self, results: list[CheckResult]) -> None:
        self.results.extend(results)

    # -- aggregation ---------------------------------------------------------
    def by_status(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Status}
        for result in self.results:
            counts[result.status.value] += 1
        return counts

    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.FAIL]

    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status is Status.WARN]

    def worst_status(self) -> str:
        return max((r.status.value for r in self.results), key=lambda s: SEVERITY_ORDER[s], default="PASS")

    def quality_score(self) -> float:
        """A 0-1 summary score for the overview page.

        Deliberately simple and documented: 1 - (weighted failing+warning checks) /
        total, with WARN worth a quarter of a FAIL. It is a *display* number; the
        report also lists every individual check, and no research claim depends on it.
        """
        if not self.results:
            return 1.0
        penalty = sum(1.0 if r.status is Status.FAIL else 0.25 if r.status is Status.WARN else 0.0 for r in self.results)
        return round(max(0.0, 1.0 - penalty / len(self.results)), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "worst_status": self.worst_status(),
            "quality_score": self.quality_score(),
            "counts": self.by_status(),
            "results": [r.to_dict() for r in self.results],
        }

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "source_id": r.source_id,
                    "check": r.check,
                    "expected": str(r.expected),
                    "observed": str(r.observed),
                    "status": r.status.value,
                    "severity": r.severity,
                    "rows_affected": r.rows_affected,
                    "details": repr(r.details),
                }
                for r in self.results
            ],
            schema={
                "source_id": pl.Utf8,
                "check": pl.Utf8,
                "expected": pl.Utf8,
                "observed": pl.Utf8,
                "status": pl.Utf8,
                "severity": pl.Utf8,
                "rows_affected": pl.Int64,
                "details": pl.Utf8,
            },
        )
