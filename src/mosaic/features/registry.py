"""Feature registry: lineage, leakage class, and cost for every feature.

A feature is not a function; it is a *contract*. The registry records, per feature:

* name, family, description, dtype, window
* source columns it reads
* whether it needs a fitted state (and therefore a training period)
* its **leakage class**, which is the important field:
  - ``CAUSAL``          - value at t uses only data at or before t. Safe everywhere.
  - ``FITTED``          - needs statistics estimated on a training period (means,
                          quantiles, encodings). Safe if fitted on train only.
  - ``SPLIT_ONLY``      - depends on which rows are in which period (labels,
                          period-relative ranks). Never used as a feature.
  - ``FORBIDDEN``       - a future-dependent quantity kept for diagnostics only.

The registry is validated at load time: a feature declared ``FITTED`` without a
``requires_fit`` flag, or declared ``CAUSAL`` while listing a future window, is a
hard error. That turns leakage prevention from a review question into a check.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from mosaic.schema.ids import stable_hash


class LeakageClass(str, Enum):
    CAUSAL = "causal"
    FITTED = "fitted"
    SPLIT_ONLY = "split_only"
    FORBIDDEN = "forbidden"


class FeatureFamily(str, Enum):
    FREQUENCY = "frequency"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    RELATIONAL = "relational"
    CROSS_SOURCE = "cross_source"
    DATA_QUALITY = "data_quality"
    SEQUENCE = "sequence"


@dataclass(frozen=True)
class FeatureSpec:
    """One feature's full declaration."""

    name: str
    family: FeatureFamily
    description: str
    dtype: str
    window: str = "point"
    source_columns: tuple[str, ...] = ()
    requires_fit: bool = False
    leakage: LeakageClass = LeakageClass.CAUSAL
    cost: str = "low"
    version: str = "1.0.0"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.leakage is LeakageClass.FITTED and not self.requires_fit:
            raise ValueError(
                f"feature {self.name!r} is FITTED but declares requires_fit=False; "
                "fitted features must name the training period they need"
            )
        if self.leakage is LeakageClass.FORBIDDEN and not self.notes:
            raise ValueError(
                f"feature {self.name!r} is FORBIDDEN and must carry a note explaining why "
                "it is kept (diagnostics only)"
            )
        if not self.source_columns and self.family is not FeatureFamily.STATISTICAL:
            raise ValueError(f"feature {self.name!r} must declare its source columns")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["family"] = self.family.value
        out["leakage"] = self.leakage.value
        out["source_columns"] = list(self.source_columns)
        return out

    def fingerprint(self) -> str:
        return stable_hash(self.to_dict(), length=12)


@dataclass
class FeatureRegistry:
    """Ordered, versioned collection of feature specs."""

    version: str
    specs: dict[str, FeatureSpec] = field(default_factory=dict)

    def register(self, spec: FeatureSpec) -> FeatureSpec:
        if spec.name in self.specs:
            raise ValueError(f"feature {spec.name!r} already registered")
        self.specs[spec.name] = spec
        return spec

    def add_all(self, specs: Iterable[FeatureSpec]) -> None:
        for spec in specs:
            self.register(spec)

    # -- queries -------------------------------------------------------------
    def names(self, families: Iterable[FeatureFamily] | None = None) -> list[str]:
        if families is None:
            return list(self.specs)
        wanted = set(families)
        return [name for name, spec in self.specs.items() if spec.family in wanted]

    def by_family(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for name, spec in self.specs.items():
            out.setdefault(spec.family.value, []).append(name)
        return {family: sorted(names) for family, names in sorted(out.items())}

    def specs_for(self, names: Iterable[str]) -> list[FeatureSpec]:
        missing = [n for n in names if n not in self.specs]
        if missing:
            raise KeyError(f"unknown features: {missing}")
        return [self.specs[n] for n in names]

    def fitted_features(self) -> list[str]:
        return [n for n, s in self.specs.items() if s.requires_fit]

    def forbidden_features(self) -> list[str]:
        return [n for n, s in self.specs.items() if s.leakage is LeakageClass.FORBIDDEN]

    def fingerprint(self) -> str:
        """Hash of the whole registry: a model result cites this."""
        return stable_hash(
            {name: spec.fingerprint() for name, spec in sorted(self.specs.items())}, length=16
        )

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fingerprint": self.fingerprint(),
            "n_features": len(self.specs),
            "by_family": self.by_family(),
            "features": {name: spec.to_dict() for name, spec in sorted(self.specs.items())},
        }

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | str) -> FeatureRegistry:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        registry = cls(version=data["version"])
        for name, spec in data["features"].items():
            registry.register(
                FeatureSpec(
                    name=spec["name"],
                    family=FeatureFamily(spec["family"]),
                    description=spec["description"],
                    dtype=spec["dtype"],
                    window=spec["window"],
                    source_columns=tuple(spec["source_columns"]),
                    requires_fit=spec["requires_fit"],
                    leakage=LeakageClass(spec["leakage"]),
                    cost=spec["cost"],
                    version=spec.get("version", "1.0.0"),
                    notes=spec.get("notes", ""),
                )
            )
        return registry

    def markdown(self) -> str:
        """Human-readable catalog for the report and the docs."""
        lines = [
            f"# Feature Catalog - {self.version}",
            "",
            f"registry fingerprint: `{self.fingerprint()}`",
            f"total features: **{len(self.specs)}**",
            "",
            "| feature | family | leakage | window | fit | cost | description |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, spec in sorted(self.specs.items()):
            lines.append(
                f"| `{name}` | {spec.family.value} | {spec.leakage.value} | {spec.window} "
                f"| {'yes' if spec.requires_fit else 'no'} | {spec.cost} | {spec.description} |"
            )
        return "\n".join(lines) + "\n"
