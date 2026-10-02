"""Controlled feature sets F0-F5 for the M6 model comparison.

The point of naming them is that "which features did the model see?" must be an
answerable question about a result artifact, not a guess. Each set is a
cumulative subset of the families, and each set's identity is a stable hash, so
two runs citing the same ``feature_set_id`` really did see the same columns in
the same order.

``RELATIONAL`` is the graph-backed family (degree, PageRank, clustering, new
neighbours). It is excluded from F0-F5 by construction: M6 Phase 3 must first
establish the *non-graph* baseline, and ``FeatureConfig(include_graph=False)``
already omits it. Graph features are a later, separate question.
"""

from __future__ import annotations

from dataclasses import dataclass

from mosaic.features.registry import FeatureFamily
from mosaic.schema.ids import stable_hash

#: The family the graph feature builder owns.
GRAPH_FAMILY = FeatureFamily.RELATIONAL

#: Cumulative family sets. ``F5`` is every non-graph family the registry offers.
CUMULATIVE: dict[str, tuple[FeatureFamily, ...]] = {
    "F0": (FeatureFamily.FREQUENCY,),
    "F1": (FeatureFamily.FREQUENCY, FeatureFamily.STATISTICAL),
    "F2": (
        FeatureFamily.FREQUENCY,
        FeatureFamily.STATISTICAL,
        FeatureFamily.TEMPORAL,
    ),
    "F3": (
        FeatureFamily.FREQUENCY,
        FeatureFamily.STATISTICAL,
        FeatureFamily.TEMPORAL,
        FeatureFamily.BEHAVIORAL,
    ),
    "F4": (
        FeatureFamily.FREQUENCY,
        FeatureFamily.STATISTICAL,
        FeatureFamily.TEMPORAL,
        FeatureFamily.BEHAVIORAL,
        FeatureFamily.CROSS_SOURCE,
    ),
    "F5": (
        FeatureFamily.FREQUENCY,
        FeatureFamily.STATISTICAL,
        FeatureFamily.TEMPORAL,
        FeatureFamily.BEHAVIORAL,
        FeatureFamily.CROSS_SOURCE,
        FeatureFamily.DATA_QUALITY,
        FeatureFamily.SEQUENCE,
    ),
}


@dataclass(frozen=True)
class FeatureSet:
    """A named, hashed, family-scoped set of feature columns."""

    name: str
    families: tuple[FeatureFamily, ...]
    columns: tuple[str, ...]

    @property
    def id(self) -> str:
        """Stable identity: the family list plus the resolved column names.

        Two runs with the same id used the same families *and* the same concrete
        columns, so a feature-set id in an artifact is auditable.
        """
        return f"{self.name}_{stable_hash([f.value for f in self.families] + list(self.columns), length=10)}"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "id": self.id,
            "families": [f.value for f in self.families],
            "n_columns": len(self.columns),
            "columns": list(self.columns),
            "excludes_graph": GRAPH_FAMILY not in self.families,
        }


def build_feature_sets(registry, available: list[str]) -> dict[str, FeatureSet]:
    """Resolve the family sets against what the registry actually produced.

    ``available`` is the list of feature columns present in the computed frame,
    so a set can never claim a column the pipeline did not emit. A set that
    resolves to nothing raises, because a silently empty feature set would make
    a model look broken for a reason that has nothing to do with the model.
    """
    usable = set(available)
    out: dict[str, FeatureSet] = {}
    for name, families in CUMULATIVE.items():
        if GRAPH_FAMILY in families:  # pragma: no cover - defensive
            raise ValueError(f"{name} must not include the graph family")
        columns = tuple(n for n in registry.names(families) if n in usable)
        if not columns:
            raise ValueError(f"feature set {name} resolved to zero columns")
        out[name] = FeatureSet(name=name, families=families, columns=columns)
    return out


def single_family_set(name: str, family: FeatureFamily, registry, available: list[str]) -> FeatureSet:
    """One family in isolation, for E013's per-family comparison."""
    usable = set(available)
    columns = tuple(n for n in registry.names([family]) if n in usable)
    if not columns:
        raise ValueError(f"feature set {name} resolved to zero columns")
    return FeatureSet(name=name, families=(family,), columns=columns)
