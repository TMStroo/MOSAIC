"""Latent container for a generated environment.

Two fields exist purely as renderer inputs and are **not** written to disk:
``stream`` (raw NumPy arrays, needed to render per-source views) and ``er`` (the
entity-resolution corruption record).

``truth`` is ground truth. No feature, training, or scoring module in MOSAIC may
import it — that separation is asserted by ``tests/leakage/test_truth_isolation.py``
and is why ``truth`` is a container field rather than a column on the events table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    from mosaic.ingestion.synthetic.generator import Stream, SyntheticSpec

EVENT_LEVEL_FAMILIES = ("point", "contextual", "relational", "cross_source", "behavioral")


@dataclass(slots=True)
class SyntheticWorld:
    """A generated environment plus its ground truth."""

    #: Latent entity population. ``entity_key`` is the hidden true identity; each
    #: source observes it through a different, lossy identifier.
    entities: pl.DataFrame
    #: Latent event stream. ``latent_id`` survives into every source ``record_id``,
    #: which makes evaluation joinable without shipping label columns.
    events: pl.DataFrame
    #: Latent ground-truth relation edges.
    relations: pl.DataFrame
    #: Injected anomalies with exact event ids, entity keys and time windows.
    truth: pl.DataFrame
    #: Ground truth for entity resolution: (source_id, source_ref) -> entity_key.
    entity_links: pl.DataFrame
    spec: SyntheticSpec
    #: Renderer-only: latent per-event arrays. Never persisted.
    stream: Stream | None = None
    #: Renderer-only: which latent entities were merged. Never persisted.
    er: dict[str, Any] | None = None

    # -- integrity helpers ---------------------------------------------------
    def truth_summary(self) -> dict[str, Any]:
        """Family counts. Reported by the generator CLI and data-quality docs.

        Never called from a feature, training, or scoring path.
        """
        if self.truth.height == 0:
            return {"total": 0, "by_family": {}, "event_level": 0}
        counts = {
            row["family"]: row["len"] for row in self.truth.group_by("family").len().to_dicts()
        }
        return {
            "total": self.truth.height,
            "by_family": dict(sorted(counts.items())),
            "event_level": int(sum(counts.get(f, 0) for f in EVENT_LEVEL_FAMILIES)),
        }

    def entity_link_summary(self) -> dict[str, Any]:
        if self.entity_links.height == 0:
            return {"total": 0, "distinct_entities": 0, "sources": [], "ambiguous_rows": 0}
        return {
            "total": self.entity_links.height,
            "distinct_entities": int(self.entity_links["entity_key"].n_unique()),
            "sources": sorted(self.entity_links["source_id"].unique().to_list()),
            "ambiguous_rows": int(self.entity_links["is_ambiguous"].sum())
            if "is_ambiguous" in self.entity_links.columns
            else 0,
        }
