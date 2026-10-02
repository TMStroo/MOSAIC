"""Pydantic models for the MOSAIC canonical representation.

Design rules enforced here:

1. Every canonical record keeps source lineage (``source_id`` + ``source_record_id``
   + ``raw_reference``). Lineage is never dropped, only extended.
2. ``event_metadata`` is the single escape hatch for source-specific fields. Source
   adapters map their native columns into it instead of widening the core schema.
3. Timestamps are timezone-naive UTC. Adapters are responsible for converting;
   the canonical layer refuses ambiguous or offset-carrying input (see
   ``mosaic.cleaning.timestamps``).
4. Ground-truth anomaly labels live in a *separate* contract (``AnomalyLabel``)
   that is never joined into feature-generation code paths.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

Epoch = Annotated[datetime, Field(description="Timezone-naive UTC timestamp")]


class StrictModel(BaseModel):
    """Base for canonical contracts: immutable, extra-forbidding, validated."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


class EventType(enum.StrEnum):
    """Canonical event taxonomy.

    Source-specific event types are preserved in ``event_metadata['source_event_type']``
    and mapped here with a configurable, per-source dictionary.
    """

    ACCESS = "access"
    TRANSACTION = "transaction"
    MOVEMENT = "movement"
    SENSOR_READING = "sensor_reading"
    COMMUNICATION = "communication"
    INCIDENT = "incident"
    MAINTENANCE = "maintenance"
    OTHER = "other"


class AnomalyFamily(enum.StrEnum):
    """Anomaly taxonomy used by the synthetic generator and by evaluation.

    The same enum is used for injected (synthetic) and natural (public-dataset)
    labels; ``AnomalyLabel.injected`` distinguishes them.
    """

    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    RELATIONAL = "relational"
    CROSS_SOURCE = "cross_source"
    DISTRIBUTION = "distribution"
    MISSINGNESS = "missingness"
    ENTITY_RESOLUTION = "entity_resolution"
    NATURAL = "natural"


class SourceMeta(StrictModel):
    """Provenance and quality metadata for one ingested source."""

    source_id: str = Field(description="Stable source key, e.g. 'transit_feed'")
    source_name: str
    source_version: str = "unknown"
    downloaded_at: datetime | None = None
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    schema_hash: str = Field(description="sha256 of the sorted column->dtype map")
    data_checksum: str = Field(description="sha256 over canonical row content")
    time_range: tuple[datetime, datetime] | None = None
    known_limitations: list[str] = Field(default_factory=list)


class RawRecordMeta(StrictModel):
    """Lineage stub stored with every canonical event.

    ``raw_reference`` is a *logical* pointer (dataset + partition + row ordinal),
    never a filesystem path: the API must not be able to leak raw paths.
    """

    source_id: str
    source_record_id: str
    dataset_version: str
    raw_reference: str
    transform_version: str


class CanonicalEvent(StrictModel):
    """The normalized event record every MOSAIC subsystem consumes.

    Field-level notes that matter for research integrity:

    * ``timestamp`` is the *event* time used for every temporal feature and split.
    * ``ingestion_time`` is when MOSAIC saw the record. It is a data-quality
      feature, never a training-time ordering key, because it is unavailable
      for forward simulation.
    * ``source_confidence`` in [0, 1] is the adapter's own reliability estimate.
    * ``related_entity_ids`` carries cross-source references before entity
      resolution, and canonical ``entity_id`` values after it.
    """

    event_id: str
    source_id: str
    source_record_id: str
    timestamp: Epoch
    ingestion_time: Epoch
    entity_id: str | None = None
    entity_type: str | None = None
    event_type: EventType = EventType.OTHER
    location_id: str | None = None
    source_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    event_value: float | None = None
    parent_event_id: str | None = None
    related_entity_ids: tuple[str, ...] = ()
    event_metadata: dict[str, Any] = Field(default_factory=dict)
    raw_reference: str | None = None
    dataset_version: str = "unknown"

    @field_validator("timestamp", "ingestion_time")
    @classmethod
    def _reject_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is not None:
            raise ValueError("canonical timestamps must be timezone-naive UTC")
        return value


class Relation(StrictModel):
    """A directed, weighted, time-bounded edge between canonical entities."""

    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    first_seen: Epoch
    last_seen: Epoch
    weight: float = Field(default=1.0, ge=0.0)
    event_ids: tuple[str, ...] = ()
    evidence_sources: tuple[str, ...] = ()
    resolution_method: str = "observed"


class EntityMatch(StrictModel):
    """One entity-resolution decision, with the evidence that produced it."""

    left_record_id: str
    right_record_id: str
    left_source_id: str
    right_source_id: str
    match_probability: float = Field(ge=0.0, le=1.0)
    matching_method: str
    evidence_fields: dict[str, Any] = Field(default_factory=dict)
    decision: str = Field(description="'match' | 'non_match' | 'possible_match'")
    decision_threshold: float
    blocking_key: str | None = None


class AnomalyLabel(StrictModel):
    """Ground truth. Held out of every feature path by design.

    ``target_ref`` is an event id, an entity id, or a ``(entity_id, time_range)``
    pair, depending on the family. ``event_ids`` is populated for all families so
    detection delay can be measured against the earliest labelled event.
    """

    anomaly_id: str
    family: AnomalyFamily
    target_ref: str
    event_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    started_at: Epoch
    ended_at: Epoch | None = None
    severity: float = Field(default=1.0, ge=0.0, le=1.0)
    injected: bool = True
    notes: str = ""
