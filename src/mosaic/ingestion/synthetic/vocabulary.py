"""The source-vocabulary contract, in one place.

Three subsystems need to agree on the same table:

* the **renderer** picks a native label for each latent event type,
* the **adapters** map native labels back to the canonical taxonomy,
* the **cleaning** step repairs damaged labels before mapping.

When those three hold separate copies, they drift silently: the renderer's
output stops being mappable and every row falls through to ``other`` while the
validation report still says the data looks fine. So the table lives here once,
the reverse mapping is *derived* from it, and the normalizer is a single function
used by all three.
"""

from __future__ import annotations

#: canonical event type -> the native labels a source uses for it.
#: A source may use several labels for one canonical type (transit: bus/tram).
NATIVE_CATEGORIES: dict[str, dict[str, tuple[str, ...]]] = {
    "transit_feed": {
        "movement": ("bus", "tram"),
        "access": ("tap_in",),
        "communication": ("help_point",),
        "incident": ("delay_notice",),
    },
    "ledger_api": {
        "transaction": ("card_present", "card_not_present"),
        "access": ("account_login",),
        "communication": ("support_ticket",),
        "maintenance": ("chargeback",),
    },
    "incident_logs": {
        "incident": ("outage",),
        "maintenance": ("scheduled",),
        "access": ("after_hours",),
        "communication": ("handoff",),
    },
    "sensor_net": {
        "sensor_reading": ("analog", "digital"),
        "maintenance": ("calibration",),
        "access": ("service_login",),
    },
}

#: native category -> canonical event type, per source. Derived, never hand-written.
CATEGORY_TO_EVENT_TYPE: dict[str, dict[str, str]] = {
    source_id: {native: canonical for canonical, natives in table.items() for native in natives}
    for source_id, table in NATIVE_CATEGORIES.items()
}

#: Every native label across all sources -> canonical type, for sources that are
#: not source-specific in their vocabulary.
GLOBAL_CATEGORY_MAP: dict[str, str] = {
    native: canonical
    for mapping in CATEGORY_TO_EVENT_TYPE.values()
    for native, canonical in mapping.items()
}

#: Version markers the renderer may append to simulate a schema migration.
SUFFIX_MARKERS = ("_v2", "_v3", "_legacy")


def normalize_native_category(value: str | None) -> str | None:
    """Canonical form of a native category label.

    Undoes exactly the damage the renderer inflicts and nothing else:
    surrounding whitespace, letter case, internal whitespace, and a trailing
    version marker. It deliberately does *not* try to fuzzy-match unknown labels —
    an unrecognised label must surface as a validation finding rather than be
    silently repaired into a confident (possibly wrong) category.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    for marker in SUFFIX_MARKERS:
        if text.endswith(marker):
            text = text[: -len(marker)]
            break
    return " ".join(text.split())


def canonical_event_type(value: str | None) -> str:
    """Map a (possibly damaged) native label to a canonical event type.

    Returns ``'other'`` for anything unrecognised, which the validator then counts
    as an unmapped row.
    """
    from mosaic.schema.canonical import EventType

    normalized = normalize_native_category(value)
    if normalized is None:
        return EventType.OTHER.value
    return GLOBAL_CATEGORY_MAP.get(normalized, EventType.OTHER.value)


def source_labels(source_id: str, canonical_type: str) -> tuple[str, ...]:
    """Native labels a given source uses for a canonical type."""
    return NATIVE_CATEGORIES.get(source_id, {}).get(canonical_type, ())
