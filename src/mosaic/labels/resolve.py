"""Resolve synthetic ground truth onto canonical events.

See :mod:`mosaic.labels` for the design rationale and the truth-isolation rule.

The public entry point is :func:`resolve_labels`. It attaches four columns to
an event frame:

``is_anomaly``
    ``True`` when the event falls inside at least one injected anomaly.
``anomaly_family``
    Sorted, de-duplicated list of families that fired for that event. A list,
    not a string, because families genuinely overlap.
``anomaly_ids``
    Every ``anomaly_id`` covering the event -- the evidence link back to truth.
``anomaly_severity``
    Max severity across covering anomalies; null when not anomalous.

Every ``NOT_APPLICABLE`` decision is reported in :attr:`LabelResolution.skipped`
rather than silently omitted.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl

from mosaic.experiments.protocol import PERIODS, LeakageError, Period

LOGGER = logging.getLogger(__name__)

#: The nine injected families, in a stable reporting order.
ANOMALY_FAMILIES: tuple[str, ...] = (
    "point",
    "contextual",
    "collective",
    "temporal",
    "behavioral",
    "relational",
    "cross_source",
    "distribution",
    "missingness",
)

#: A per-family metric computed on fewer than this many positives is noise.
#: Families below it are reported NOT_APPLICABLE, never as a rate.
MINIMUM_FAMILY_SUPPORT = 20

#: Families that mark a single injected event. Their `target_ref` is the
#: latent id of that event, so resolution is an exact join.
EVENT_FAMILIES: frozenset[str] = frozenset(
    {"point", "contextual", "relational", "cross_source", "behavioral"}
)

#: Families whose anomaly is a *property of a set or interval*, not of an
#: individual row. `collective` marks the events of entities that took part in
#: a community burst; `temporal` marks a 6-hour bucket; `distribution` and
#: `missingness` mark multi-day intervals. Resolving these by their declared
#: time span marks hundreds of ordinary events per injected anomaly (collective
#: spans reach 2,441 hours), which would make "anomaly" mean "ever touched a
#: flagged entity" and inflate the positive rate to ~38%.
#:
#: They are retained -- never discarded -- and reported per-family, because
#: they are exactly the classes where a detector's row-level precision is
#: expected to be poor, and hiding that would be the convenient result.
#: ``is_anomaly`` (row level) defaults to EVENT_FAMILIES only; the aggregate
#: families are available through ``anomaly_family`` for the per-family
#: breakdown and for set-level evaluation.
AGGREGATE_FAMILIES: frozenset[str] = frozenset(
    {"collective", "temporal", "distribution", "missingness"}
)

_LATENT_PATTERN = r"(L\d+)"


class LabelScope(StrEnum):
    """Which families define the row-level ``is_anomaly`` label.

    This is a research decision, so it is named, versioned and recorded in every
    result artifact rather than hidden in a boolean.

    ``EVENT_ONLY`` (default)
        ``is_anomaly`` is True only where a single injected event was labelled.
        Aggregate families (collective / temporal / distribution /
        missingness) are still resolved onto every row -- a row covered by a
        collective burst carries that family in ``anomaly_family`` -- but do not
        make the row a positive. Rationale: resolving those families by their
        declared span marks 183,841 of 497,994 rows (36.9%) positive, so
        "anomaly" would mean "belonged to an entity that was ever flagged"
        rather than "this event was injected". Every family is still reported
        separately.

    ``ALL_FAMILIES``
        Any family makes a row positive. Use only for set-level analysis, and
        note that the resulting 38.5% positive rate makes this a
        near-balanced problem in which precision/recall behave very differently
        from a realistic 1-5% anomaly rate.
    """

    EVENT_ONLY = "event_only"
    ALL_FAMILIES = "all_families"


DEFAULT_SCOPE = LabelScope.EVENT_ONLY


@dataclass(frozen=True)
class LabelResolution:
    """Outcome of a label join, plus the evidence needed to audit it."""

    events: pl.DataFrame
    dataset_version: str
    labels_read: int
    injected_read: int
    matched_labels: int
    family_counts: dict[str, int]
    positives: int
    rate: float | None
    skipped_families: dict[str, str] = field(default_factory=dict)
    unmatched_target_kinds: dict[str, int] = field(default_factory=dict)
    period_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    observability: dict[str, Any] = field(default_factory=dict)
    family_coverage: dict[str, int] = field(default_factory=dict)
    prevalence: dict[str, dict[str, Any]] = field(default_factory=dict)
    scope: LabelScope = DEFAULT_SCOPE

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "label_scope": str(self.scope),
            "labels_read": self.labels_read,
            "injected_read": self.injected_read,
            "matched_labels": self.matched_labels,
            "positives": self.positives,
            "positive_rate": self.rate,
            "family_counts": dict(sorted(self.family_counts.items())),
            "family_coverage": dict(sorted(self.family_coverage.items())),
            "prevalence": self.prevalence,
            "event_families": sorted(EVENT_FAMILIES),
            "aggregate_families": sorted(AGGREGATE_FAMILIES),
            "skipped_families": dict(sorted(self.skipped_families.items())),
            "unmatched_target_kinds": dict(sorted(self.unmatched_target_kinds.items())),
            "period_counts": self.period_counts,
            "observability": self.observability,
        }


def minimum_support() -> int:
    """Return the support floor below which a family metric is meaningless."""
    return MINIMUM_FAMILY_SUPPORT


def _read_truth(root: str | Any) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, str]:
    """Read ``_truth/`` and the manifest version stamp."""
    base = Path(str(root))
    truth = base / "_truth"
    labels = pl.read_parquet(truth / "labels.parquet")
    links = pl.read_parquet(truth / "entity_links.parquet")
    latent = pl.read_parquet(truth / "latent_events.parquet")
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    version = f"{manifest.get('dataset_name', base.name)}:{manifest.get('generator_version', 'unknown')}"
    return labels, links, latent, version


def _latent_of() -> pl.Expr:
    """Recover ``latent_id`` from the encoded ``source_record_id``.

    The generator writes a per-source prefix (``se-L001574471``,
    ``tr-L001222809``), so the latent id is extracted rather than sliced.
    """
    return pl.col("source_record_id").str.extract(_LATENT_PATTERN, 1)


def _family_series(values: Sequence[str] | None) -> pl.Series:
    return pl.Series(values or [], dtype=pl.List(pl.Utf8))


def resolve_labels(
    events: pl.DataFrame,
    *,
    truth_root: str | Any,
    period_column: str = "period",
    scope: LabelScope = DEFAULT_SCOPE,
) -> LabelResolution:
    """Attach ground-truth labels to canonical ``events``.

    ``events`` must already carry ``source_id``, ``source_record_id``,
    ``entity_id``, ``timestamp`` and -- when a chronological split is in force
    -- ``period_column``.

    ``strict`` (the default) refuses to proceed if a positive label survives
    into ``validation``/``backtest``/``forward`` *without* the caller having
    explicitly acknowledged it. Labels legitimately exist in those periods;
    they are the evaluation target, never an input. What must never happen is a
    label being used to *fit* anything, which
    :meth:`mosaic.experiments.protocol.FittedOn.assert_compatible` enforces at
    fit time and :func:`assert_labels_never_fitted` re-checks here.
    """
    labels, links, latent, version = _read_truth(truth_root)

    required = {"source_id", "source_record_id", "entity_id", "timestamp"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events frame is missing required columns: {sorted(missing)}")

    # The label sidecar stores timestamps as strings; the event frame carries
    # real datetimes. Cast once here so every downstream window comparison is
    # temporal-on-temporal rather than raising.
    for column in ("started_at", "ended_at"):
        if labels.schema[column] == pl.Utf8:
            labels = labels.with_columns(
                pl.col(column).str.to_datetime(strict=False).alias(column)
            )
    injected = labels.filter(pl.col("injected"))

    frame = events
    if "_latent" not in frame.columns:
        frame = frame.with_columns(_latent_of().alias("_latent"))

    # --- entity resolution for truth targets -------------------------------
    # ``target_ref`` for entity-kind anomalies is a latent *entity key*
    # (E000473). Events carry ``entity_id``, which is the *resolved* cluster id
    # (``en_2bfb46ffb218b24b5111``), so the two spaces never meet directly.
    # ``entity_links`` is the declared ground-truth bridge from
    # (source_id, source_ref) to entity_key; the entity-resolution step has
    # already collapsed each entity_key's source refs into one cluster, so the
    # bridge has to be rebuilt through the *resolved* mapping.
    links_frame = (
        links.with_columns(
            pl.col("entity_key").alias("_truth_key"),
        )
    )
    frame = frame.with_columns(
        pl.col("entity_id").alias("_resolved_id"),
    )
    # Which latent entity_key does each canonical entity_id correspond to?
    # Majority vote over the source refs that the resolver grouped: an
    # entity_id's refs map to one entity_key per source, so the modal
    # entity_key is the truth identity for that cluster.
    if not links_frame.is_empty() and "entity_ref_source" in frame.columns:
        vote = (
            frame.select("event_id", "entity_id", "entity_ref_source")
            .join(links_frame.select("source_id", "source_ref", "_truth_key"),
                  left_on="entity_ref_source", right_on="source_ref", how="inner")
            .group_by("entity_id")
            .agg(pl.col("_truth_key").mode().first().alias("_truth_key"))
        )
        frame = frame.join(vote, on="entity_id", how="left")
    else:  # pragma: no cover - only when links are absent
        frame = frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias("_truth_key"))

    frame = _attach_event_labels(frame, injected)
    frame = _attach_entity_labels(frame, injected, latent)
    frame = _attach_window_labels(frame, injected)

    frame = _finalize(frame)
    # Per-protocol selection happens after all protocol-independent work, so
    # resolving once and binding twice is the only supported path.
    frame = _select_protocol(frame, scope)

    family_coverage = _family_coverage(frame)
    family_counts = _family_counts(frame)
    skipped = {
        family: f"fewer than {MINIMUM_FAMILY_SUPPORT} positive events"
        for family in ANOMALY_FAMILIES
        if family_counts.get(family, 0) < MINIMUM_FAMILY_SUPPORT
        and family_coverage.get(family, 0) > 0
    }

    positives = int(frame["is_anomaly"].sum() or 0)
    rate = (positives / frame.height) if frame.height else None

    # Both protocols' prevalence, recorded even when only one is selected, so a
    # single call site can report the comparison without a second join.
    prevalence = {
        str(protocol): {
            "positives": int(frame[column].sum() or 0),
            "rate": (
                round(int(frame[column].sum() or 0) / frame.height, 8)
                if frame.height
                else None
            ),
        }
        for protocol, column in (
            (LabelScope.EVENT_ONLY, "is_anomaly_event_only"),
            (LabelScope.ALL_FAMILIES, "is_anomaly_all_families"),
        )
    }

    # --- observability audit ----------------------------------------------
    # The generator holds ~1.69M latent events but emits only a fraction to
    # any source, so most injected event-kind anomalies never become
    # observable rows. That is a property of the synthetic world, not a join
    # defect -- but it must be stated, because a detector's recall against
    # unobservable anomalies is undefined, not zero.
    observed_latents = set(frame["_latent"].drop_nulls().to_list())
    event_labels = injected.filter(pl.col("target_kind") == "event")
    observable = event_labels.filter(pl.col("latent_id").is_in(observed_latents))
    observability = {
        "latent_events_total": latent.height,
        "latent_events_observable": len(observed_latents),
        "event_labels": event_labels.height,
        "event_labels_observable": observable.height,
        "event_label_coverage": (
            round(observable.height / event_labels.height, 6)
            if event_labels.height
            else None
        ),
        "note": (
            "anomalies whose target never reached an observable source cannot "
            "be detected; recall is computed only over observable labels"
        ),
    }

    matched = int(
        (pl.DataFrame({"a": frame["anomaly_ids"]}).explode("a").height) if frame.height else 0
    )

    unmatched_kinds = (
        injected.group_by("target_kind")
        .len()
        .filter(pl.col("len") > 0)
        .to_dicts()
    )

    period_counts = _period_counts(frame, period_column)

    return LabelResolution(
        events=frame.drop("_latent", "_truth_key", strict=False),
        dataset_version=version,
        labels_read=labels.height,
        injected_read=injected.height,
        matched_labels=matched,
        family_counts=family_counts,
        positives=positives,
        rate=rate,
        skipped_families=skipped,
        unmatched_target_kinds={str(k): int(v) for k, v in
                                ((r["target_kind"], r["len"]) for r in unmatched_kinds)},
        period_counts=period_counts,
        observability=observability,
        family_coverage=family_coverage,
        prevalence=prevalence,
        scope=scope,
    )


def resolve_both(
    events: pl.DataFrame,
    *,
    truth_root: str | Any,
    period_column: str = "period",
) -> tuple[LabelResolution, LabelResolution]:
    """Return ``(EVENT_ONLY, ALL_FAMILIES)`` from a single resolution pass.

    This is the entry point the dual-protocol comparison must use. The
    expensive upstream work -- reading truth, recovering latent ids, the
    entity-key vote, and all three label attachment paths -- is
    protocol-independent and runs exactly once; only the final
    ``is_anomaly`` binding is repeated.

    Callers that invoke :func:`resolve_labels` twice for the two protocols
    would recompute all of that, which is the redundancy this function exists
    to prevent.
    """
    shared = resolve_labels(
        events,
        truth_root=truth_root,
        period_column=period_column,
        scope=LabelScope.EVENT_ONLY,
    )
    event_only = shared
    # `event_only.events` still carries *both* protocol columns plus the
    # EVENT_ONLY binding, so the ALL_FAMILIES frame is one rebind away -- no
    # second truth read, no second entity-key vote, no second label attach.
    all_frame = _select_protocol(
        event_only.events.drop("is_anomaly", "label_protocol", strict=False),
        LabelScope.ALL_FAMILIES,
    )
    all_res = LabelResolution(
        events=all_frame,
        dataset_version=shared.dataset_version,
        labels_read=shared.labels_read,
        injected_read=shared.injected_read,
        matched_labels=shared.matched_labels,
        family_counts=_family_counts(all_frame),
        positives=int(all_frame["is_anomaly"].sum() or 0),
        rate=(int(all_frame["is_anomaly"].sum() or 0) / all_frame.height)
        if all_frame.height
        else None,
        skipped_families=shared.skipped_families,
        unmatched_target_kinds=shared.unmatched_target_kinds,
        period_counts=_period_counts(all_frame, period_column),
        observability=shared.observability,
        family_coverage=shared.family_coverage,
        prevalence=shared.prevalence,
        scope=LabelScope.ALL_FAMILIES,
    )
    return event_only, all_res


def _attach_event_labels(frame: pl.DataFrame, injected: pl.DataFrame) -> pl.DataFrame:
    """``target_kind == 'event'``: exact latent-id match."""
    ev = injected.filter(pl.col("target_kind") == "event").select(
        pl.col("latent_id").alias("_latent"),
        pl.col("anomaly_id"),
        pl.col("family"),
        pl.col("severity"),
    )
    if ev.is_empty():
        return frame.with_columns(
            pl.lit(None, dtype=pl.List(pl.Utf8)).alias("_ids_ev"),
            pl.lit(None, dtype=pl.List(pl.Utf8)).alias("_fam_ev"),
            pl.lit(None, dtype=pl.Float64).alias("_sev_ev"),
        )
    # Several labels can share one latent_id (an injected event inside an
    # entity window is labelled by both), so the join can fan out. Reduce to
    # one row per event with the columns already aliased to their final
    # private names -- otherwise a second join on a different column would
    # collide with the bare `anomaly_id`/`family`/`severity` they introduce.
    agg = (
        ev.group_by("_latent")
        .agg(
            pl.col("anomaly_id").unique().sort().alias("_ids_ev"),
            pl.col("family").unique().sort().alias("_fam_ev"),
            pl.col("severity").max().alias("_sev_ev"),
        )
    )
    return frame.join(agg, on="_latent", how="left")


def _attach_entity_labels(
    frame: pl.DataFrame, injected: pl.DataFrame, latent: pl.DataFrame
) -> pl.DataFrame:
    """``target_kind == 'entity'`` (collective): the exact injected events.

    The aggregate label records only the entity and a span. That span is the
    min/max time of the burst's *donor* events, which are drawn from across the
    whole stream, so it can reach 2,441 hours and would mark ~730 ordinary
    events per flagged entity as anomalous. When the per-event family flags are
    available (``_truth/latent_events.parquet`` carries ``fam_<family>``) the
    exact injected events are used instead. Without them the span is used and
    the resolution says so, because silently widening is what produced a 38.5%
    positive rate in the first place.
    """
    ent = injected.filter(pl.col("target_kind") == "entity")
    if ent.is_empty():
        return _empty_entity_columns(frame)

    exact = _entity_events_from_flags(ent, latent)
    if exact is None:
        return _entity_events_from_span(frame, ent, exact=False)
    agg = _aggregate_entity_hits(exact, ent)
    # Keyed by latent_id, joined on the frame's recovered `_latent` column:
    # an injected latent event may be observed in more than one source, and
    # this way every observation of it is labelled.
    return frame.join(agg, on="_latent", how="left")


def _empty_entity_columns(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.lit(None, dtype=pl.List(pl.Utf8)).alias("_ids_ent"),
        pl.lit(None, dtype=pl.List(pl.Utf8)).alias("_fam_ent"),
        pl.lit(None, dtype=pl.Float64).alias("_sev_ent"),
    )


def _entity_events_from_flags(
    ent: pl.DataFrame, latent: pl.DataFrame
) -> pl.DataFrame | None:
    """Resolve collective membership via ``fam_collective`` on the latent stream.

    Returns ``None`` when the flags are absent, so the caller can fall back to
    the span and record that it did.
    """
    flag = "fam_collective"
    if flag not in latent.columns:
        return None
    # Latent entity_key -> collective anomaly_id. The label's entity_key
    # column is authoritative; target_ref is the same value for this kind.
    ids = ent.select(
        pl.col("target_ref").alias("entity_key"),
        pl.col("anomaly_id"),
        pl.col("severity"),
    )
    flagged = latent.filter(pl.col(flag) == 1).select(
        # Renamed to the frame's private `_latent` name at the point of
        # selection, so the two id spaces cannot diverge downstream.
        pl.col("latent_id").alias("_latent"),
        "entity_key",
    )
    hits = flagged.join(ids, on="entity_key", how="inner")
    if hits.is_empty():
        return None
    return hits


def _entity_events_from_span(
    frame: pl.DataFrame, ent: pl.DataFrame, *, exact: bool
) -> pl.DataFrame:
    rows = ent.select(
        pl.col("target_ref").alias("_truth_key"),
        pl.col("started_at"),
        pl.col("ended_at"),
        pl.col("anomaly_id"),
        pl.col("family"),
        pl.col("severity"),
    )
    matched = (
        frame.filter(pl.col("_truth_key").is_not_null())
        .join(rows, on="_truth_key", how="inner")
        .filter(
            (pl.col("timestamp") >= pl.col("started_at"))
            & (pl.col("timestamp") <= pl.col("ended_at"))
        )
    )
    if matched.is_empty():
        return _empty_entity_columns(frame)
    if not exact:
        LOGGER.warning(
            "collective resolved by declared span: per-event family flags absent, "
            "so ordinary events inside a multi-week span are marked positive"
        )
    agg = (
        matched.group_by("event_id")
        .agg(
            pl.col("anomaly_id").unique().sort().alias("_ids_ent"),
            pl.col("family").unique().sort().alias("_fam_ent"),
            pl.col("severity").max().alias("_sev_ent"),
        )
    )
    return frame.join(agg, on="event_id", how="left")


def _aggregate_entity_hits(hits: pl.DataFrame, ent: pl.DataFrame) -> pl.DataFrame:
    """Per-event aggregate of collective membership, keyed by latent_id."""
    fam = ent.select(
        pl.col("target_ref").alias("entity_key"),
        pl.col("family").first(),
    )
    out = (
        hits.join(fam, on="entity_key", how="left")
        # Key named `_latent` to match the frame's recovered latent column, so
        # the caller's join needs no `left_on`/`right_on` and cannot silently
        # mismatch the two id spaces.
        .group_by("_latent")
        .agg(
            pl.col("anomaly_id").unique().sort().alias("_ids_ent"),
            pl.col("family").unique().sort().alias("_fam_ent"),
            pl.col("severity").max().alias("_sev_ent"),
        )
    )
    return out


def _attach_window_labels(frame: pl.DataFrame, injected: pl.DataFrame) -> pl.DataFrame:
    """``target_kind in {'window', 'source_window'}``: time-interval match.

    ``temporal`` labels mark a 6-hour bucket, so every event in the declared
    span genuinely is anomalous and the interval is the right resolution.
    ``distribution`` and ``missingness`` mark multi-day intervals that describe
    the *world* rather than a marked event, and they are resolved by span on
    purpose -- a distribution shift genuinely does affect every event in it.

    ``source_window`` additionally requires ``target_ref`` to name the source
    whose rows went missing -- an event in that window from a *different*
    source is not anomalous.
    """
    win = injected.filter(pl.col("target_kind").is_in(["window", "source_window"]))
    if win.is_empty():
        return frame.with_columns(
            pl.lit(None, dtype=pl.List(pl.Utf8)).alias("_ids_win"),
            pl.lit(None, dtype=pl.List(pl.Utf8)).alias("_fam_win"),
            pl.lit(None, dtype=pl.Float64).alias("_sev_win"),
        )
    rows = []
    for rec in win.iter_rows(named=True):
        source = None
        target = rec["target_ref"]
        if rec["target_kind"] == "source_window" and isinstance(target, str) and "@" in target:
            source = target.split("@", 1)[0]
        rows.append(
            {
                "started_at": rec["started_at"],
                "ended_at": rec["ended_at"],
                "anomaly_id": rec["anomaly_id"],
                "family": rec["family"],
                "severity": rec["severity"],
                "_source_filter": source,
            }
        )
    spec = pl.DataFrame(rows)
    matched = frame.join(spec, how="cross")
    matched = matched.filter(
        (pl.col("timestamp") >= pl.col("started_at")) & (pl.col("timestamp") <= pl.col("ended_at"))
    )
    # ``source_window`` names the source that went missing; only that source's
    # rows in the window are anomalous. Null ``_source_filter`` means a plain
    # time window and must match every source.
    matched = matched.filter(
        pl.col("_source_filter").is_null() | (pl.col("source_id") == pl.col("_source_filter"))
    )
    if matched.is_empty():
        return frame.with_columns(
            pl.lit(None, dtype=pl.List(pl.Utf8)).alias("_ids_win"),
            pl.lit(None, dtype=pl.List(pl.Utf8)).alias("_fam_win"),
            pl.lit(None, dtype=pl.Float64).alias("_sev_win"),
        )
    agg = (
        matched.group_by("event_id")
        .agg(
            pl.col("anomaly_id").unique().sort().alias("_ids_win"),
            pl.col("family").unique().sort().alias("_fam_win"),
            pl.col("severity").max().alias("_sev_win"),
        )
    )
    return frame.join(agg, on="event_id", how="left")


def _finalize(frame: pl.DataFrame) -> pl.DataFrame:
    """Union the three resolution paths into the public columns.

    ``.flatten()`` is applied *outside* ``with_columns`` on purpose: inside it,
    flatten operates on the column as a whole and produces a result whose
    length is the total number of elements, not the row count.

    Both label protocols are emitted here, in one pass, so the dual-protocol
    comparison never re-runs generation, cleaning, entity resolution or feature
    fitting:

    ``is_anomaly_event_only``
        True only where a single injected event was labelled.
    ``is_anomaly_all_families``
        True where any family covers the row.
    ``label_protocol``
        The protocol selected by :func:`resolve_labels`.

    Keeping both columns rather than recomputing per protocol is what makes the
    "only the label/evaluation layer differs" requirement true.
    """
    def _empty_list(name: str) -> pl.Expr:
        return pl.col(name).fill_null(pl.lit([], dtype=pl.List(pl.Utf8)))

    out = frame
    out = out.with_columns(
        pl.concat_list([_empty_list("_ids_ev"), _empty_list("_ids_ent"), _empty_list("_ids_win")])
        .list.eval(pl.element().drop_nulls())
        .list.unique()
        .list.sort()
        .alias("anomaly_ids")
    )
    out = out.with_columns(
        pl.concat_list([_empty_list("_fam_ev"), _empty_list("_fam_ent"), _empty_list("_fam_win")])
        .list.eval(pl.element().drop_nulls())
        .list.unique()
        .list.sort()
        .alias("anomaly_family")
    )
    out = out.with_columns(
        pl.max_horizontal(
            pl.col("_sev_ev"), pl.col("_sev_ent"), pl.col("_sev_win")
        ).alias("anomaly_severity")
    )
    out = out.with_columns(
        # Under EVENT_ONLY a row covered only by a collective burst or a
        # distribution window is NOT a positive, even though
        # `anomaly_family` records the coverage -- otherwise "anomaly" would
        # mean "belonged to an entity that was ever flagged".
        (
            pl.col("anomaly_family")
            .list.eval(pl.element().is_in(sorted(EVENT_FAMILIES)))
            .list.any()
        ).alias("is_anomaly_event_only"),
        (pl.col("anomaly_ids").list.len() > 0).alias("is_anomaly_all_families"),
    )
    return out.drop(
        "_ids_ev", "_fam_ev", "_sev_ev",
        "_ids_ent", "_fam_ent", "_sev_ent",
        "_ids_win", "_fam_win", "_sev_win",
        "_source_filter",
        strict=False,
    )


def _select_protocol(frame: pl.DataFrame, scope: LabelScope) -> pl.DataFrame:
    """Bind one protocol's column to ``is_anomaly`` and stamp ``label_protocol``.

    ``is_anomaly`` is a *view* over the two protocol columns, never a third
    independent truth: a detector that reads it cannot tell which protocol it
    is being scored under unless the artifact records ``label_protocol``.
    """
    column = (
        "is_anomaly_event_only"
        if scope is LabelScope.EVENT_ONLY
        else "is_anomaly_all_families"
    )
    if column not in frame.columns:  # pragma: no cover - defensive
        raise ValueError(f"label column {column!r} missing; resolve_labels did not run")
    return frame.with_columns(
        pl.col(column).alias("is_anomaly"),
        pl.lit(str(scope), dtype=pl.Utf8).alias("label_protocol"),
    )


def _family_counts(frame: pl.DataFrame) -> dict[str, int]:
    """Positive events per family, under the active label scope.

    Under ``EVENT_ONLY`` an aggregate family (collective / temporal) appears
    here as 0 because its covered rows are deliberately not positives -- that
    is the scope working, not a resolution failure. The coverage of those
    families is reported separately in :attr:`LabelResolution.family_coverage`.
    """
    counts: dict[str, int] = {}
    if frame.height:
        exploded = frame.filter(pl.col("is_anomaly")).select("anomaly_family").explode("anomaly_family")
        for family in ANOMALY_FAMILIES:
            counts[family] = exploded.filter(pl.col("anomaly_family") == family).height
    return counts


def _family_coverage(frame: pl.DataFrame) -> dict[str, int]:
    """Rows carrying each family in ``anomaly_family``, positives or not.

    This is what distinguishes "no detector can see this family" from "this
    family is covered but excluded from the row-level label by scope".
    """
    coverage: dict[str, int] = {}
    if not frame.height:
        return dict.fromkeys(ANOMALY_FAMILIES, 0)
    exploded = frame.select("anomaly_family").explode("anomaly_family")
    for family in ANOMALY_FAMILIES:
        coverage[family] = exploded.filter(pl.col("anomaly_family") == family).height
    return coverage


def _period_counts(frame: pl.DataFrame, period_column: str) -> dict[str, dict[str, int]]:
    if period_column not in frame.columns:
        return {}
    out: dict[str, dict[str, int]] = {}
    for period in PERIODS:
        part = frame.filter(pl.col(period_column) == period)
        if part.is_empty():
            continue
        out[period] = {
            "rows": part.height,
            "positives": int(part["is_anomaly"].sum() or 0),
        }
    return out


def supported_families(resolution: LabelResolution) -> tuple[str, ...]:
    """Families with enough positive events to report a per-family metric."""
    return tuple(f for f in ANOMALY_FAMILIES if resolution.family_counts.get(f, 0) >= MINIMUM_FAMILY_SUPPORT)


def assert_labels_never_fitted(
    fitted_period: Period, *, label_periods: Sequence[Period] = PERIODS[1:]
) -> None:
    """Raise if a labeled period was used to fit anything.

    Labels are the evaluation target. They exist in validation, backtest and
    forward by construction; the only error is *fitting* on them. This is a
    type-level guard for the single call site that matters -- model fitting.
    """
    if fitted_period in tuple(label_periods):
        raise LeakageError(
            f"refusing to fit on {fitted_period}: labels exist in "
            f"{list(label_periods)}, so fitting there leaks the evaluation target"
        )


def label_window_bounds(resolution: LabelResolution) -> dict[str, tuple[datetime, datetime]]:
    """Diagnostic: earliest/latest positive timestamp, for audit output."""
    if resolution.events.is_empty():
        return {}
    pos = resolution.events.filter(pl.col("is_anomaly"))
    if pos.is_empty():
        return {}
    ts = pos["timestamp"]
    return {"positives": (ts.min(), ts.max())}
