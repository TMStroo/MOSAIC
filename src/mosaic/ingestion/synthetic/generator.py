"""Configurable synthetic multi-source world generator.

Why a generator
---------------
MOSAIC's research questions are about *integration and attribution*: which feature
family carries the signal, how much does entity-resolution error cost, what happens
when a source disappears. Answering those needs environments where the answer is
known in advance. So one seed produces:

* a latent entity population with per-entity activity archetypes,
* an event stream with hourly profiles, weekly seasonality, trend and regime shifts,
* a latent co-occurrence graph (communities emerge from shared site membership),
* a ground-truth entity map that each source observes *lossily*,
* nine families of injected anomalies with exact event and window labels.

Labels never enter the ingested data files. Every latent event carries a stable
``latent_id`` that is encoded into each source's ``record_id``; the label map is
written to a separate ``_truth/`` sidecar. The only module that joins them is
:mod:`mosaic.ingestion.synthetic.labels`, used by evaluation only.

Cost model: O(n_entities x n_days) for rate sampling, then O(n_events) for the
stream. Memory is dominated by the per-event NumPy arrays.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, ClassVar

import numpy as np
import polars as pl

from mosaic.ingestion.synthetic.world import SyntheticWorld
from mosaic.schema.ids import stable_hash

MINUTE = 60
HOUR = 3600
DAY = 86_400

#: Latent activity archetype -> (mean events/day, over-dispersion).
PROFILES: dict[str, tuple[float, float]] = {
    "commuter": (16.0, 0.35),
    "logistics": (34.0, 0.30),
    "sensor": (150.0, 0.20),
    "sporadic": (1.0, 0.80),
    "bursty": (8.0, 0.90),
    "night_owl": (13.0, 0.45),
}

TYPE_NAMES = [
    "access",
    "transaction",
    "movement",
    "sensor_reading",
    "communication",
    "incident",
    "maintenance",
]

#: Latent type mixture per archetype (ordered like ``TYPE_NAMES``).
TYPE_MIXTURES: dict[str, np.ndarray] = {
    "commuter": np.array([0.50, 0.20, 0.20, 0.04, 0.04, 0.01, 0.01]),
    "logistics": np.array([0.04, 0.46, 0.30, 0.04, 0.10, 0.04, 0.02]),
    "sensor": np.array([0.01, 0.01, 0.02, 0.82, 0.02, 0.10, 0.02]),
    "sporadic": np.array([0.28, 0.10, 0.18, 0.18, 0.12, 0.08, 0.06]),
    "bursty": np.array([0.20, 0.24, 0.26, 0.08, 0.12, 0.06, 0.04]),
    "night_owl": np.array([0.30, 0.08, 0.24, 0.16, 0.10, 0.06, 0.06]),
}

#: Latent value log-normal (location, sigma) per event type.
VALUE_PARAMS: dict[str, tuple[float, float]] = {
    "access": (-0.7, 0.8),
    "transaction": (3.4, 1.0),
    "movement": (1.0, 0.6),
    "sensor_reading": (2.3, 0.5),
    "communication": (0.4, 0.9),
    "incident": (1.6, 0.7),
    "maintenance": (0.2, 0.7),
}

SECTORS = ["retail", "manufacturing", "logistics", "healthcare", "energy", "public"]

FAMILIES = (
    "point",
    "contextual",
    "collective",
    "temporal",
    "behavioral",
    "relational",
    "cross_source",
    "distribution",
    "missingness",
    "entity_resolution",
)

#: Latent event types each source is structurally able to observe. This is what
#: makes multi-source integration necessary rather than decorative.
SOURCE_TYPE_WEIGHTS: dict[str, dict[str, float]] = {
    "transit_feed": {"movement": 0.72, "access": 0.16, "communication": 0.07, "incident": 0.05},
    "ledger_api": {"transaction": 0.66, "access": 0.14, "communication": 0.14, "maintenance": 0.06},
    "incident_logs": {"incident": 0.56, "maintenance": 0.18, "access": 0.14, "communication": 0.12},
    "sensor_net": {"sensor_reading": 0.86, "maintenance": 0.08, "access": 0.06},
}

#: source -> (reporting lag in days, per-day field-missing probability,
#:             probability a given latent event is observed at all)
SOURCE_PLAN: dict[str, tuple[float, float, float]] = {
    "transit_feed": (1.0, 0.05, 0.60),
    "ledger_api": (0.0, 0.08, 0.54),
    "incident_logs": (0.0, 0.12, 0.30),
    "sensor_net": (0.0, 0.03, 0.34),
}

#: How each source renders the same entity. Deliberately incompatible.
SOURCE_ID_STYLE: dict[str, str] = {
    "transit_feed": "entity_{n:05d}",
    "ledger_api": "ENT-{n:05d}",
    "incident_logs": "{n:05d}",
    "sensor_net": "site::{slug}",
}

SOURCE_SLUG: dict[str, str] = {
    "transit_feed": "municipal_transit_feed",
    "ledger_api": "merchant_ledger_api",
    "incident_logs": "incident_management_logs",
    "sensor_net": "industrial_sensor_net",
}

LETTERS = "abcdefghijklmnopqrstuvwxyz"


@dataclass(slots=True)
class SyntheticSpec:
    """Configuration of one synthetic environment.

    Robustness experiments vary ``missing_rate``, ``noise_rate`` and ``er_noise``
    and then *regenerate the world*, so degradation is a property of the data
    rather than a post-hoc corruption of a clean world.
    """

    name: str = "world_a"
    seed: int = 20260901
    n_entities: int = 400
    target_events: int = 40_000
    n_sources: int = 4
    start: str = "2021-01-04"  # a Monday
    days: int = 365
    profile_weights: dict[str, float] = field(
        default_factory=lambda: {
            "commuter": 0.24,
            "logistics": 0.18,
            "sensor": 0.12,
            "sporadic": 0.20,
            "bursty": 0.16,
            "night_owl": 0.10,
        }
    )
    season_amplitude: float = 0.30
    trend_rate_per_day: float = 0.0004
    regime_count: int = 2
    missing_rate: float = 0.04
    duplicate_rate: float = 0.02
    noise_rate: float = 0.01
    er_noise: float = 0.0
    source_reliability: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(SOURCE_PLAN, 0.9)
    )
    ambiguous_rate: float = 0.06
    anomaly_prevalence: float = 0.03
    anomaly_families: tuple[str, ...] = FAMILIES
    scale_target: bool = True
    #: When true, ``n_entities`` is derived from ``target_events``. Off by default:
    #: the entity population is a modelling decision, not a function of volume.
    derive_entities: bool = False

    def sources(self) -> list[str]:
        return list(SOURCE_PLAN)[: int(max(2, min(len(SOURCE_PLAN), self.n_sources)))]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def n_entities_for_target(self) -> int:
        """Entity population implied by ``target_events`` at this duration.

        Used by the scaling presets, where the *event count* is the thing being
        varied. For research presets ``n_entities`` is set explicitly instead: a
        population of 20 actors makes entity resolution, community structure and
        per-entity behavioural baselines all degenerate, so the entity count is
        never left to an accident of arithmetic.
        """
        days = float(max(7, int(self.days)))
        weights = np.array([self.profile_weights[p] for p in PROFILES], dtype=float)
        weights = weights / weights.sum()
        mean_per_entity = float(
            (weights * np.array([PROFILES[p][0] for p in PROFILES])).sum()
        )
        want = self.target_events / (mean_per_entity * days)
        return int(max(20, min(1_000_000, round(want))))

    def resolved(self) -> SyntheticSpec:
        """Optionally derive the entity population from the event target.

        ``derive_entities`` is off for research presets and on for scaling presets.
        Either way the *realized* event count is recorded in the manifest, so the
        experiment never has to assume it hit its target.
        """
        if self.derive_entities and self.scale_target:
            self.n_entities = self.n_entities_for_target()
        self.n_entities = int(max(20, self.n_entities))
        return self

    def budget(self) -> int:
        """Total number of *events* to label as anomalous.

        This is the contract the benchmark relies on: injecting roughly
        ``anomaly_prevalence`` of the stream keeps the class balance comparable
        across experiments, so a prevalence change actually changes prevalence
        rather than just the label count. Window-level families (source outages)
        are not event budgets and are budgeted separately.
        """
        return int(max(30, round(self.anomaly_prevalence * self.target_events)))

    #: Families whose budget is a count of individual events.
    EVENT_FAMILIES = (
        "point",
        "contextual",
        "behavioral",
        "relational",
        "cross_source",
        "distribution",
    )
    #: Families whose budget is a count of *groups* (each group labels many events).
    #: ClassVar: a fixed budget map, never mutated per-instance.
    GROUP_FAMILIES: ClassVar[dict[str, int]] = {"collective": 45, "temporal": 90}

    def family_budgets(self) -> dict[str, int]:
        """Split the event budget across families, deterministically.

        The split is proportional to how many events each family can plausibly
        touch, so no family silently swallows the whole budget.
        """
        event_families = [f for f in self.anomaly_families if f in self.EVENT_FAMILIES]
        group_families = [f for f in self.anomaly_families if f in self.GROUP_FAMILIES]
        out: dict[str, int] = {}
        if event_families:
            per = max(4, self.budget() // len(event_families))
            for fam in event_families:
                out[fam] = per
        for fam in group_families:
            size = self.GROUP_FAMILIES[fam]
            out[fam] = max(1, self.budget() // size)  # number of groups
        if "missingness" in self.anomaly_families:
            out["missingness"] = max(1, self.budget() // 90)
        return out

    def fingerprint(self) -> str:
        return stable_hash(self.as_dict(), length=20)


@dataclass(slots=True)
class Stream:
    """Mutable per-event arrays, kept in one object so injectors stay short."""

    latent_id: list[str]
    t: np.ndarray
    ent: np.ndarray
    etype: np.ndarray
    value: np.ndarray
    location: np.ndarray
    related: np.ndarray
    fam: dict[str, np.ndarray]
    outage: list[tuple[str, int, int]] = field(default_factory=list)
    dist_window: list[tuple[int, int]] = field(default_factory=list)

    def n(self) -> int:
        return int(self.t.size)

    def grow(self, extra: int) -> None:
        """Append ``extra`` empty slots and return the indices of the new slots."""
        start = len(self.latent_id)
        self.latent_id.extend(f"L{i:09d}" for i in range(start, start + extra))
        self.t = np.concatenate([self.t, np.zeros(extra, dtype=np.int64)])
        self.ent = np.concatenate([self.ent, np.zeros(extra, dtype=np.int64)])
        self.etype = np.concatenate([self.etype, np.empty(extra, dtype=object)])
        self.value = np.concatenate([self.value, np.zeros(extra, dtype=np.float64)])
        self.location = np.concatenate([self.location, np.zeros(extra, dtype=np.int64)])
        self.related = np.concatenate([self.related, np.full(extra, -1, dtype=np.int64)])
        for name, arr in list(self.fam.items()):
            self.fam[name] = np.concatenate([arr, np.zeros(extra, dtype=np.int8)])
        return np.arange(start, start + extra, dtype=np.int64)


def _epoch0(spec: SyntheticSpec) -> int:
    return int(datetime.fromisoformat(spec.start).timestamp())


def _iso(unix_s: int | np.ndarray) -> Any:
    """Unix seconds -> ISO-8601 string (scalar) or array of strings."""
    return np.asarray(unix_s, dtype="datetime64[s]").astype(str)


def _new_stream(n: int) -> Stream:
    return Stream(
        latent_id=[f"L{i:09d}" for i in range(n)],
        t=np.zeros(n, dtype=np.int64),
        ent=np.zeros(n, dtype=np.int64),
        etype=np.empty(n, dtype=object),
        value=np.zeros(n, dtype=np.float64),
        location=np.zeros(n, dtype=np.int64),
        related=np.full(n, -1, dtype=np.int64),
        fam={name: np.zeros(n, dtype=np.int8) for name in FAMILIES},
    )


# --------------------------------------------------------------- sampling
def _sample_entities(spec: SyntheticSpec, rng: np.random.Generator) -> pl.DataFrame:
    names = list(PROFILES)
    weights = np.array([spec.profile_weights[n] for n in names], dtype=float)
    weights = weights / weights.sum()
    archetype = np.array(names, dtype=object)[rng.choice(len(names), size=spec.n_entities, p=weights)]
    n_sites = max(1, spec.n_entities // 6)
    site = rng.integers(0, n_sites, size=spec.n_entities)
    return pl.DataFrame(
        {
            "entity_key": [f"E{i:06d}" for i in range(spec.n_entities)],
            "archetype": archetype.tolist(),
            "sector": rng.choice(SECTORS, size=spec.n_entities).tolist(),
            "site": [f"S{s:04d}" for s in site],
        }
    )


def _day_levels(spec: SyntheticSpec, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    """Per-day global multiplier: weekday effect, seasonality, trend, regime changes."""
    n_days = max(7, int(spec.days))
    days = np.arange(n_days, dtype=np.float64)
    weekday = (datetime.fromisoformat(spec.start).weekday() + days) % 7
    level = np.where(weekday < 5, 1.0, 0.58)
    level = level * (1.0 + spec.season_amplitude * np.sin(2 * np.pi * days / 28.0))
    level = level * np.exp(spec.trend_rate_per_day * days)

    cuts: list[int] = []
    n_reg = int(max(0, min(4, spec.regime_count)))
    if n_reg and n_days > 40:
        candidates = np.arange(15, n_days - 15)
        chosen = sorted(rng.choice(candidates, size=min(n_reg, candidates.size), replace=False).tolist())
        mults = rng.choice([0.60, 0.75, 1.35, 1.70], size=len(chosen), replace=False)
        for cut, mult in zip(chosen, mults.tolist(), strict=True):
            level[cut:] *= float(mult)
            cuts.append(int(cut))
    return level, cuts


def _hour_profile(archetype: str) -> np.ndarray:
    hours = np.arange(24, dtype=np.float64)
    if archetype == "sensor":
        base = np.ones(24)
    elif archetype == "night_owl":
        base = 0.2 + np.exp(-0.5 * ((hours - 22) / 3.0) ** 2)
    elif archetype == "logistics":
        base = (
            0.5
            + np.exp(-0.5 * ((hours - 9) / 2.4) ** 2)
            + 0.9 * np.exp(-0.5 * ((hours - 17) / 2.6) ** 2)
        )
    else:
        base = (
            0.2
            + np.exp(-0.5 * ((hours - 8) / 1.9) ** 2)
            + 0.85 * np.exp(-0.5 * ((hours - 18) / 2.1) ** 2)
        )
    return base / base.sum()


def _sample_stream(spec: SyntheticSpec, rng: np.random.Generator, entities: pl.DataFrame) -> Stream:
    """Negative-binomial per entity-day, then hour / type / value / partner sampling."""
    n_days = max(7, int(spec.days))
    n_ent = spec.n_entities
    day_level, _cuts = _day_levels(spec, rng)

    archetype = entities["archetype"].to_numpy()
    mean_rate = np.array([PROFILES[p][0] for p in archetype], dtype=np.float64)
    dispersion = np.clip(np.array([PROFILES[p][1] for p in archetype], dtype=np.float64), 0.05, 2.0)
    shape = 1.0 / dispersion**2
    scale = mean_rate * dispersion**2

    lam = np.clip(mean_rate[:, None] * day_level[None, :], 1e-6, 1e5)
    counts = rng.poisson(rng.gamma(shape=shape[:, None], scale=scale[:, None], size=lam.shape))

    per_entity = counts.sum(axis=1)
    total = int(per_entity.sum())
    if total == 0:  # pragma: no cover - guarded by spec.resolved()
        raise ValueError("spec produced zero events; raise target_events or n_entities")

    # Flatten the (entity, day) count matrix once: cell index = entity * n_days + day
    # is entity-major / day-minor, which is exactly the row-major flatten order.
    cell = np.repeat(np.arange(n_ent * n_days, dtype=np.int64), counts.reshape(-1))
    ent_idx = cell // n_days
    day_idx = cell % n_days

    cdfs = np.cumsum(np.stack([_hour_profile(p) for p in archetype]), axis=1)
    cdfs = cdfs / cdfs[:, -1:]
    hour = (cdfs[ent_idx] < rng.random(total)[:, None]).sum(axis=1).clip(0, 23)
    t = _epoch0(spec) + day_idx * DAY + hour.astype(np.int64) * HOUR + (
        rng.random(total) * 3600
    ).astype(np.int64)

    etype = np.empty(total, dtype=object)
    for name, mix in TYPE_MIXTURES.items():
        mask = archetype[ent_idx] == name
        k = int(mask.sum())
        if k:
            etype[mask] = rng.choice(TYPE_NAMES, size=k, p=mix)

    value = np.zeros(total, dtype=np.float64)
    for name, (loc, sigma) in VALUE_PARAMS.items():
        mask = etype == name
        k = int(mask.sum())
        if k:
            value[mask] = np.exp(rng.normal(loc, sigma, size=k))
    value = np.round(value, 4)

    # partners: same site, with archetype affinity -> communities without an
    # explicit clustering step
    site_of = entities["site"].to_numpy()
    partner_of = np.full(n_ent, -1, dtype=np.int64)
    for site in np.unique(site_of):
        members = np.flatnonzero(site_of == site)
        if members.size >= 2:
            partner_of[members] = members[rng.integers(0, members.size, size=members.size)]
    has_partner = partner_of >= 0
    related = partner_of[ent_idx]
    keep = has_partner[ent_idx] & (rng.random(total) < 0.6)
    related = np.where(keep, related, -1)

    location = rng.integers(0, 24, size=total)

    stream = _new_stream(total)
    stream.t = t
    stream.ent = ent_idx
    stream.etype = etype
    stream.value = value
    stream.location = location
    stream.related = related
    return stream


# ------------------------------------------------------------- injectors
def _inject_point(spec: SyntheticSpec, rng: np.random.Generator, s: Stream, budget: int) -> None:
    k = min(budget, s.n())
    idx = rng.choice(s.n(), size=k, replace=False)
    med = float(np.median(s.value))
    mad = max(1e-6, 1.4826 * float(np.median(np.abs(s.value - med))))
    s.value[idx] = np.round(
        med + rng.uniform(9.0, 22.0, size=k) * mad * rng.choice([-1.0, 1.0], size=k), 4
    )
    s.fam["point"][idx] = 1


def _inject_contextual(spec: SyntheticSpec, rng: np.random.Generator, s: Stream, budget: int) -> None:
    """Globally ordinary values at an unusual hour for this entity."""
    attempts = max(1, min(budget, s.n() // 4))
    t0 = _epoch0(spec)
    for _ in range(attempts):
        e = int(rng.integers(0, spec.n_entities))
        cand = np.flatnonzero(s.ent == e)
        if cand.size < 3:
            continue
        chosen = int(rng.choice(cand))
        day_start = t0 + ((int(s.t[chosen]) - t0) // DAY) * DAY
        s.t[chosen] = day_start + int(rng.choice([2, 3, 4, 23])) * HOUR
        s.value[chosen] = round(float(s.value[chosen] * rng.uniform(1.3, 1.9)), 4)
        s.fam["contextual"][chosen] = 1


def _inject_collective(spec: SyntheticSpec, rng: np.random.Generator, s: Stream, groups: int) -> None:
    """A swarm: many events by *distinct* entities inside one short window.

    Implemented by drawing events from across the whole stream and moving them
    into the window, so the group size is a controlled parameter rather than
    whatever the stream happened to contain. Without this, a collective anomaly
    silently vanishes at low event rates.
    """
    t0 = _epoch0(spec)
    t1 = t0 + max(7, int(spec.days)) * DAY
    span = int(2.5 * HOUR)
    target_per_group = spec.GROUP_FAMILIES["collective"]
    for _ in range(max(1, groups)):
        center = int(rng.integers(t0 + DAY, max(t0 + DAY + 1, t1 - DAY)))
        k = int(rng.integers(max(8, target_per_group // 2), target_per_group + 1))
        donors = rng.choice(s.n(), size=min(k * 3, s.n()), replace=False)
        # prefer distinct entities: take at most one event per entity first
        seen: set[int] = set()
        picks: list[int] = []
        for i in donors.tolist():
            e = int(s.ent[i])
            if e in seen:
                continue
            seen.add(e)
            picks.append(i)
            if len(picks) >= k:
                break
        if len(picks) < 5:
            continue
        take = np.array(sorted(picks), dtype=np.int64)
        s.t[take] = center + rng.integers(0, span, size=take.size)
        s.etype[take] = "incident"
        s.value[take] = np.round(s.value[take] * rng.uniform(0.5, 1.4, size=take.size), 4)
        s.fam["collective"][take] = 1


def _inject_temporal(spec: SyntheticSpec, rng: np.random.Generator, s: Stream, windows: int) -> None:
    """Event-rate spike shared across many entities inside a short interval.

    Existing events in the window are *added to* by cloning them with fresh ids,
    so the density increase is a controlled multiple rather than a function of
    how busy that hour happened to be.
    """
    n_days = max(7, int(spec.days))
    t0 = _epoch0(spec)
    t1 = t0 + n_days * DAY
    target_per_window = spec.GROUP_FAMILIES["temporal"]
    for _ in range(max(1, windows)):
        start = int(rng.integers(t0 + 2 * DAY, max(t0 + 3 * DAY, t1 - DAY)))
        span = int(rng.choice([2, 3, 5]) * HOUR)
        mask = np.flatnonzero((s.t >= start) & (s.t < start + span))
        donors = mask if mask.size >= 10 else rng.choice(s.n(), size=min(10, s.n()), replace=False)
        want = max(20, int(rng.uniform(0.6, 1.2) * target_per_window))
        extra = max(10, want - int(mask.size))
        new = s.grow(extra)
        pick = donors[rng.integers(0, donors.size, size=extra)]
        s.t[new] = start + rng.integers(0, span, size=extra)
        s.ent[new] = s.ent[pick]
        s.etype[new] = s.etype[pick]
        s.value[new] = s.value[pick] * rng.uniform(0.7, 1.3, size=extra)
        s.location[new] = s.location[pick]
        s.related[new] = s.related[pick]
        # the window is labelled as a whole: existing events in it plus the clones
        s.fam["temporal"][new] = 1
        if mask.size:
            s.fam["temporal"][mask] = 1


def _inject_behavioral(spec: SyntheticSpec, rng: np.random.Generator, s: Stream, budget: int) -> None:
    """A subset of entities switch to a quiet maintenance regime for a stretch.

    Entities and window lengths are chosen so the *number of labelled events* lands
    near the budget, instead of labelling an arbitrary slice of the stream.
    """
    n_days = max(7, int(spec.days))
    t0 = _epoch0(spec)
    t1 = t0 + n_days * DAY
    max_entities = max(2, min(spec.n_entities // 4, 40))
    per_entity = max(1, budget // max_entities)
    entities = rng.choice(spec.n_entities, size=max_entities, replace=False)
    for e in entities:
        candidates = np.flatnonzero(s.ent == e)
        if candidates.size == 0:
            continue
        want = min(candidates.size, per_entity)
        span_days = int(np.clip(want / max(1.0, candidates.size / n_days), 3, max(3, n_days // 3)))
        start = int(rng.integers(t0 + DAY, max(t0 + 2 * DAY, t1 - span_days * DAY)))
        end = start + span_days * DAY
        inside = candidates[(s.t[candidates] >= start) & (s.t[candidates] < end)]
        if inside.size == 0:
            continue
        take = inside if inside.size <= want else rng.choice(inside, size=want, replace=False)
        s.etype[take] = "maintenance"
        s.value[take] = np.round(s.value[take] * rng.uniform(0.1, 0.35, size=take.size), 4)
        s.fam["behavioral"][take] = 1


def _inject_relational(spec: SyntheticSpec, rng: np.random.Generator, s: Stream, budget: int) -> None:
    """Rewire a few events to partners outside the entity's own community."""
    k = min(budget, s.n() // 4)
    idx = rng.choice(s.n(), size=k, replace=False)
    s.related[idx] = rng.integers(0, spec.n_entities, size=k)
    s.fam["relational"][idx] = 1


def _inject_cross_source(spec: SyntheticSpec, rng: np.random.Generator, s: Stream, budget: int) -> None:
    """Mark events that sources will render inconsistently (see ``render.py``)."""
    k = min(budget, s.n() // 3)
    idx = rng.choice(s.n(), size=k, replace=False)
    s.fam["cross_source"][idx] = 1


def _inject_distribution(spec: SyntheticSpec, rng: np.random.Generator, s: Stream, budget: int) -> None:
    """Global location-prevalence shift over a period (distribution anomaly)."""
    n_days = max(7, int(spec.days))
    t0 = _epoch0(spec)
    for _ in range(max(1, budget // 200)):
        start_day = int(rng.integers(0, max(1, n_days - 6)))
        end_day = min(n_days, start_day + int(rng.integers(3, 12)))
        idx = np.flatnonzero((s.t >= t0 + start_day * DAY) & (s.t < t0 + end_day * DAY))
        if idx.size == 0:
            continue
        s.location[idx] = rng.integers(0, 3, size=idx.size)  # concentration on rare zones
        s.fam["distribution"][idx] = 1
        s.dist_window.append((start_day, end_day))


def _inject_missingness(spec: SyntheticSpec, rng: np.random.Generator, s: Stream, budget: int) -> None:
    """Source outage windows: a data-quality failure mode, not a behavioural one."""
    n_days = max(7, int(spec.days))
    sources = spec.sources()
    for _ in range(max(1, budget // 90)):
        src = sources[int(rng.integers(0, len(sources)))]
        start_day = int(rng.integers(0, max(1, n_days - 3)))
        end_day = min(n_days, start_day + int(rng.integers(2, 7)))
        s.outage.append((src, start_day, end_day))


INJECTORS = {
    "point": _inject_point,
    "contextual": _inject_contextual,
    "collective": _inject_collective,
    "temporal": _inject_temporal,
    "behavioral": _inject_behavioral,
    "relational": _inject_relational,
    "cross_source": _inject_cross_source,
    "distribution": _inject_distribution,
    "missingness": _inject_missingness,
}


def _sort_stream(s: Stream) -> Stream:
    order = np.argsort(s.t, kind="stable")
    s.t = s.t[order]
    s.ent = s.ent[order]
    s.etype = s.etype[order]
    s.value = s.value[order]
    s.location = s.location[order]
    s.related = s.related[order]
    s.latent_id = [s.latent_id[i] for i in order]
    for key in s.fam:
        s.fam[key] = s.fam[key][order]
    return s


# -------------------------------------------------------------- ER ground truth
def _entity_links(spec: SyntheticSpec, rng: np.random.Generator, er: dict[str, Any]) -> pl.DataFrame:
    """Ground-truth (source_ref -> entity_key) map for the *clean* rendered ids.

    The map is extended with the typo-damaged references the renderer actually
    emitted (see :func:`mosaic.ingestion.synthetic.render_source`, which returns
    the observed->latent mapping directly). It must cover every reference present
    in the data: if it only recorded clean ids, the resolver would be graded only
    on the references it finds easy, and the measured precision/recall would
    describe a different problem than the one being solved.
    """
    n = spec.n_entities
    numeric = np.random.default_rng(spec.seed + 991).integers(10_000_000, 99_000_000, size=n)
    ambiguous = set(
        rng.choice(n, size=int(n * float(np.clip(spec.ambiguous_rate, 0.0, 0.5))), replace=False).tolist()
    )
    rows: list[dict[str, Any]] = []
    for source_id in spec.sources():
        style = SOURCE_ID_STYLE[source_id]
        clean = [render_entity_id(style, i, int(numeric[i])) for i in range(n)]
        for victim, donor in er["merges"]:
            clean[victim] = clean[donor]
        for e in range(n):
            rows.append(
                {
                    "source_id": source_id,
                    "source_ref": clean[e],
                    "entity_key": f"E{e:06d}",
                    "is_ambiguous": e in ambiguous,
                    "is_clean": True,
                }
            )
    return pl.DataFrame(rows)


def with_observed_refs(
    links: pl.DataFrame, observed_maps: dict[str, dict[str, int]]
) -> pl.DataFrame:
    """Add the damaged references observed in the data to the truth map.

    ``observed_maps[source_id]`` maps every reference that source actually emitted
    to its latent entity index, produced by the renderer itself. Because the
    renderer is the only component that damages identifiers, asking it is exact -
    no repair heuristic is needed, and a reference that genuinely cannot be
    attributed is simply absent rather than guessed.
    """
    extra: list[dict[str, Any]] = []
    for source_id, mapping in observed_maps.items():
        clean_rows = links.filter(pl.col("source_id") == source_id)
        known = set(clean_rows["source_ref"].to_list())
        by_entity = dict(clean_rows.select("entity_key", "source_ref").iter_rows())
        for ref, entity_index in mapping.items():
            if ref in known:
                continue
            key = f"E{int(entity_index):06d}"
            if key in by_entity:
                extra.append(
                    {
                        "source_id": source_id,
                        "source_ref": ref,
                        "entity_key": key,
                        "is_ambiguous": False,
                        "is_clean": False,
                    }
                )
    if not extra:
        return links
    return pl.concat([links, pl.DataFrame(extra)], how="vertical")


def render_entity_id(style: str, entity: int, numeric: int) -> str:
    """Per-source identifier style. Incompatible on purpose."""
    if style.startswith("entity_"):
        return f"entity_{entity:05d}"
    if style.startswith("ENT-"):
        return f"ENT-{entity:05d}"
    if style == "site::{slug}":
        return f"site::{LETTERS[entity % 26]}{LETTERS[(entity // 26) % 26]}{entity % 1000:03d}"
    del numeric
    return f"{entity:05d}"


def _inject_er_noise(spec: SyntheticSpec, rng: np.random.Generator) -> dict[str, Any]:
    """Corrupt the latent entity map with merges; return the corruption record."""
    rate = float(np.clip(spec.er_noise, 0.0, 0.4))
    if rate <= 0 or spec.n_entities < 20:
        return {"merges": [], "rate": 0.0}
    k = int(spec.n_entities * rate)
    victims = rng.choice(spec.n_entities, size=k, replace=False)
    donors = rng.choice(spec.n_entities, size=k, replace=False)
    merges = [
        [int(v), int(d)]
        for v, d in zip(victims.tolist(), donors.tolist(), strict=True)
        if v != d
    ]
    return {"merges": merges, "rate": rate}


# ----------------------------------------------------------------- outputs
def _latent_frame(spec: SyntheticSpec, s: Stream) -> pl.DataFrame:
    t0 = _epoch0(spec)
    # Per-event family flags, one Int8 column per injected family. The
    # aggregate ``labels.parquet`` deliberately records only a target
    # (entity / window / event) and its span, which is lossy for the families
    # whose anomaly is a *set* of events: a collective burst moves ~45 events
    # by ~45 distinct entities into one 2.5h window, but the label's
    # ``started_at``/``ended_at`` can span months because donors are drawn
    # from across the stream. Resolving that span marks hundreds of ordinary
    # events positive. These flags are the exact injected events, so
    # ``mosaic.labels`` can resolve them without widening the span.
    #
    # This is additive: no existing column changes, and no existing artifact or
    # published result depends on these names.
    family_flags = {
        f"fam_{name}": s.fam[name].astype(int).tolist()
        for name in sorted(s.fam)
    }
    return pl.DataFrame(
        {
            "latent_id": s.latent_id,
            "timestamp": _iso(s.t),
            "entity_key": [f"E{int(x):06d}" for x in s.ent],
            "related_key": [f"E{int(x):06d}" if x >= 0 else None for x in s.related],
            "event_type": s.etype.tolist(),
            "event_value": s.value.tolist(),
            "location_id": [f"Z{int(x):03d}" for x in s.location],
            "day_index": ((s.t - t0) // DAY).astype(int).tolist(),
            **family_flags,
        }
    )


def _relations_frame(s: Stream) -> pl.DataFrame:
    mask = s.related >= 0
    return pl.DataFrame(
        {
            "source_key": [f"E{int(a):06d}" for a in s.related[mask]],
            "target_key": [f"E{int(e):06d}" for e, m in zip(s.ent, mask, strict=True) if m],
            "weight": np.ones(int(mask.sum()), dtype=np.int64).tolist(),
        }
    )


TRUTH_SCHEMA: dict[str, Any] = {
    "anomaly_id": pl.Utf8,
    "family": pl.Utf8,
    "target_kind": pl.Utf8,
    "target_ref": pl.Utf8,
    "latent_id": pl.Utf8,
    "entity_key": pl.Utf8,
    "started_at": pl.Utf8,
    "ended_at": pl.Utf8,
    "severity": pl.Float64,
    "injected": pl.Boolean,
}


def _truth_frame(spec: SyntheticSpec, s: Stream) -> pl.DataFrame:
    """Event-level and window-level ground truth. Never used outside evaluation."""
    rows: list[dict[str, Any]] = []
    t0 = _epoch0(spec)

    def add(**kwargs: Any) -> None:
        rows.append(
            {
                "anomaly_id": None,
                "family": None,
                "target_kind": None,
                "target_ref": None,
                "latent_id": None,
                "entity_key": None,
                "started_at": None,
                "ended_at": None,
                "severity": 1.0,
                "injected": True,
                **kwargs,
            }
        )

    for fam in ("point", "contextual", "relational", "cross_source", "behavioral"):
        for i in np.flatnonzero(s.fam[fam] == 1):
            add(
                anomaly_id=f"lat_{fam}_{s.latent_id[i]}",
                family=fam,
                target_kind="event",
                target_ref=s.latent_id[i],
                latent_id=s.latent_id[i],
                entity_key=f"E{int(s.ent[i]):06d}",
                started_at=_iso(int(s.t[i])),
                ended_at=_iso(int(s.t[i])),
            )

    collective = np.flatnonzero(s.fam["collective"] == 1)
    for key in sorted({f"E{int(s.ent[i]):06d}" for i in collective}):
        members = [i for i in collective if f"E{int(s.ent[i]):06d}" == key]
        add(
            anomaly_id=f"lat_collective_{key}",
            family="collective",
            target_kind="entity",
            target_ref=key,
            entity_key=key,
            started_at=_iso(int(s.t[members].min())),
            ended_at=_iso(int(s.t[members].max())),
            severity=round(len(members) / 50.0, 4),
        )

    temporal = np.flatnonzero(s.fam["temporal"] == 1)
    for key in sorted({f"W{int((s.t[i] - t0) // (6 * HOUR)):06d}" for i in temporal}):
        members = [i for i in temporal if f"W{int((s.t[i] - t0) // (6 * HOUR)):06d}" == key]
        add(
            anomaly_id=f"lat_temporal_{key}",
            family="temporal",
            target_kind="window",
            target_ref=key,
            started_at=_iso(int(s.t[members].min())),
            ended_at=_iso(int(s.t[members].max())),
            severity=round(len(members) / 100.0, 4),
        )

    for start_day, end_day in s.dist_window:
        add(
            anomaly_id=f"lat_distribution_{start_day:04d}",
            family="distribution",
            target_kind="window",
            target_ref=f"D{start_day:04d}-{end_day:04d}",
            started_at=_iso(t0 + start_day * DAY),
            ended_at=_iso(t0 + end_day * DAY),
        )

    for src, start_day, end_day in s.outage:
        add(
            anomaly_id=f"lat_missingness_{src}_{start_day:04d}",
            family="missingness",
            target_kind="source_window",
            target_ref=f"{src}@{start_day:04d}",
            started_at=_iso(t0 + start_day * DAY),
            ended_at=_iso(t0 + end_day * DAY),
        )

    if not rows:
        return pl.DataFrame(schema=TRUTH_SCHEMA)
    # Column-wise construction: every value is coerced to a plain Python scalar
    # first, because NumPy str_/0-d arrays are not accepted by from_dicts.
    columns: dict[str, list[Any]] = {name: [] for name in TRUTH_SCHEMA}
    for row in rows:
        extra = set(row) - set(TRUTH_SCHEMA)
        if extra:
            raise KeyError(f"truth row has keys outside the schema: {sorted(extra)}")
        for name in TRUTH_SCHEMA:
            columns[name].append(_scalar(row[name]))
    return pl.DataFrame(columns, schema=TRUTH_SCHEMA)


def _scalar(value: Any) -> Any:
    """Coerce NumPy scalars / 0-d arrays / bytes into plain Python values."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def generate_world(spec: SyntheticSpec) -> SyntheticWorld:
    """Build a complete latent world plus ground truth. Pure function of the seed.

    The returned :class:`SyntheticWorld` also carries the raw :class:`Stream` and
    the entity-resolution corruption record (``_stream`` / ``_er``) because the
    source renderer needs the latent entity index, not just its rendered form.
    Those two fields are never serialised to disk.
    """
    resolved = SyntheticSpec(**spec.as_dict()).resolved()
    rng = np.random.default_rng(resolved.seed)

    entities = _sample_entities(resolved, rng)
    stream = _sample_stream(resolved, rng, entities)
    er = _inject_er_noise(resolved, rng)

    budgets = resolved.family_budgets()
    for fam in resolved.anomaly_families:
        injector = INJECTORS.get(fam)
        if injector is not None:
            injector(resolved, rng, stream, budgets.get(fam, 4))

    stream = _sort_stream(stream)
    return SyntheticWorld(
        entities=entities,
        events=_latent_frame(resolved, stream),
        relations=_relations_frame(stream),
        truth=_truth_frame(resolved, stream),
        entity_links=_entity_links(resolved, rng, er),
        spec=resolved,
        stream=stream,
        er=er,
    )
