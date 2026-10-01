"""Per-source renderers: latent world -> four incompatible native schemas.

Each renderer owns its source's *observation process*: which latent event types
it can see, how it names entities, how much it lags, which fields go missing,
and what its records look like on disk. The schemas share no column names, so a
source adapter has to do real mapping work.

Noise injected here (not in the feature layer) so the validator has something to
find: missing fields, out-of-range values, malformed timestamps, whitespace and
case damage on categories, near-duplicate rows, and identifier typos.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from mosaic.ingestion.synthetic.vocabulary import (
    CATEGORY_TO_EVENT_TYPE,
    NATIVE_CATEGORIES,
    canonical_event_type,
)
from mosaic.ingestion.synthetic.generator import (
    DAY,
    HOUR,
    SOURCE_ID_STYLE,
    SOURCE_PLAN,
    SOURCE_TYPE_WEIGHTS,
    SyntheticSpec,
    Stream,
    render_entity_id,
)
from mosaic.utils.io import write_parquet

#: Native column name per canonical field, per source. Nothing is shared.
NATIVE_COLUMNS: dict[str, dict[str, str]] = {
    "transit_feed": {
        "record": "trip_uid",
        "time": "departure_iso",
        "actor": "cardholder_ref",
        "kind": "leg_type",
        "value": "fare_amount",
        "place": "stop_code",
        "peer": "transfer_to_ref",
        "flag": "is_reissued",
    },
    "ledger_api": {
        "record": "txn_hash",
        "time": "posted_at_utc",
        "actor": "merchant_customer_id",
        "kind": "channel_type",
        "value": "amount_eur",
        "place": "terminal_city",
        "peer": "counterparty_ref",
        "flag": "reversal_flag",
    },
    "incident_logs": {
        "record": "ticket_no",
        "time": "opened_at",
        "actor": "assignee_ref",
        "kind": "category",
        "value": "downtime_minutes",
        "place": "facility_code",
        "peer": "linked_actor_ref",
        "flag": "was_escalated",
    },
    "sensor_net": {
        "record": "reading_id",
        "time": "observed_at",
        "actor": "probe_id",
        "kind": "channel",
        "value": "reading",
        "place": "zone",
        "peer": "upstream_probe_id",
        "flag": "calibration_flag",
    },
}

def _types_allowed(source_id: str) -> dict[str, float]:
    return SOURCE_TYPE_WEIGHTS[source_id]


def _iso_array(t: np.ndarray) -> np.ndarray:
    """Unix seconds -> array of ISO-8601 strings (vectorised)."""
    return np.asarray(t, dtype="int64").astype("datetime64[s]").astype(str)


def _corrupt_timestamps(
    iso: np.ndarray, rng: np.random.Generator, noise_rate: float
) -> np.ndarray:
    """Apply realistic timestamp defects: offsets, day/month swaps, blanks."""
    out = iso.astype(object).copy()
    n = out.size
    if n == 0:
        return iso
    k = max(1, int(n * noise_rate))
    idx = rng.choice(n, size=k, replace=False)
    mode = rng.integers(0, 4, size=k)

    # 1. day/month transposition (a classic export bug)
    for i in idx[mode == 0]:
        text = str(out[i])
        if len(text) >= 10:
            out[i] = f"{text[0:8]}{text[8:10]}-{text[5:7]}-{text[0:4]}T{text[11:]}"
    # 2. trailing junk
    for i in idx[mode == 1]:
        out[i] = f"{out[i]} UTC"
    # 3. blank
    for i in idx[mode == 2]:
        out[i] = ""
    # 4. hour shifted by a day (off-by-one export)
    for i in idx[mode == 3]:
        try:
            ts = np.datetime64(str(out[i]))
            out[i] = str(ts + np.timedelta64(1, "D"))
        except Exception:  # pragma: no cover - defensive
            pass
    return np.array(out, dtype=object)


def _typo_ids(ids: np.ndarray, rng: np.random.Generator, rate: float) -> np.ndarray:
    """Character-level damage to a fraction of identifiers (ER must cope)."""
    out = ids.astype(object).copy()
    n = out.size
    if n == 0 or rate <= 0:
        return out
    k = max(1, int(n * rate))
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        text = str(out[i])
        if len(text) < 4:
            continue
        pos = int(rng.integers(1, len(text) - 1))
        roll = float(rng.random())
        if roll < 0.4:  # drop a character
            out[i] = text[:pos] + text[pos + 1 :]
        elif roll < 0.7:  # double a character
            out[i] = text[:pos] + text[pos] + text[pos:]
        else:  # separator damage
            out[i] = text.replace("_", " ").replace("-", " ").replace("::", " ")
    return out


def _dirty_categories(cats: np.ndarray, rng: np.random.Generator, rate: float) -> np.ndarray:
    """Case / whitespace damage. Normalized string matching must recover this."""
    out = cats.astype(object).copy()
    n = out.size
    if n == 0 or rate <= 0:
        return out
    k = max(1, int(n * rate))
    idx = rng.choice(n, size=k, replace=False)
    mode = rng.integers(0, 3, size=k)
    for i in idx[mode == 0]:
        out[i] = f"  {str(out[i]).upper()} "
    for i in idx[mode == 1]:
        out[i] = f"{out[i]}_v2"
    for i in idx[mode == 2]:
        out[i] = str(out[i]).title()
    return out


def render_source(
    spec: SyntheticSpec,
    source_id: str,
    s: Stream,
    rng: np.random.Generator,
    entity_rendered: np.ndarray,
) -> pl.DataFrame:
    """Project the latent stream into one source's native rows.

    ``entity_rendered`` is indexed by *latent entity index* and is read straight
    from the ground-truth link table, so identifier corruption (entity-resolution
    noise) is visible in the observed data exactly as it is in the truth map.
    """
    lag_days, field_missing, obs_prob = SOURCE_PLAN[source_id]
    cols = NATIVE_COLUMNS[source_id]
    cats = NATIVE_CATEGORIES[source_id]
    allowed = _types_allowed(source_id)

    # --- which latent events this source observes ---------------------------
    type_ok = np.zeros(s.n(), dtype=bool)
    for etype, weight in allowed.items():
        type_ok |= (s.etype == etype) & (rng.random(s.n()) < weight)
    seen = rng.random(s.n()) < obs_prob
    day_of = (s.t - int(np.datetime64(spec.start, "s").astype("int64"))) // DAY
    keep = type_ok & seen

    for outage_source, start_day, end_day in s.outage:
        if outage_source != source_id:
            continue
        keep &= ~((day_of >= start_day) & (day_of < end_day))

    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return _empty_native(source_id)

    m = idx.size
    rng_evt = np.random.default_rng(spec.seed + 5_000 * (list(SOURCE_PLAN).index(source_id) + 1))
    times = (s.t[idx] + int(lag_days * DAY) + rng_evt.integers(-600, 600, size=m)).astype(np.int64)
    actors = entity_rendered[s.ent[idx]]

    # --- native category (latent type -> source vocabulary) -----------------
    etypes = s.etype[idx]
    native_cat = np.empty(m, dtype=object)
    # NATIVE_CATEGORIES is keyed by canonical type and holds the *native* labels
    # that source uses for it, so the lookup is cats[latent], not the other way
    # round. Getting this backwards is silent: every row would arrive carrying a
    # canonical label that the adapter's dictionary cannot resolve.
    for latent, weight in allowed.items():
        mask = etypes == latent
        k = int(mask.sum())
        if not k:
            continue
        choices = sorted(cats.get(latent, ()))
        if not choices:  # latent type the source declares but does not label
            continue
        native_cat[mask] = rng_evt.choice(choices, size=k)

    # cross-source disagreement: the source reports a *different* category and a
    # large clock offset for events marked cross_source
    cross = s.fam["cross_source"][idx] == 1
    if cross.any():
        others = np.array(sorted({v for group in cats.values() for v in group}))
        n_cross = int(cross.sum())
        native_cat[cross] = others[rng_evt.integers(0, len(others), size=n_cross)]
        # large clock offset: the source is reporting a *different* event time for
        # the same latent event, which is what the cross-source agreement feature
        # has to detect
        times[cross] += rng_evt.integers(20, 60, size=n_cross) * HOUR

    # --- value: source-specific unit distortion ----------------------------
    value = s.value[idx].astype(np.float64).copy()
    if source_id == "incident_logs":
        value = np.round(value * rng_evt.uniform(2.0, 90.0, size=m), 2)  # minutes
    elif source_id == "ledger_api":
        value = np.round(value * 100.0, 2)  # cents
    elif source_id == "sensor_net":
        value = np.round(value + rng_evt.normal(0, 0.05, size=m), 4)
    else:
        value = np.round(value, 2)

    # --- partner references -------------------------------------------------
    partner = s.related[idx]
    peer = np.where(partner >= 0, entity_rendered[np.clip(partner, 0, None)], "")
    peer = np.where(peer == actors, "", peer)  # a source never points at itself

    # --- identifier damage + missing fields --------------------------------
    actors = _typo_ids(actors, rng_evt, 0.03)
    peer = _typo_ids(peer, rng_evt, 0.02)
    iso = _iso_array(times)
    iso = _corrupt_timestamps(iso, rng_evt, 0.006)
    native_cat = _dirty_categories(native_cat, rng_evt, 0.04)

    # NOTE: missing values are applied *after* the corruption guard below, so the
    # two are independent: 'this source skipped the field' must not be turned into
    # 0.0, and corrupt values must not become NaN.
    missing_peer = rng_evt.random(m) < field_missing * 1.4
    peer = np.where(missing_peer, "", peer)

    # Corrupt values are *impossible* values (negative, huge), never NaN. NaN is
    # reserved for genuinely missing fields above, so the validator can tell
    # "this source skipped a field" from "this source emitted nonsense" and the
    # cleaning profile can treat them differently.
    out_of_range = rng_evt.random(m) < spec.noise_rate
    if out_of_range.any():
        sign = rng_evt.choice([-1.0, 1.0], size=m)
        value = np.where(out_of_range, np.round(value * sign * rng_evt.uniform(50, 500, size=m), 2), value)
    value = np.where(np.isfinite(value), value, 0.0)

    # declared-missing fields, applied last. NaN is the only null sentinel a NumPy
    # float array has; the adapter converts it to a real null on read, so nothing
    # downstream ever computes a mean over a NaN.
    value = np.where(rng_evt.random(m) < field_missing, np.nan, value)

    record_id = [f"{source_id[:2]}-{s.latent_id[i]}" for i in idx]
    flag = (rng_evt.random(m) < 0.07).astype(np.int8)

    frame = pl.DataFrame(
        {
            cols["record"]: record_id,
            cols["time"]: iso.tolist(),
            cols["actor"]: actors.tolist(),
            cols["kind"]: native_cat.tolist(),
            cols["value"]: value.tolist(),
            cols["place"]: [f"P{int(x):04d}" for x in s.location[idx]],
            cols["peer"]: peer.tolist(),
            cols["flag"]: flag.tolist(),
            # lineage only: identifies the latent event, carries no label
            "source_batch": f"{spec.name}-{source_id}",
        }
    )

    # --- duplicates: exact copies and a corrupted near-duplicate ----------
    n_dup = int(m * spec.duplicate_rate)
    if n_dup:
        d = rng_evt.choice(m, size=n_dup, replace=False)
        dup = frame[d]
        if n_dup > 1 and rng_evt.random() < 0.5:
            # near-duplicate: same latent event, mangled record id + tiny value drift
            record_col = cols["record"]
            mangled = dup.with_columns(
                pl.col(record_col).str.replace("-L", "-Lr", literal=False).alias(record_col)
            )
            if cols["value"] in dup.columns:
                mangled = mangled.with_columns(
                    pl.col(cols["value"]).cast(pl.Float64, strict=False).mul(1.01).alias(cols["value"])
                )
            frame = pl.concat([frame, dup, mangled], how="vertical_relaxed")

    return frame


def _empty_native(source_id: str) -> pl.DataFrame:
    cols = NATIVE_COLUMNS[source_id]
    schema = {
        cols["record"]: pl.Utf8,
        cols["time"]: pl.Utf8,
        cols["actor"]: pl.Utf8,
        cols["kind"]: pl.Utf8,
        cols["value"]: pl.Float64,
        cols["place"]: pl.Utf8,
        cols["peer"]: pl.Utf8,
        cols["flag"]: pl.Int64,
        "source_batch": pl.Utf8,
    }
    return pl.DataFrame(schema=schema)


def render_world(spec: SyntheticSpec, s: Stream, entity_links: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Render every configured source from one latent stream."""
    out: dict[str, pl.DataFrame] = {}
    for slot, source_id in enumerate(spec.sources()):
        rows = entity_links.filter(pl.col("source_id") == source_id).sort("entity_key")
        latent_index = np.array([int(k[1:]) for k in rows["entity_key"]], dtype=np.int64)
        # source_ref is authoritative: it already contains the entity-resolution
        # corruption, so the observed ids and the truth map cannot disagree
        rendered = rows["source_ref"].to_numpy().astype(object)
        out[source_id] = render_source(
            spec,
            source_id,
            s,
            # deterministic per-source stream: slot index, never a string hash
            np.random.default_rng(spec.seed + 1009 * (slot + 1)),
            rendered,
        )
    return out


def write_source_parquet(spec: SyntheticSpec, frames: dict[str, pl.DataFrame], root) -> dict[str, int]:
    """Write each source as partitioned Parquet; return row counts."""
    import pathlib

    counts: dict[str, int] = {}
    base = pathlib.Path(root) / spec.name
    for source_id, frame in frames.items():
        target = base / source_id
        write_parquet(target / "part-000.parquet", frame)
        counts[source_id] = frame.height
    return counts
