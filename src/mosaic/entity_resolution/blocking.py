"""Blocking: candidate generation without a Cartesian product.

With ~300 references per source and 4 sources, the full cross-source product is
O(n^2) and, at 10M events, hopeless. Blocking restricts the pairs a matcher ever
sees to those that share a cheap, high-recall key.

Design decision: **blocking recall is measured, not assumed.** A strategy that
discards a true match lowers the ceiling on entity-resolution recall, so
:func:`generate_candidates` returns the *candidate set* and the evaluation layer
reports how many true matches survived blocking. Adding a strategy is additive;
it cannot lower recall because candidate sets are unioned.

Strategies, cheapest and highest-yield first:

``prefix3``     first 3 alphanumeric characters of the normalized ref
``digits``      the digit run extracted from the ref (the part that actually
                carries the identity in every synthetic id style)
``exact_norm``  full normalized string (catches identical renderings)
``char_bigram`` a hashed bigram prefix (catches single-character typos)
``anchor_type`` the first and last character plus length bucket (weak but cheap)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import polars as pl

Strategy = Literal[
    "exact_norm", "digits", "digits_suffix", "prefix3", "char_bigram", "anchor_type"
]
#: Order is cheapest/highest-yield first. ``digits_suffix`` exists because
#: measurement showed a source whose id style encodes only a *suffix* of the
#: identity (``site::cp042`` vs ``entity_00042``) shares no full digit run with the
#: others - without it, blocking recall caps at 0.5 no matter how good the matcher is.
STRATEGIES: tuple[Strategy, ...] = (
    "exact_norm",
    "digits",
    "digits_suffix",
    "prefix3",
    "char_bigram",
    "anchor_type",
)

_DIGITS = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class BlockKey:
    """One blocking key for one reference row."""

    reference_id: str
    source_id: str
    strategy: Strategy
    value: str

    def as_tuple(self) -> tuple[str, str]:
        return (self.strategy, self.value)


def normalized_ref(raw: str | None) -> str:
    """Canonical form of a source reference.

    Lowercases, collapses separators and whitespace, strips the source-specific
    punctuation. This is the same normalization cleaning applies, reused so the
    resolver and the cleaner can never disagree about what a reference *is*.
    """
    if raw is None:
        return ""
    text = str(raw).strip().lower()
    for ch in ("_", "-", "::", "/", "."):
        text = text.replace(ch, " ")
    return " ".join(text.split())


def digits_of(raw: str | None) -> str:
    """The digit run in a reference, or '' if it has none."""
    if raw is None:
        return ""
    match = _DIGITS.findall(str(raw))
    return "".join(match) if match else ""


def digits_suffix(raw: str | None, width: int = 3) -> str:
    """Trailing digits of the digit run, or '' if shorter than ``width``.

    Bridges id systems that encode only part of the identity: ``entity_00042``
    and ``site::cp042`` both end in ``042``. Deliberately weak on its own (three
    digits over a 400-entity population is not unique), which is why it is one
    strategy among several and why the matcher - not the block - decides.
    """
    digits = digits_of(raw)
    return digits[-width:] if len(digits) >= width else ""


def _prefix3(norm: str) -> str:
    return "".join(ch for ch in norm if ch.isalnum())[:3]


def _bigram(norm: str) -> str:
    compact = "".join(ch for ch in norm if ch.isalnum())
    if len(compact) < 2:
        return compact
    return compact[:2]  # a hashed bigram prefix, kept short so blocks stay small


def _anchor(norm: str) -> str:
    compact = "".join(ch for ch in norm if ch.isalnum())
    if not compact:
        return ""
    return f"{compact[0]}{compact[-1]}{min(len(compact), 20) // 4}"


_EXTRACTORS: dict[Strategy, Any] = {
    "exact_norm": normalized_ref,
    "digits": digits_of,
    "digits_suffix": digits_suffix,
    "prefix3": _prefix3,
    "char_bigram": _bigram,
    "anchor_type": _anchor,
}

_DESCRIPTIONS: dict[Strategy, str] = {
    "exact_norm": "full normalized reference string",
    "digits": "digit run only - carries identity when the id encodes it in full",
    "digits_suffix": "last 3 digits - bridges id styles that encode only a suffix",
    "prefix3": "first three alphanumeric characters",
    "char_bigram": "first two alphanumeric characters, typo tolerant at scale",
    "anchor_type": "first char + last char + length bucket",
}


def blocking_strategies() -> list[dict[str, str]]:
    """Machine-readable description of each strategy, for the report and UI."""
    return [
        {"strategy": name, "extractor": _EXTRACTORS[name].__name__, "description": _DESCRIPTIONS[name]}
        for name in STRATEGIES
    ]


def build_keys(frame: pl.DataFrame, *, ref_col: str = "entity_ref_norm", id_col: str = "ref_id") -> pl.DataFrame:
    """Explode references into (reference, strategy, value) rows.

    Empty values are dropped: a key that is '' matches every other empty key, so
    it would generate pure noise rather than candidates.
    """
    parts: list[pl.DataFrame] = []
    for name in STRATEGIES:
        extract = _EXTRACTORS[name]
        parts.append(
            frame.select(
                pl.col(id_col),
                pl.col("source_id"),
                pl.lit(name, dtype=pl.Utf8).alias("strategy"),
                # one expression per strategy rather than a Python-level apply:
                # this runs over every reference, so row-wise Python would dominate
                # the cost of candidate generation on large datasets
                _value_expr(pl.col(ref_col).cast(pl.String, strict=False), extract).alias("value"),
            )
        )
    keys = pl.concat(parts, how="vertical")
    return keys.filter(pl.col("value") != "")


def _value_expr(column: pl.Expr, extract: Any) -> pl.Expr:
    """Build the extractor as a Polars expression.

    Each strategy is expressed with native string operations so the whole key
    table is built vectorised; the Python callables above exist only as the
    readable reference implementation (used by the unit tests to prove the two
    agree).
    """
    if extract is normalized_ref:
        return (
            column.str.to_lowercase()
            .str.replace_all("_", " ")
            .str.replace_all("-", " ")
            .str.replace_all("::", " ")
            .str.replace_all("/", " ")
            .str.replace_all(".", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )
    if extract is digits_of:
        return column.str.replace_all(r"\D", "")
    if extract is digits_suffix:
        return (
            column.str.replace_all(r"\D", "")
            .str.replace_all(r"^(\d*?)(\d{3})$", "$2")
        )
    if extract.__name__ == "_prefix3":
        return column.str.replace_all(r"[^a-z0-9]", "").str.slice(0, 3)
    if extract.__name__ == "_bigram":
        return column.str.replace_all(r"[^a-z0-9]", "").str.slice(0, 2)
    if extract.__name__ == "_anchor":
        compact = column.str.replace_all(r"[^a-z0-9]", "")
        return (
            pl.when(compact.str.len_chars() == 0)
            .then(pl.lit(""))
            .otherwise(
                compact.str.slice(0, 1)
                + compact.str.slice(-1, 1)
                + (compact.str.len_chars().clip(0, 20) // 4).cast(pl.Utf8)
            )
        )
    raise ValueError(f"unhandled extractor: {extract}")  # pragma: no cover


def generate_candidates(
    frame: pl.DataFrame,
    *,
    ref_col: str = "entity_ref_norm",
    id_col: str = "ref_id",
    strategies: Iterable[Strategy] = STRATEGIES,
    max_block_size: int = 400,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Build the cross-source candidate pair set.

    Only pairs from *different* sources are candidates: two references inside one
    source are the same source's own bookkeeping, not a resolution decision.

    Returns ``(pairs, stats)`` where pairs has one row per unordered candidate with
    the strategy (or strategies) that produced it.
    """
    keys = build_keys(frame, ref_col=ref_col, id_col=id_col)
    keys = keys.filter(pl.col("strategy").is_in(list(strategies)))

    meta = frame.select(pl.col(id_col), pl.col("source_id"), pl.col(ref_col).alias("ref_norm"))
    keyed = keys.join(meta, on=id_col, how="left")

    # 'value' is already a scalar string per row (one row per strategy per
    # reference), so there is nothing to explode - the loop below is the grouping.
    buckets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for strategy, value, ref_id, source_id in keyed.select(
        "strategy", "value", id_col, "source_id"
    ).iter_rows():
        buckets[(strategy, value)].add(f"{source_id}\x1f{ref_id}")

    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    oversized = 0
    for (strategy, _value), members in buckets.items():
        if len(members) < 2:
            continue
        if len(members) > max_block_size:
            # A block this large carries no information (a 3-char prefix shared by
            # thousands of refs) and would dominate cost. Dropping it is recorded so
            # the blocking recall of the report stays honest.
            oversized += 1
            continue
        ordered = sorted(members)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                ls, lid = left.split("\x1f")
                rs, rid = right.split("\x1f")
                if ls == rs:
                    continue
                pairs[(lid, rid)].add(strategy)

    if not pairs:
        return pl.DataFrame(
            schema={
                "left_ref_id": pl.Utf8,
                "right_ref_id": pl.Utf8,
                "left_source_id": pl.Utf8,
                "right_source_id": pl.Utf8,
                "left_ref": pl.Utf8,
                "right_ref": pl.Utf8,
                "strategies": pl.Utf8,
                "n_strategies": pl.Int32,
            }
        ), {"blocks": len(buckets), "candidates": 0, "oversized_blocks": oversized}

    rows = [
        {
            "left_ref_id": lid,
            "right_ref_id": rid,
            "strategies": ",".join(sorted(strategies_for)),
            "n_strategies": len(strategies_for),
        }
        for (lid, rid), strategies_for in pairs.items()
    ]
    pairs_frame = pl.DataFrame(rows)
    # Carry the raw reference through explicitly. Deriving it from ``ref_id`` by
    # stripping a ``source::`` prefix is unsafe: a source id may itself contain
    # ``::`` (``site::qb042``), so a greedy strip silently truncates the reference
    # and every lookup misses.
    raw_of = dict(frame.select(pl.col(id_col), pl.col("raw_ref")).iter_rows())
    source_of = dict(frame.select(pl.col(id_col), pl.col("source_id")).iter_rows())
    pairs_frame = pairs_frame.with_columns(
        pl.col("left_ref_id").replace_strict(source_of, default=None).alias("left_source_id"),
        pl.col("right_ref_id").replace_strict(source_of, default=None).alias("right_source_id"),
        pl.col("left_ref_id").replace_strict(raw_of, default=None).alias("left_ref"),
        pl.col("right_ref_id").replace_strict(raw_of, default=None).alias("right_ref"),
    ).sort("left_ref_id", "right_ref_id")

    stats = {
        "blocks": len(buckets),
        "candidates": pairs_frame.height,
        "oversized_blocks": oversized,
        "references": frame.height,
        "max_block_size": max_block_size,
        "reduction_ratio": round(
            (frame.height * (frame.height - 1) / 2) / max(1, pairs_frame.height), 3
        ),
    }
    return pairs_frame, stats
