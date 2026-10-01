"""Architecture and methodology figures for the MOSAIC README.

These are structural diagrams, not data charts. Each one makes a claim that
prose tends to blur:

* :func:`architecture` — the whole pipeline, and which stages do not exist yet.
* :func:`research_workflow` — how a research question becomes an experiment.
* :func:`data_integration` — why one clean table is not the starting point.
* :func:`temporal_leakage` — information may not flow backward through time.
* :func:`graph_methodology` — why a global graph is forbidden.
* :func:`evidence_lineage` — score back to source record.

Nothing here reads experiment results. The figures describe *design*, so they
cannot drift from the numbers the way a chart regenerated from a stale artifact
would. The one place a measured value appears is the declared-implementation
band in :func:`architecture`, and that band is passed in explicitly rather than
imported from a results directory.

Style follows the author's other project READMEs so the three read as one body of
work: white ground, no top/right spines, the claim as a bold left-aligned title,
a restrained categorical palette, and no decoration that does not carry meaning.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Palette. Deliberately small: a figure that needs six hues to be read is a
# figure with too much in it.
BLUE = "#2f6fb5"  # implemented stages
ORANGE = "#d97b29"  # the thing being emphasised
GREY = "#8b93a1"  # not yet implemented
INK = "#1a1a1a"
RED = "#c0392b"  # forbidden flow
GREEN = "#2e8b57"  # permitted flow
BAND = "#f2f4f7"  # stage grouping band

# Stage types, mapped to a fill. Adding a type means deciding what it means.
IMPLEMENTED = "implemented"
PLANNED = "planned"
FORBIDDEN = "forbidden"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "axes.grid": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
        }
    )


def _fill(kind: str) -> str:
    return {
        IMPLEMENTED: "#ffffff",
        PLANNED: "#ffffff",
        FORBIDDEN: "#ffffff",
    }[kind]


def _edge(kind: str) -> str:
    return {
        IMPLEMENTED: BLUE,
        PLANNED: GREY,
        FORBIDDEN: RED,
    }[kind]


def _pt(ax, points: float) -> float:
    """Convert a font size in points into axes units.

    Axes units are not points: on an 11x5 inch figure one point is roughly 0.0026
    axes units. Hard-coding 1/72 overstates every text height by more than 5x and
    silently pushes captions outside their boxes, so the scale is read from the
    live transform instead.
    """
    fig = ax.figure
    fig.canvas.draw()
    y0 = ax.transData.inverted().transform((0.0, 0.0))[1]
    y1 = ax.transData.inverted().transform((0.0, 100.0))[1]
    return abs(y1 - y0) / 100.0 * points


def _text_height(ax, size: float) -> float:
    """Height in axes units of one rendered line at ``size`` points.

    Measured from the renderer rather than derived from the em square: matplotlib's
    text extent includes ascent and descent and is noticeably taller than
    size/72 in axes units. Estimating it put captions on top of their headings.
    """
    fig = ax.figure
    probe = ax.text(
        0.0,
        0.0,
        "Ag",
        fontsize=size,
        ha="left",
        va="baseline",
        alpha=0.0,
    )
    fig.canvas.draw()
    bb = probe.get_window_extent()
    inv = ax.transData.inverted()
    (_, y0) = inv.transform((bb.x0, bb.y0))
    (_, y1) = inv.transform((bb.x1, bb.y1))
    probe.remove()
    return abs(y1 - y0)


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    kind: str = IMPLEMENTED,
    *,
    weight: str = "normal",
    size: float = 8.0,
    style: str = "round,pad=0.02,rounding_size=0.02",
    zorder: int = 3,
) -> None:
    """One stage box. Label may contain newlines; centred on the box."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=style,
            linewidth=1.3,
            edgecolor=_edge(kind),
            facecolor=_fill(kind),
            zorder=zorder,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        color=INK if kind != PLANNED else "#5c6470",
        linespacing=1.35,
        zorder=zorder + 1,
    )


def _arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str = "#4a5058",
    style: str = "-|>",
    lw: float = 1.15,
    rad: float = 0.0,
    ls: str = "-",
    zorder: int = 2,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            mutation_scale=9,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=ls,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=zorder,
        )
    )


def _title(fig, text: str, *, y: float = 0.975) -> None:
    """The claim, as a left-aligned bold title. Not a neutral label."""
    fig.text(0.012, y, text, ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK)


def _subtitle(fig, text: str, *, y: float = 0.928) -> None:
    fig.text(0.012, y, text, ha="left", va="top", fontsize=8.6, color="#5c6470")


def _band(ax, x: float, y: float, w: float, h: float, label: str) -> None:
    """A labelled grouping band behind the stages it contains."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0.9,
            edgecolor="#c9cfd8",
            facecolor=BAND,
            zorder=0,
        )
    )
    ax.text(
        x + 0.008,
        y + h - 0.028,
        label.upper(),
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color="#7c848f",
        zorder=1,
    )


def _legend(ax, entries: list[tuple[str, str]], *, loc: str = "lower left") -> None:
    handles = [
        mpatches.Patch(facecolor="#ffffff", edgecolor=color, linewidth=1.3, label=label)
        for label, color in entries
    ]
    leg = ax.legend(
        handles=handles,
        loc=loc,
        frameon=False,
        fontsize=7.6,
        handlelength=1.5,
        handleheight=1.0,
        labelspacing=0.35,
    )
    for text in leg.get_texts():
        text.set_color("#5c6470")


def _finish(fig, ax, out_path: Path) -> str:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return str(out_path)


# --------------------------------------------------------------------- figures
def _band_around(
    ax,
    boxes: list[tuple[float, float, float]],
    *,
    label: str,
    pad_x: float = 0.016,
    pad_top: float = 0.052,
    pad_bot: float = 0.030,
) -> tuple[float, float, float]:
    """Draw a band that provably contains the boxes given.

    Takes each box's own (x, y, h) so the band is derived from the geometry rather
    than from separately hard-coded numbers. A box row that drifts outside its own
    group shading is otherwise invisible until someone looks at the PNG.
    """
    left = min(x for x, _, _ in boxes) - pad_x
    right = max(x + w for x, w, _ in boxes) + pad_x
    bottom = min(y for _, y, _ in boxes) - pad_bot
    top = max(y + h for _, y, h in boxes) + pad_top
    ax.add_patch(
        FancyBboxPatch(
            (left, bottom),
            right - left,
            top - bottom,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0.9,
            edgecolor="#c9cfd8",
            facecolor=BAND,
            zorder=0,
        )
    )
    ax.text(
        left + 0.006,
        top - 0.014,
        label.upper(),
        ha="left",
        va="top",
        fontsize=6.8,
        fontweight="bold",
        color="#8b93a1",
        zorder=1,
    )
    return left, right, bottom


def _row(
    n: int,
    *,
    gap: float,
    x0: float = 0.028,
    right_margin: float = 0.030,
) -> tuple[float, float]:
    """Width and stride for a row of ``n`` boxes that must fit the canvas.

    Computing the width from the count is what keeps a six-box row from running
    off the right edge while a four-box row leaves a gap.
    """
    usable = 1.0 - x0 - right_margin
    width = (usable - (n - 1) * gap) / n
    return width, width + gap


def _multiline_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    head: str,
    note: str,
    *,
    kind: str = IMPLEMENTED,
    head_size: float = 8.0,
    note_size: float = 6.6,
    zorder: int = 3,
) -> None:
    """A box whose explanatory note lives INSIDE it.

    Notes placed beside a box are inevitably crossed by the connector that runs past
    them, so they go inside where nothing can overlap them.

    ``h`` is treated as a MINIMUM: the box grows when the note wraps to more lines
    than the height allows. Sizing a box to a fixed height while its caption wraps
    to three lines is what pushes the last line through the bottom border.
    """
    lines = note.count("\n") + 1 if note else 0
    # Measured, not estimated: a single 6.6pt line occupies ~1.8x the em box, so
    # deriving the height from font size alone understates it and overlaps the text.
    line_h = _text_height(ax, note_size) * 1.45
    head_h = _text_height(ax, head_size) * 1.30
    pad = _pt(ax, 4.0)
    block_h = head_h + (lines * line_h if lines else 0.0)
    needed = block_h + 2 * pad
    h = max(h, needed)
    # Centre the head+note block vertically. Anchoring the note to the box bottom
    # instead makes a box that is TALLER than needed push its own caption outside.
    block_bottom = y + (h - block_h) / 2

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.002,rounding_size=0.016",
            linewidth=1.3,
            edgecolor=_edge(kind),
            facecolor=_fill(kind),
            zorder=zorder,
        )
    )
    if lines:
        ax.text(
            x + w / 2,
            block_bottom + block_h - head_h / 2,
            head,
            ha="center",
            va="center",
            fontsize=head_size,
            fontweight="bold",
            color=INK if kind != PLANNED else "#5c6470",
            zorder=zorder + 1,
        )
        # va="top", not "center": matplotlib expands a multi-line block upward from
        # the anchor, so centring it places the first line over the heading.
        ax.text(
            x + w / 2,
            block_bottom + head_h + lines * line_h,
            note,
            ha="center",
            va="top",
            fontsize=note_size,
            color="#7c848f",
            linespacing=1.42,
            zorder=zorder + 1,
        )
    else:
        ax.text(
            x + w / 2,
            block_bottom + block_h / 2,
            head,
            ha="center",
            va="center",
            fontsize=head_size,
            fontweight="bold",
            color=INK if kind != PLANNED else "#5c6470",
            zorder=zorder + 1,
        )


def architecture(out_path: str | Path) -> str:
    """Full MOSAIC pipeline, marking which stages exist and which do not.

    The grey half is the point of the figure: a reader should see at a glance that
    the implemented system stops at features and graphs.
    """
    _style()
    fig, ax = plt.subplots(figsize=(11.4, 7.0))
    _title(fig, "MOSAIC: sources in, ranked anomalies and traceable evidence out")
    _subtitle(
        fig,
        "Blue stages are implemented, executed and tested. Grey stages are designed for but not yet built.",
    )

    H = 0.098

    def row(y: float, items: list[tuple[str, str, str]], gap: float, size: float) -> list[tuple[float, float, float]]:
        w, stride = _row(len(items), gap=gap)
        out = []
        for i, (head, note, kind) in enumerate(items):
            x = 0.028 + i * stride
            _multiline_box(ax, x, y, w, H, head, note, kind=kind, head_size=size, note_size=size - 1.6)
            out.append((x, y, w))
        return out

    # --- row 1: four deliberately incompatible sources
    src_boxes = row(
        0.762,
        [
            ("transit_feed", "snake_case fields\nepoch-second stamps", IMPLEMENTED),
            ("sensor_grid", "camelCase fields\nlocal timezone", IMPLEMENTED),
            ("ops_log", "free-text categories\narrival-time ordered", IMPLEMENTED),
            ("billing_extract", "prefixed identifiers\ndelayed ~2 days", IMPLEMENTED),
        ],
        gap=0.030,
        size=8.2,
    )
    _band_around(ax, src_boxes, label="Four deliberately incompatible sources")

    # --- row 2: ingestion through splits
    pipe_boxes = row(
        0.556,
        [
            ("Adapters", "native → canonical", IMPLEMENTED),
            ("Validation", "0 FAIL, q=0.95", IMPLEMENTED),
            ("Cleaning", "named steps + ledger", IMPLEMENTED),
            ("Entity resolution", "P=0.996, R=0.997", IMPLEMENTED),
            ("Splits", "chronological, one source", IMPLEMENTED),
        ],
        gap=0.024,
        size=7.9,
    )
    _band_around(ax, pipe_boxes, label="Data layer — implemented and executed")
    _, pipe_w, _ = pipe_boxes[0]
    for i in range(1, 5):
        x0 = pipe_boxes[i - 1][0]
        _arrow(ax, x0 + pipe_w, 0.556 + H / 2, pipe_boxes[i][0], 0.556 + H / 2)

    # sources merge on a bus, then one arrow into the adapters box. Drawn as a bus
    # rather than four diagonals so no connector has to cross the source labels.
    bus_y = 0.700
    adapters_cx = pipe_boxes[0][0] + pipe_w / 2
    for x, _y, w in src_boxes:
        cx = x + w / 2
        ax.plot([cx, cx], [0.762 - 0.030, bus_y], color="#c2c8d1", linewidth=1.0, zorder=1)
        ax.plot([cx, adapters_cx], [bus_y, bus_y], color="#c2c8d1", linewidth=1.0, zorder=1)
    _arrow(ax, adapters_cx, bus_y, adapters_cx, 0.556 + H)

    # --- row 3: research layer
    res_boxes = row(
        0.330,
        [
            ("Features", "8 causal families", IMPLEMENTED),
            ("Temporal graph", "26.6 s / 498k rows", IMPLEMENTED),
            ("Feature registry", "lineage + leakage class", IMPLEMENTED),
            ("Baselines", "statistical", PLANNED),
            ("ML models", "gradient boosting", PLANNED),
        ],
        gap=0.024,
        size=7.9,
    )
    _band_around(ax, res_boxes, label="Research layer — implemented and executed")
    res_w = res_boxes[0][1]
    for i in range(1, 5):
        color = BLUE if i < 3 else GREY
        _arrow(ax, res_boxes[i - 1][0] + res_w, 0.330 + H / 2, res_boxes[i][0], 0.330 + H / 2, color=color)

    # Splits feeds the implemented research row. Both targets get a real down-arrow
    # into the box top; the earlier version left one dangling in empty space.
    splits_cx = pipe_boxes[4][0] + pipe_w / 2
    features_cx = res_boxes[0][0] + res_w / 2
    graph_cx = res_boxes[1][0] + res_w / 2
    feed_y = 0.470
    ax.plot([splits_cx, splits_cx], [0.556 - 0.030, feed_y], color=BLUE, linewidth=1.15, zorder=2)
    ax.plot([splits_cx, graph_cx], [feed_y, feed_y], color=BLUE, linewidth=1.15, zorder=2)
    ax.plot([features_cx, graph_cx], [feed_y, feed_y], color=BLUE, linewidth=1.15, zorder=2)
    _arrow(ax, graph_cx, feed_y, graph_cx, 0.330 + H, color=BLUE)
    _arrow(ax, features_cx, feed_y, features_cx, 0.330 + H, color=BLUE)

    # --- row 4: delivery, all planned
    del_boxes = row(
        0.078,
        [
            ("Detection", "statistical + ML", PLANNED),
            ("Fusion", "multi-evidence", PLANNED),
            ("Explainability", "why this alert", PLANNED),
            ("Evidence store", "score → events", PLANNED),
            ("Search API", "queryable", PLANNED),
            ("Analytical UI", "analyst-facing", PLANNED),
        ],
        gap=0.018,
        size=7.4,
    )
    _band_around(ax, del_boxes, label="Delivery — designed for, not yet built", pad_bot=0.034)
    del_w = del_boxes[0][1]
    for i in range(1, 6):
        _arrow(ax, del_boxes[i - 1][0] + del_w, 0.078 + H / 2, del_boxes[i][0], 0.078 + H / 2, color=GREY)

    # The delivery row is fed from the models, not from the implemented baselines:
    # the pipeline it depends on is the planned one.
    models_cx = res_boxes[4][0] + res_w / 2
    detection_cx = del_boxes[0][0] + del_w / 2
    hand_y = 0.222
    ax.plot([models_cx, models_cx], [0.330 - 0.030, hand_y], color=GREY, linewidth=1.15, zorder=2)
    ax.plot([models_cx, detection_cx], [hand_y, hand_y], color=GREY, linewidth=1.15, zorder=2)
    _arrow(ax, detection_cx, hand_y, detection_cx, 0.078 + H, color=GREY)

    _legend(
        ax,
        [
            ("implemented, executed, tested", BLUE),
            ("designed for, not yet built", GREY),
        ],
        loc="upper right",
    )
    ax.set_ylim(-0.03, 1.0)
    return _finish(fig, ax, Path(out_path))


def research_workflow(out_path: str | Path) -> str:
    """How a research question becomes a defensible result.

    The loop back to the research question is deliberate: an experiment that
    cannot change the framing was not worth running.
    """
    _style()
    fig, ax = plt.subplots(figsize=(10.4, 3.5))
    _title(fig, "From research question to a claim that survives its own ablations")
    _subtitle(
        fig,
        "Every stage produces a recorded artifact. Stages in grey have no artifact yet.",
    )

    steps = [
        ("Research\nquestion", ORANGE),
        ("Dataset\nsynthetic + public", BLUE),
        ("Protocol\nchronological\nsplits", BLUE),
        ("Baselines\nstatistical + ML", GREY),
        ("Intervention\nadd family\nor source", GREY),
        ("Out-of-time\nevaluation", GREY),
        ("Ablation\nwhat caused\nit", GREY),
        ("Robustness\nER noise,\noutages", GREY),
        ("Report\nincl. negative\nresults", GREY),
    ]
    w, gap, y, h = 0.098, 0.0125, 0.30, 0.30
    for i, (label, color) in enumerate(steps):
        x = 0.012 + i * (w + gap)
        kind = IMPLEMENTED if color == BLUE else (PLANNED if color == GREY else IMPLEMENTED)
        edge = ORANGE if color == ORANGE else _edge(kind)
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.006,rounding_size=0.02",
                linewidth=1.4,
                edgecolor=edge,
                facecolor="#ffffff",
                zorder=3,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=7.3,
            fontweight="bold" if color == ORANGE else "normal",
            color=INK if color != GREY else "#5c6470",
            linespacing=1.35,
            zorder=4,
        )
        if i < len(steps) - 1:
            _arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)

    # the loop back: a result either supports the framing or revises it
    last_x = 0.012 + (len(steps) - 1) * (w + gap)
    _arrow(ax, last_x + w / 2, y, last_x + w / 2, 0.135, color="#4a5058", rad=0.0)
    _arrow(ax, last_x + w / 2, 0.135, 0.061, 0.135, color="#4a5058")
    _arrow(ax, 0.061, 0.135, 0.061, y, color=ORANGE)
    ax.text(
        0.500,
        0.108,
        "a result that cannot change the framing was not worth running",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#5c6470",
        style="italic",
    )

    _legend(
        ax,
        [("done", BLUE), ("planned — no artifact yet", GREY)],
        loc="lower left",
    )
    return _finish(fig, ax, Path(out_path))


def data_integration(out_path: str | Path) -> str:
    """Why the pipeline starts from incompatible sources.

    The point is not that the sources differ in field names — they disagree about
    identity, time and units, which is what makes integration a modelling decision
    rather than a concatenation.
    """
    _style()
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    _title(fig, "Four sources that disagree about identity, time and units")
    _subtitle(
        fig,
        "Canonicalization is where information is discarded. Raw records are kept so every decision stays reversible.",
    )

    H = 0.150

    # --- four incompatible sources, full width so no caption can be occluded
    src_y = 0.700
    src = [
        ("transit_feed", "snake_case fields\nepoch-second stamps\nno location"),
        ("sensor_grid", "camelCase fields\nlocal timezone\nimprecise minutes"),
        ("ops_log", "free-text categories\narrival-time ordered\nno units"),
        ("billing_extract", "prefixed identifiers\ndelayed ~2 days\nmoney as text"),
    ]
    w, stride = _row(len(src), gap=0.036)
    src_boxes: list[tuple[float, float, float]] = []
    for i, (head, note) in enumerate(src):
        x = 0.028 + i * stride
        _multiline_box(ax, x, src_y, w, H, head, note, head_size=8.2, note_size=6.6)
        src_boxes.append((x, src_y, w))
    _band_around(ax, src_boxes, label="Native schemas — mutually incompatible", pad_bot=0.030)

    # --- merge on a bus below the sources so no connector crosses a caption
    bus_y = src_y - 0.058
    adapters_w = 0.300
    adapters_x = 0.028 + (1.0 - 0.028 - 0.030 - adapters_w) / 2
    adapters_cx = adapters_x + adapters_w / 2
    for x, _y, bw in src_boxes:
        cx = x + bw / 2
        ax.plot([cx, cx], [src_y - 0.030, bus_y], color="#c2c8d1", linewidth=1.0, zorder=1)
        ax.plot([cx, adapters_cx], [bus_y, bus_y], color="#c2c8d1", linewidth=1.0, zorder=1)
    _arrow(ax, adapters_cx, bus_y, adapters_cx, src_y - 0.098, color=BLUE)

    # --- adapters
    ad_y = src_y - 0.200
    _multiline_box(
        ax,
        adapters_x,
        ad_y,
        adapters_w,
        0.100,
        "Adapters",
        "the only place that knows\neach source's conventions",
        head_size=8.2,
        note_size=6.7,
    )
    _arrow(ax, adapters_cx, ad_y, adapters_cx, ad_y - 0.050, color=BLUE)

    # --- canonical representation, then entity resolution, then the unified view.
    # Three columns on one baseline. The earlier version overlapped them in BOTH
    # axes, which put the ER title through the canonical box's border.
    canon_y = ad_y - 0.185
    col_w, col_gap = 0.290, 0.035
    col_x = [0.028 + i * (col_w + col_gap) for i in range(3)]
    for cx_, cw_ in zip(col_x, [col_w] * 3, strict=True):
        assert cx_ + cw_ <= 1.0 - 0.030 + 1e-9, "column escapes canvas"

    _multiline_box(
        ax,
        col_x[0],
        canon_y,
        col_w,
        0.155,
        "Canonical representation",
        "event_id · timestamp\nentity_ref_norm · event_type\nsource_id · provenance kept",
        head_size=7.8,
        note_size=6.4,
    )
    _multiline_box(
        ax,
        col_x[1],
        canon_y,
        col_w,
        0.155,
        "Entity resolution",
        "blocking → match → cluster\nsame entity, four name styles",
        head_size=7.8,
        note_size=6.4,
    )
    _multiline_box(
        ax,
        col_x[2],
        canon_y,
        col_w,
        0.155,
        "Unified analytical view",
        "504 clusters\nfor 500 true entities\nP=0.996, R=0.997",
        head_size=7.8,
        note_size=6.4,
    )
    _arrow(ax, col_x[0] + col_w, canon_y + 0.0775, col_x[1], canon_y + 0.0775, color=BLUE)
    _arrow(ax, col_x[1] + col_w, canon_y + 0.0775, col_x[2], canon_y + 0.0775, color=ORANGE)

    ax.text(
        0.500,
        0.040,
        "Every arrow above is a modelling decision: a time origin, an identity rule, a unit convention.",
        ha="center",
        va="center",
        fontsize=7.8,
        color=RED,
        style="italic",
    )
    ax.set_ylim(0.01, 1.0)
    return _finish(fig, ax, Path(out_path))


def temporal_leakage(out_path: str | Path) -> str:
    """The temporal protocol, and the flow that is forbidden.

    Red is used exactly once, on the arrow that must never be taken. That is the
    only thing this figure needs the reader to remember.
    """
    _style()
    fig, ax = plt.subplots(figsize=(10.6, 4.3))
    _title(fig, "Information may flow forward through time, and only forward")
    _subtitle(
        fig,
        "A fitted artifact carries the period it was fitted on and raises rather than warns if applied to an earlier one.",
    )

    stages = [
        ("Backtest", "inside the train period", BLUE),
        ("Train", "fit baselines and models", BLUE),
        ("Validation", "select thresholds", BLUE),
        ("Forward / holdout", "scored only,\nnothing is fitted here", ORANGE),
    ]

    # Box height follows the longest note, and the title always sits above the
    # note block with a guaranteed gap. Sizing boxes to a fixed height while the
    # note wraps to two lines is what puts the title through the text.
    notes = [note.count("\n") + 1 for _, note, _ in stages]
    max_lines = max(notes)
    h = 0.150 + 0.030 * (max_lines - 1)
    y = 0.520
    w, stride = _row(len(stages), gap=0.058)
    centres: list[float] = []
    for i, (name, note, color) in enumerate(stages):
        x = 0.028 + i * stride
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.004,rounding_size=0.018",
                linewidth=1.4,
                edgecolor=color,
                facecolor="#ffffff",
                zorder=3,
            )
        )
        ax.text(
            x + w / 2,
            y + h - 0.040,
            name,
            ha="center",
            va="center",
            fontsize=8.4,
            fontweight="bold",
            color=INK,
            zorder=4,
        )
        # Anchor the note to the BOTTOM of the box and grow the box upward for
        # extra lines. Offsetting the note by line count instead let a two-line
        # note push up into the title.
        ax.text(
            x + w / 2,
            y + 0.022,
            note,
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#5c6470",
            linespacing=1.45,
            zorder=4,
        )
        centres.append(x + w / 2)
        if i < len(stages) - 1:
            _arrow(ax, x + w, y + h / 2, x + stride, y + h / 2, color=color)

    # timeline spine
    axis_y = 0.215
    ax.annotate(
        "",
        xy=(0.972, axis_y),
        xytext=(0.028, axis_y),
        arrowprops={"arrowstyle": "-|>", "color": "#c9cfd8", "linewidth": 2.2},
        zorder=1,
    )
    for cx in centres:
        ax.plot([cx, cx], [axis_y, y - 0.006], color="#d5dae1", linewidth=1.0, zorder=1)
    ax.text(0.028, axis_y - 0.055, "earliest", ha="left", va="center", fontsize=7.0, color="#9aa2ad")
    ax.text(0.972, axis_y - 0.055, "latest", ha="right", va="center", fontsize=7.0, color="#9aa2ad")

    # the forbidden flow: forward period back into the period that was fitted on it
    _arrow(ax, centres[3], y + h, centres[2], y + h, color=RED, ls=(0, (4.5, 2.5)), lw=1.6, rad=-0.50)
    ax.text(
        0.500,
        y + h + 0.115,
        "forbidden — fitting on the forward period, or applying a forward-fitted artifact backwards",
        ha="center",
        va="center",
        fontsize=7.9,
        fontweight="bold",
        color=RED,
    )
    ax.text(
        0.500,
        0.075,
        "Random train/test splits are not used as a primary evaluation method anywhere in this project.",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#5c6470",
        style="italic",
    )
    ax.set_ylim(0.02, 1.0)
    return _finish(fig, ax, Path(out_path))


def graph_methodology(out_path: str | Path) -> str:
    """Temporal graph construction, and why a global graph is forbidden.

    The top band is the construction MOSAIC uses. The bottom band is the tempting
    shortcut, drawn and marked, because the reason it leaks is not obvious from the
    code that avoids it.
    """
    _style()
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    _title(fig, "A snapshot at T contains only edges first seen at or before T")
    _subtitle(
        fig,
        "Each edge stores the time it was first observed. Historical measures are computed by replaying edges in timestamp order.",
    )

    H = 0.130

    def chain(y: float, items: list[tuple[str, str, str]], gap: float) -> list[tuple[float, float, float]]:
        w, stride = _row(len(items), gap=gap)
        out = []
        for i, (head, note, kind) in enumerate(items):
            x = 0.028 + i * stride
            _multiline_box(ax, x, y, w, H, head, note, kind=kind, head_size=7.7, note_size=6.5)
            out.append((x, y, w))
            if i:
                _arrow(
                    ax,
                    out[i - 1][0] + out[i - 1][2],
                    y + H / 2,
                    x,
                    y + H / 2,
                    color=BLUE if kind != FORBIDDEN else RED,
                )
        return out

    # --- permitted construction
    top_y = 0.700
    top_boxes = chain(
        top_y,
        [
            ("Events", "canonical entity_id\nand peers", IMPLEMENTED),
            ("Edge list", "first_seen = first\nobservation time", IMPLEMENTED),
            ("Replay to T", "keep first_seen ≤ T", IMPLEMENTED),
            ("Measures", "degree · PageRank\nbetweenness · community", IMPLEMENTED),
        ],
        gap=0.030,
    )
    _band_around(ax, top_boxes, label="Temporal construction — what MOSAIC does", pad_bot=0.105)
    graph_cx = top_boxes[2][0] + top_boxes[2][2] / 2
    # Cap the summary bar so it cannot run flush against the band edge.
    summary_w = 0.560
    summary_x = graph_cx - summary_w / 2
    summary_y = top_y - 0.086
    _multiline_box(
        ax,
        summary_x,
        summary_y,
        summary_w,
        0.052,
        "graph_* features attributed only to events inside that cutoff",
        "",
        head_size=7.3,
    )
    _arrow(ax, graph_cx, top_y - 0.004, graph_cx, summary_y + 0.052, color=BLUE)

    # --- forbidden construction
    bot_y = 0.240
    bot_boxes = chain(
        bot_y,
        [
            ("All events", "every period at once", FORBIDDEN),
            ("One global graph", "over the full span", FORBIDDEN),
            ("Measure it whole", "then ask for\npast values", FORBIDDEN),
            ("Silent leak", "degree, PageRank and\ncommunity contain edges\nfirst seen LATER", FORBIDDEN),
        ],
        gap=0.030,
    )
    _band_around(ax, bot_boxes, label="Forbidden — the shortcut that leaks", pad_bot=0.086)

    # The one-line reason sits above the band label, never across a box top.
    ax.text(
        0.500,
        0.512,
        "A relationship observed in the future cannot inform a historical measurement.",
        ha="center",
        va="center",
        fontsize=8.0,
        color=RED,
        style="italic",
        fontweight="bold",
    )

    _legend(
        ax,
        [
            ("permitted construction", BLUE),
            ("forbidden: future edges reach the past", RED),
        ],
        loc="lower left",
    )
    ax.set_ylim(0.02, 1.0)
    return _finish(fig, ax, Path(out_path))


def evidence_lineage(out_path: str | Path) -> str:
    """The traceability chain, both directions.

    Reading right-to-left is the audit path a reviewer takes; left-to-right is the
    path an analyst takes when investigating an alert.
    """
    _style()
    fig, ax = plt.subplots(figsize=(11.0, 3.6))
    _title(fig, "Any anomaly walks back to its source records and forward to the experiment that measured it")
    _subtitle(
        fig,
        "The evidence store is the join between a score and the rows that produced it. It is not implemented yet.",
    )

    stages = [
        ("Source\nrecord", BLUE),
        ("Canonical\nevent", BLUE),
        ("Feature value\n+ registry\nfingerprint", BLUE),
        ("Model", GREY),
        ("Score", GREY),
        ("Anomaly", GREY),
        ("Explanation\ncontributing\nfeatures", GREY),
        ("Experiment\nrecord", GREY),
    ]
    w, gap, y, h = 0.104, 0.0212, 0.400, 0.270
    for i, (label, color) in enumerate(stages):
        x = 0.012 + i * (w + gap)
        kind = IMPLEMENTED if color == BLUE else PLANNED
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.006,rounding_size=0.02",
                linewidth=1.35,
                edgecolor=_edge(kind),
                facecolor="#ffffff",
                zorder=3,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=7.2,
            color=INK if kind == IMPLEMENTED else "#5c6470",
            linespacing=1.4,
            zorder=4,
        )
        if i < len(stages) - 1:
            _arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2, color=GREY if kind == PLANNED else BLUE)

    ax.text(
        0.500,
        0.255,
        "analyst investigates an alert  ←",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#5c6470",
    )
    ax.text(
        0.500,
        0.150,
        "→  reviewer audits a claim: same fingerprint, same feature definitions",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#5c6470",
    )
    _legend(
        ax,
        [("implemented", BLUE), ("designed for, not yet built", GREY)],
        loc="lower left",
    )
    return _finish(fig, ax, Path(out_path))


ALL_FIGURES = {
    "architecture": architecture,
    "research-workflow": research_workflow,
    "data-integration": data_integration,
    "temporal-leakage": temporal_leakage,
    "graph-methodology": graph_methodology,
    "evidence-lineage": evidence_lineage,
}


def render_all(out_dir: str | Path) -> list[str]:
    """Render every README figure. Returns the paths written."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return [fn(directory / f"{name}.png") for name, fn in ALL_FIGURES.items()]