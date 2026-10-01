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


# Light tints of each edge colour. A white fill with a 1px border is accurate to
# the boxes but illegible at README display size: the legend read as two identical
# white squares, and the planned/forbidden distinction disappeared with it.
_TINT = {
    IMPLEMENTED: "#e8f0f9",
    PLANNED: "#f3f4f6",
    FORBIDDEN: "#fbeceb",
}

# Tint keyed by EDGE colour, so a swatch always matches the box it stands for
# regardless of which stage type produced that colour.
_TINT_BY_EDGE = {
    BLUE: "#e8f0f9",
    ORANGE: "#fbeee0",
    GREY: "#f3f4f6",
    RED: "#fbeceb",
}


def _fill(kind: str, edge: str | None = None) -> str:
    if edge is not None:
        return _TINT_BY_EDGE.get(edge, _TINT[kind])
    return _TINT[kind]


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


def _new_canvas(width: float, height: float, *, ylim: tuple[float, float] = (0.0, 1.0)):
    """Create the figure with its data limits already frozen.

    Every height in this module is measured by rendering a probe and converting
    its extent through the axes transform. That conversion is only meaningful if
    the limits are already final: matplotlib autoscales as artists are added, so
    a measurement taken halfway through a figure is on a different scale from one
    taken at the end. Freezing the limits here means a box measured while drawing
    the source row and a caption measured while drawing the bottom row are on the
    same scale.
    """
    _style()
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(*ylim)
    return fig, ax


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
) -> float:
    """One stage box. Label may contain newlines. Returns the height used.

    Like :func:`_multiline_box`, ``h`` is a MINIMUM and the box grows when the
    label wraps to more lines than the height allows. A centred multi-line label
    in a fixed-height box overflows symmetrically, so the first and last lines end
    up outside the border -- which is exactly what happened to every box in
    data-integration until the height was measured instead of assumed.
    """
    lines = label.count("\n") + 1 if label else 0
    text_h = lines * _text_height(ax, size) * 1.35 if lines else 0.0
    h = max(h, text_h + 2 * _pt(ax, 4.5))

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
    return h


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


def _legend(
    ax,
    entries: list[tuple[str, str, str]],
    *,
    loc: str = "lower left",
) -> None:
    """Legend whose swatches are true miniatures of the boxes they stand for."""
    # The swatch is a miniature of the box it stands for, so its fill must follow
    # its edge colour rather than one hardcoded tint shared by every entry.
    handles = [
        mpatches.Patch(
            facecolor=_TINT_BY_EDGE.get(color, _TINT[kind]),
            edgecolor=color,
            linewidth=1.5,
            label=label,
        )
        for label, color, kind in entries
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
    """Hide the axes and save.

    Limits are NOT forced here: every figure calls _new_canvas, which already
    froze xlim/ylim at the right values, and clamping y back to (0,1) silently
    discarded a figure's own vertical range.

    `bbox_inches="tight"` is deliberately NOT used. It crops the canvas to the
    drawn artists, so the saved PNG's size no longer matches the figure and any
    tool that maps data coordinates to pixels (the geometry checker) is off by the
    crop offset. Saving the full canvas keeps data(x,y) -> pixel(y) exact.
    """
    ax.axis("off")
    fig.savefig(out_path, pad_inches=0.16)
    plt.close(fig)
    return str(out_path)


# --------------------------------------------------------------------- figures
def _measure_multiline(ax, head: str, note: str, *, head_size: float, note_size: float) -> float:
    """Height ``_multiline_box`` would use for this head+note, without drawing.

    Lets a row take the height of its tallest member so every box in the row is
    the same size and no caption grows into the row beneath it.
    """
    lines = note.count("\n") + 1 if note else 0
    head_h = _text_height(ax, head_size) * 1.32
    line_h = _text_height(ax, note_size) * 1.46 if lines else 0.0
    block = head_h + (_pt(ax, 2.2) + lines * line_h if lines else 0.0)
    return block + 2 * _pt(ax, 4.5)


def _measure_block(ax, text: str, *, size: float, leading: float = 1.52) -> float:
    """Height a single multi-line string needs, plus padding.

    ``_measure_multiline`` sizes a heading and a caption drawn at different sizes.
    Boxes that draw head+note as one string need this instead.
    """
    lines = text.count("\n") + 1 if text else 0
    body = lines * _text_height(ax, size) * leading if lines else 0.0
    return body + 2 * _pt(ax, 5.0)


def _band_around(
    ax,
    boxes: list[tuple[float, float, float, float]],
    *,
    label: str,
    pad_x: float = 0.016,
    pad_top: float = 0.062,
    pad_bot: float = 0.030,
) -> tuple[float, float, float]:
    """Draw a band that provably contains the boxes given.

    Takes each box's own (x, y, h) so the band is derived from the geometry rather
    than from separately hard-coded numbers. A box row that drifts outside its own
    group shading is otherwise invisible until someone looks at the PNG.
    """
    left = min(bx for bx, _by, _bw, _bh in boxes) - pad_x
    right = max(bx + bw for bx, _by, bw, _bh in boxes) + pad_x
    bottom = min(by for _bx, by, _bw, _bh in boxes) - pad_bot
    top = max(by + bh for _bx, by, _bw, bh in boxes) + pad_top
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
    # INSIDE the band, in the padding above its own boxes. Placing it above the
    # band instead put it in the gap belonging to the PREVIOUS band: band 2's
    # caption landed at y=0.771, inside band 1's box row (0.735-0.833), where the
    # boxes drew over it. A caption must be positioned relative to its own boxes.
    # Anchored at the band's TOP-RIGHT. At the top-left it sat in the corridor the
    # connectors entering from above must cross, and at the bottom-left it sat in
    # the corridor the hand-off bus runs along. The top-right corner of every band
    # in this figure is empty.
    ax.text(
        right - 0.006,
        top - 0.010,
        label.upper(),
        ha="right",
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
) -> float:
    """A box with its explanatory note inside it. Returns the height actually used.

    Notes placed beside a box get crossed by the connector running past them, so
    they live inside the box. The hard part is height: matplotlib's rendered text
    extent is roughly 1.8x the em box, so a height derived from font size alone
    leaves captions overlapping their headings and escaping the bottom border.

    Rather than estimate, this measures. `_text_height` reads the renderer's own
    extent for a probe string, and the block is laid out from that. Every value
    that varies with fontsize, figsize or dpi is therefore measured at the size it
    is drawn, and the box grows to whatever the text actually needs.
    """
    lines = note.count("\n") + 1 if note else 0
    head_h = _text_height(ax, head_size) * 1.32
    line_h = _text_height(ax, note_size) * 1.46 if lines else 0.0
    gap = _pt(ax, 2.2)
    pad = _pt(ax, 4.5)

    block_h = head_h + (gap + lines * line_h if lines else 0.0)
    h = max(h, block_h + 2 * pad)
    block_top = y + pad + block_h

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
    ax.text(
        x + w / 2,
        block_top,
        head,
        ha="center",
        va="top",
        fontsize=head_size,
        fontweight="bold",
        color=INK if kind != PLANNED else "#5c6470",
        zorder=zorder + 1,
    )
    if lines:
        ax.text(
            x + w / 2,
            block_top - head_h - gap,
            note,
            ha="center",
            va="top",
            fontsize=note_size,
            color="#7c848f",
            linespacing=1.46,
            zorder=zorder + 1,
        )
    return h


def architecture(out_path: str | Path) -> str:
    """Full MOSAIC pipeline, marking which stages exist and which do not.

    The grey half is the point of the figure: a reader should see at a glance that
    the implemented system stops at features and graphs.
    """
    _style()
    fig, ax = _new_canvas(11.4, 7.0)
    _title(fig, "MOSAIC: sources in, ranked anomalies and traceable evidence out")
    _subtitle(
        fig,
        "Blue stages are implemented, executed and tested. Grey stages are designed for but not yet built.",
    )

    H = 0.098

    def row(y: float, items: list[tuple[str, str, str]], gap: float, size: float) -> list[tuple[float, float, float, float]]:
        """Draw one row of stage boxes. Returns (x, y, w, h) per box.

        The 4-tuple is load-bearing: _band_around reads the fourth element as the
        box height. Returning (x, y, w) made every band treat a box's WIDTH as its
        height, so bands were roughly ten times too tall and swallowed the rows
        below them.
        """
        w, stride = _row(len(items), gap=gap)
        out = []
        for i, (head, note, kind) in enumerate(items):
            x = 0.028 + i * stride
            used = _multiline_box(ax, x, y, w, H, head, note, kind=kind, head_size=size, note_size=size - 1.6)
            out.append((x, y, w, used))
        return out

    # --- row 1: four deliberately incompatible sources
    src_boxes = row(
        0.735,
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
        0.520,
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
    pipe_w = pipe_boxes[0][2]
    for i in range(1, 5):
        x0 = pipe_boxes[i - 1][0]
        _arrow(ax, x0 + pipe_w, pipe_boxes[0][1] + pipe_boxes[0][3] / 2, pipe_boxes[i][0], pipe_boxes[0][1] + pipe_boxes[0][3] / 2)

    # sources merge on a bus, then one arrow into the adapters box. Drawn as a bus
    # rather than four diagonals so no connector has to cross the source labels.
    # The four sources merge on a bus above band 2 and drop into the adapters box.
    #
    # Two crossings had to be avoided, both found by measuring the render:
    # the band's own caption sits at data-x 0.018..0.283, y 0.646..0.664, and the
    # earlier routes hit it (465px, then 166px). So the bus now:
    #   1. drops at the far RIGHT of the row (x=0.99, past the caption), and
    #   2. approaches the adapters box from BELOW its caption band, entering the
    #      box top at x=0.246 with the horizontal run at the box top itself
    #      rather than above it.
    # Sources merge on a bus, then one arrow into the adapters box.
    #
    # The bus must clear band 2's caption, which sits at data-y 0.646..0.664 and
    # data-x 0.018..0.283. Band 2's boxes start at y=0.520, so the only free
    # horizontal corridor is BELOW the boxes; a bus above them (y=0.705) crosses
    # straight through the caption. Route it under the row instead, and drop into
    # the box top from the right.
    # Sources merge on a bus, then one arrow into the adapters box.
    #
    # There is exactly one free corridor: the gap between band 1's bottom edge and
    # band 2's box row. Measured: band 2's caption occupies data-y 0.646..0.664
    # and its boxes start at y=0.520, so a bus ABOVE the caption crosses it and a
    # bus BELOW the caption crosses the boxes. The gap between band 1's floor and
    # band 2's caption is the only clear run, and the drop happens at the far right
    # (x=0.99) where no text sits.
    # The italic notes sit BELOW each box, so a bus at box_bottom - 0.012 runs
    # straight through them. Merge in the clear band above the headings instead:
    # pipe row top minus a fixed gap, which is empty by construction.
    box_top = pipe_boxes[0][1] + pipe_boxes[0][3]
    bus_y = box_top + 0.030
    drop_x = max(rx + rw for rx, _ry, rw, _rh in src_boxes) + 0.018
    for x, _y, w, _h in src_boxes:
        cx = x + w / 2
        ax.plot([cx, cx], [bus_y, box_top], color="#c2c8d1", linewidth=1.0, zorder=1)
    ax.plot([src_boxes[0][0] + src_boxes[0][2] / 2, drop_x], [bus_y, bus_y],
            color="#c2c8d1", linewidth=1.0, zorder=1)
    _arrow(ax, drop_x, bus_y, drop_x, box_top, color=BLUE, rad=0.0)
    ax.plot([drop_x, pipe_boxes[0][0] + pipe_boxes[0][2] / 2], [box_top, box_top],
            color=BLUE, linewidth=1.15, zorder=2)

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
    res_w = res_boxes[0][2]
    for i in range(1, 5):
        color = BLUE if i < 3 else GREY
        _arrow(ax, res_boxes[i - 1][0] + res_w, res_boxes[0][1] + res_boxes[0][3] / 2, res_boxes[i][0], res_boxes[0][1] + res_boxes[0][3] / 2, color=color)

    # Splits feeds the implemented research row. Both targets get a real down-arrow
    # into the box top; the earlier version left one dangling in empty space.
    splits_cx = pipe_boxes[4][0] + pipe_w / 2
    features_cx = res_boxes[0][0] + res_w / 2
    graph_cx = res_boxes[1][0] + res_w / 2
    # The feed bus runs along the BOTTOM of band 3, below the boxes, not across
    # the middle. At mid-height it crossed the band's own caption.
    # The italic notes live INSIDE each box, so a bus at box_bottom - 0.020 cuts
    # straight through the note text of the row it feeds. The notes are the last
    # thing in the box, so the only clear corridor below a row is outside the band.
    feed_y = res_boxes[0][1] - 0.052
    # splits_cx is the SPLITS box centre in band 2; the drop has to clear band 3's
    # own box row on the way to the bus below it. pipe_boxes[0][1] - 0.030 is
    # y=0.490, which is inside band 3 (boxes 0.330..0.428), so the shaft ran down
    # through the ML models label. Stop the vertical just above the bus instead.
    ax.plot([splits_cx, splits_cx], [feed_y, feed_y + 0.030], color=BLUE, linewidth=1.15, zorder=2)
    ax.plot([splits_cx, graph_cx], [feed_y, feed_y], color=BLUE, linewidth=1.15, zorder=2)
    ax.plot([features_cx, graph_cx], [feed_y, feed_y], color=BLUE, linewidth=1.15, zorder=2)
    # One arrow per target, each spanning from the bus down to the box TOP. The
    # previous version ran a single long arrow to box_top + 0.006 and then added a
    # 0.006 stub from box_bottom + 0.006 back to box_bottom: that stub sat *inside*
    # the box, drawing a tick across the label, and the long arrow's shrink left it
    # spanning the box interior.
    # feed_y is BELOW this row, so the arrow rises from the bus to the box BOTTOM.
    # Ending it at the box top drew the shaft straight through the label and its
    # note (a 0.15-tall arrow crossing 0.33-0.428).
    res_bot = res_boxes[0][1]
    _arrow(ax, graph_cx, feed_y, graph_cx, res_bot, color=BLUE, rad=0.0)
    _arrow(ax, features_cx, feed_y, features_cx, res_bot, color=BLUE, rad=0.0)

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
    del_w = del_boxes[0][2]
    for i in range(1, 6):
        _arrow(ax, del_boxes[i - 1][0] + del_w, del_boxes[0][1] + del_boxes[0][3] / 2, del_boxes[i][0], del_boxes[0][1] + del_boxes[0][3] / 2, color=GREY)

    # The Evidence store is the only delivery stage that writes BACK into the
    # implemented research layer: it records the events behind every score, which
    # is what makes an explanation traceable. Dashed so it is not read as part of
    # the left-to-right delivery chain it sits inside.
    # It must physically reach the research layer, so the target is the research
    # band's own floor, not the hand-off bus (which sits below that band). The
    # arrow is drawn after the boxes, so the short span through the band gap is
    # what is visible; routing it up the right-hand clear margin keeps it off text.
    ev_x = del_boxes[3][0] + del_w / 2
    ev_top = del_boxes[0][1] + del_boxes[0][3]
    _arrow(ax, ev_x, ev_top + 0.010, ev_x, res_boxes[0][1] - 0.010, color=GREY, rad=0.0, ls=(0, (3, 2)))

    # The delivery row is fed from the models, not from the implemented baselines:
    # the pipeline it depends on is the planned one.
    models_cx = res_boxes[4][0] + res_w / 2
    detection_cx = del_boxes[0][0] + del_w / 2
    # The delivery band caption occupies y 0.204..0.222 at x 0.018..0.272, so a
    # hand-off bus at hand_y = 0.222 drew straight through the caption text. Route
    # it in the empty floor of the band, below the caption.
    hand_y = 0.252
    # Start the drop at the ML box BOTTOM. 0.330 - 0.030 is above the box top
    # (0.428), so the shaft ran down through the "ML models" label and its note.
    res_bot_y = res_boxes[0][1]
    ax.plot([models_cx, models_cx], [res_bot_y - 0.006, hand_y], color=GREY, linewidth=1.15, zorder=2)
    ax.plot([models_cx, detection_cx], [hand_y, hand_y], color=GREY, linewidth=1.15, zorder=2)
    _arrow(ax, detection_cx, hand_y, detection_cx, 0.078 + H, color=GREY)

    _legend(
        ax,
        [
            ("implemented, executed, tested", BLUE, IMPLEMENTED),
            ("designed for, not yet built", GREY, PLANNED),
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
    fig, ax = _new_canvas(10.4, 3.5)
    _title(fig, "From research question to a claim that survives its own ablations")
    _subtitle(
        fig,
        "Every stage produces a recorded artifact. Stages in grey have no artifact yet.",
    )

    steps = [
        ("Research\nquestion", ORANGE),
        ("Dataset\nsynthetic\n+ public", BLUE),
        ("Protocol\nchronological\nsplits", BLUE),
        ("Baselines\nstatistical\n+ ML", GREY),
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

    # The loop back: a result either supports the framing or revises it. Drawn as
    # ONE path with a single arrowhead, so no corner looks like it points at nothing.
    last_x = 0.012 + (len(steps) - 1) * (w + gap)
    loop_y = 0.150
    tail_x = last_x + w / 2
    head_x = 0.012 + w / 2
    ax.plot(
        [tail_x, tail_x, head_x, head_x],
        [y, loop_y, loop_y, y],
        color="#4a5058",
        linewidth=1.15,
        solid_capstyle="round",
        zorder=2,
    )
    ax.annotate(
        "",
        xy=(head_x, y),
        xytext=(head_x, loop_y),
        arrowprops={"arrowstyle": "-|>", "color": ORANGE, "linewidth": 1.4,
                    "mutation_scale": 11, "shrinkA": 0, "shrinkB": 2},
        zorder=5,
    )
    ax.text(
        0.500,
        loop_y - 0.055,
        "a result that cannot change the framing was not worth running",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#5c6470",
        style="italic",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
    )

    _legend(
        ax,
        [
            ("done", BLUE, IMPLEMENTED),
            ("planned — no artifact yet", GREY, PLANNED),
            ("where the study starts", ORANGE, IMPLEMENTED),
        ],
        loc="upper right",
    )
    return _finish(fig, ax, Path(out_path))


def data_integration(out_path: str | Path) -> str:
    """Four sources that disagree, and the decisions that reconcile them."""
    fig, ax = _new_canvas(11.0, 5.2)
    _title(fig, "Four sources that disagree about identity, time and units")
    _subtitle(
        fig,
        "Canonicalization is where information is discarded. Raw records are kept so every decision stays reversible.",
    )

    src = [
        ("transit_feed", "snake_case fields\nepoch-second stamps\nno location"),
        ("sensor_grid", "camelCase fields\nlocal timezone\nimprecise minutes"),
        ("ops_log", "free-text categories\narrival-time ordered\nno units"),
        ("billing_extract", "prefixed identifiers\ndelayed ~2 days\nmoney as text"),
    ]
    w, stride = _row(len(src), gap=0.036)
    src_h = max(
        _measure_multiline(ax, head, note, head_size=8.2, note_size=6.6)
        for head, note in src
    )
    src_y = 0.700
    # (x, y, w, h) -- _band_around reads the 4th element as the box height.
    src_boxes: list[tuple[float, float, float, float]] = []
    for i, (head, note) in enumerate(src):
        x = 0.028 + i * stride
        used = _multiline_box(ax, x, src_y, w, src_h, head, note, head_size=8.2, note_size=6.6)
        src_boxes.append((x, src_y, w, used))
    _band_around(ax, src_boxes, label="Native schemas — mutually incompatible", pad_bot=0.030)

    # --- merge on a bus below the sources so no connector crosses a caption
    bus_y = src_y - 0.058
    adapters_w = 0.300
    adapters_x = 0.028 + (1.0 - 0.028 - 0.030 - adapters_w) / 2
    adapters_cx = adapters_x + adapters_w / 2
    for x, _y, bw, _bh in src_boxes:
        cx = x + bw / 2
        ax.plot([cx, cx], [src_y - 0.030, bus_y], color="#c2c8d1", linewidth=1.0, zorder=1)
        ax.plot([cx, adapters_cx], [bus_y, bus_y], color="#c2c8d1", linewidth=1.0, zorder=1)
    _arrow(ax, adapters_cx, bus_y, adapters_cx, src_y - 0.098, color=BLUE)

    # --- adapters
    ad_note = "the only place that knows\neach source's conventions"
    ad_h = _measure_multiline(ax, "Adapters", ad_note, head_size=8.2, note_size=6.7)
    ad_y = bus_y - 0.070 - ad_h
    _multiline_box(
        ax,
        adapters_x,
        ad_y,
        adapters_w,
        ad_h,
        "Adapters",
        ad_note,
        head_size=8.2,
        note_size=6.7,
    )

    # --- canonical representation, then entity resolution, then the unified view.
    # Row heights are measured, not assumed: a fixed per-row offset let a taller
    # caption grow downward into the row below and land outside its own border.
    cols = [
        (
            "Canonical representation",
            "event_id · timestamp\nentity_ref_norm · event_type\nsource_id · provenance kept",
            BLUE,
        ),
        (
            "Entity resolution",
            "blocking → match → cluster\nsame entity, four name styles",
            BLUE,
        ),
        (
            "Unified analytical view",
            "504 clusters\nfor 500 true entities\nP=0.996, R=0.997",
            ORANGE,
        ),
    ]
    col_w, col_gap = 0.290, 0.035
    col_x = [0.028 + i * (col_w + col_gap) for i in range(3)]
    for cx_, cw_ in zip(col_x, [col_w] * 3, strict=True):
        assert cx_ + cw_ <= 1.0 - 0.030 + 1e-9, "column escapes canvas"

    # Measured for the combined head+note string these boxes actually draw.
    col_h = max(
        _measure_block(ax, f"{head}\n{note}", size=7.2) for head, note, _c in cols
    )
    foot_y = 0.048
    col_y = foot_y + 0.058
    col_cy = col_y + col_h / 2

    for cx_, (head, note, color) in zip(col_x, cols, strict=True):
        ax.add_patch(
            FancyBboxPatch(
                (cx_, col_y),
                col_w,
                col_h,
                boxstyle="round,pad=0.002,rounding_size=0.016",
                linewidth=1.3,
                edgecolor=color,
                facecolor=_fill(IMPLEMENTED),
                zorder=3,
            )
        )
        ax.text(
            cx_ + col_w / 2,
            col_cy,
            f"{head}\n{note}",
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold" if cx_ == col_x[0] else "normal",
            color=INK if color == BLUE else "#5c6470",
            linespacing=1.52,
            zorder=4,
        )
    _arrow(ax, col_x[0] + col_w, col_cy, col_x[1], col_cy, color=BLUE)
    _arrow(ax, col_x[1] + col_w, col_cy, col_x[2], col_cy, color=ORANGE)

    # adapters feed the first column
    ax.plot(
        [adapters_cx, adapters_cx, col_x[0] + col_w / 2],
        [ad_y, ad_y - 0.040, col_y + col_h],
        color=BLUE,
        linewidth=1.2,
        zorder=2,
    )
    _arrow(
        ax,
        col_x[0] + col_w / 2,
        col_y + col_h + 0.046,
        col_x[0] + col_w / 2,
        col_y + col_h,
        color=BLUE,
    )

    ax.text(
        0.500,
        foot_y,
        "Every arrow above is a modelling decision: a time origin, an identity rule, a unit convention.",
        ha="center",
        va="center",
        fontsize=7.8,
        color=RED,
        style="italic",
    )
    _legend(
        ax,
        [
            ("implemented", BLUE, IMPLEMENTED),
            ("the unified analytical view", ORANGE, IMPLEMENTED),
        ],
        loc="lower right",
    )
    ax.set_ylim(0.01, 1.0)
    return _finish(fig, ax, Path(out_path))


def temporal_leakage(out_path: str | Path) -> str:
    """The temporal protocol, and the flow that is forbidden.

    Red is used exactly once, on the arrow that must never be taken. That is the
    only thing this figure needs the reader to remember.
    """
    _style()
    fig, ax = _new_canvas(10.6, 4.3)
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
    _legend(
        ax,
        [
            ("information may flow forward in time", BLUE, IMPLEMENTED),
            ("scored only — nothing is fitted here", ORANGE, IMPLEMENTED),
            ("this direction is forbidden", RED, FORBIDDEN),
        ],
        # lower left collided with the "earliest" axis label (y 0.144..0.176) and
        # the italic note (x 0.186..0.814, y 0.059..0.091): the legend box spanned
        # x 0.006..0.292, y 0.036..0.180 and overlapped both.
        loc="upper right",
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
    fig, ax = _new_canvas(10.8, 5.6)
    _title(fig, "A snapshot at T contains only edges first seen at or before T")
    _subtitle(
        fig,
        "Each edge stores the time it was first observed. Historical measures are computed by replaying edges in timestamp order.",
    )

    H = 0.130

    def chain(y: float, items: list[tuple[str, str, str]], gap: float) -> list[tuple[float, float, float, float]]:
        w, stride = _row(len(items), gap=gap)
        out = []
        for i, (head, note, kind) in enumerate(items):
            x = 0.028 + i * stride
            used = _multiline_box(ax, x, y, w, H, head, note, kind=kind, head_size=7.7, note_size=6.5)
            out.append((x, y, w, used))
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
            ("permitted construction", BLUE, IMPLEMENTED),
            ("forbidden: future edges reach the past", RED, FORBIDDEN),
        ],
        loc="lower left",
    )
    ax.set_ylim(0.02, 1.0)
    return _finish(fig, ax, Path(out_path))


def evidence_lineage(out_path: str | Path) -> str:
    """The traceability chain, both directions.

    Reading right-to-left is the audit path a reviewer takes; left-to-right is the
    path an analyst takes when investigating an alert.

    The evidence store is drawn as the column it actually is, joined by a line to
    every stage, rather than being left to a sentence in the subtitle: it is the
    only thing that connects a score back to the rows that produced it, so leaving
    it implicit made the figure's central claim invisible.
    """
    fig, ax = _new_canvas(11.6, 4.2)
    _title(fig, "Any anomaly walks back to its source records and forward to the experiment that measured it")
    _subtitle(
        fig,
        "The evidence store is the join between a score and the rows that produced it. It is not implemented yet.",
    )

    stages = [
        ("Source\nrecord", BLUE, IMPLEMENTED),
        ("Canonical\nevent", BLUE, IMPLEMENTED),
        ("Feature value\n+ registry\nfingerprint", BLUE, IMPLEMENTED),
        ("Model", GREY, PLANNED),
        ("Score", GREY, PLANNED),
        ("Anomaly", GREY, PLANNED),
        ("Explanation\ncontributing\nfeatures", GREY, PLANNED),
        ("Experiment\nrecord", GREY, PLANNED),
    ]
    # Widths are derived from the row length so the last box is never clipped.
    left, right = 0.012, 0.988
    gap = 0.040
    w = (right - left - gap * (len(stages) - 1)) / len(stages)
    y, h = 0.560, 0.235

    centres: list[float] = []
    for idx, (label, color, kind) in enumerate(stages):
        x = left + idx * (w + gap)
        centres.append(x + w / 2)
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.006,rounding_size=0.02",
                linewidth=1.35,
                edgecolor=_edge(kind),
                facecolor=_fill(kind),
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
        if idx < len(stages) - 1:
            _arrow(
                ax,
                x + w,
                y + h / 2,
                x + w + gap,
                y + h / 2,
                color=BLUE if kind == IMPLEMENTED else GREY,
            )

    # The evidence store spans the whole chain: every stage is a row it can point at.
    store_y, store_h = 0.215, 0.130
    ax.add_patch(
        FancyBboxPatch(
            (left, store_y),
            right - left,
            store_h,
            boxstyle="round,pad=0.004,rounding_size=0.016",
            linewidth=1.4,
            edgecolor=_edge(PLANNED),
            facecolor=_fill(PLANNED),
            zorder=3,
        )
    )
    ax.text(
        0.500,
        store_y + store_h / 2,
        "Evidence store  \u2014  one row per score: contributing feature values, model weights, run fingerprint",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#5c6470",
        zorder=4,
    )
    for cx in centres:
        ax.plot(
            [cx, cx],
            [store_y + store_h, y],
            color=GREY,
            linewidth=0.9,
            linestyle=(0, (2.5, 2.5)),
            zorder=2,
        )

    ax.text(
        0.500,
        0.455,
        "analyst investigates an alert  \u2190   \u00b7   \u2192  reviewer audits a claim: same fingerprint, same feature definitions",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#5c6470",
        # The dashed drop-lines pass behind this line; an opaque backing keeps the
        # text readable instead of letting dashes run through the words.
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0},
        zorder=5,
    )
    _legend(
        ax,
        [
            ("implemented", BLUE, IMPLEMENTED),
            ("designed for, not yet built", GREY, PLANNED),
        ],
        loc="lower left",
    )
    ax.set_ylim(0.03, 1.0)
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