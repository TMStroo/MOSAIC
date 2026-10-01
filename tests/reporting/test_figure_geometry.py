"""Geometry tests for the README figures.

These are decidable claims, so they are asserted from matplotlib's own rendered
extents rather than by looking at the PNG. Every bug this file guards was found
by measuring: a caption overlapping its heading, a caption escaping the bottom
border, a box overlapping its neighbour, and an arrow terminating in empty space.

Looking at images finds these. It does not prove they are gone.
"""

from __future__ import annotations

from pathlib import Path

import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mosaic.reporting.figures import (
    ALL_FIGURES,
    _box,
    _multiline_box,
    _style,
    _text_height,
    evidence_lineage,
)

GOLD = "#c8973f"
MIN_SEPARATION = 0.004


def _axes() -> tuple[plt.Figure, plt.Figure, plt.Axes]:
    _style()
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    return fig, fig, ax


def _extent_y(ax: plt.Axes, artist) -> tuple[float, float]:
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    bb = artist.get_window_extent()
    (_, y0) = inv.transform((bb.x0, bb.y0))
    (_, y1) = inv.transform((bb.x1, bb.y1))
    return y0, y1


def _rects(ax: plt.Axes, *, content_only: bool = True) -> list[tuple[float, float, float, float]]:
    """Rounded rectangles on the axes.

    ``content_only`` drops the shaded layer bands, which are also FancyBboxPatch
    and are meant to sit behind the boxes rather than collide with them. A band
    spans nearly the full width and contains its boxes, so including it would
    report every row as an overlap.
    """
    out = []
    for patch in ax.patches:
        if not isinstance(patch, FancyBboxPatch):
            continue
        if content_only and patch.get_width() > 0.5:
            continue
        out.append((patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()))
    return out


@pytest.mark.parametrize(
    "note",
    [
        "one line",
        "line one\nline two",
        "line one\nline two\nline three",
        "line one\nline two\nline three\nline four",
    ],
)
def test_caption_never_overlaps_heading(note: str) -> None:
    """The measured bug: heading and caption drawn on top of each other."""
    _fig, _f, ax = _axes()
    _multiline_box(ax, 0.10, 0.30, 0.40, 0.10, "Heading", note)
    fig = ax.figure
    fig.canvas.draw()

    heading, caption = ax.texts[0], ax.texts[1]
    head_bottom, _ = _extent_y(ax, heading)
    _, note_top = _extent_y(ax, caption)

    assert head_bottom - note_top > MIN_SEPARATION, (
        f"caption overlaps heading for note={note!r}: "
        f"gap={head_bottom - note_top:.4f}"
    )
    plt.close(fig)


@pytest.mark.parametrize(
    "note",
    [
        "one line",
        "line one\nline two",
        "line one\nline two\nline three",
        "line one\nline two\nline three\nline four",
    ],
)
def test_caption_stays_inside_box(note: str) -> None:
    """The measured bug: the last caption line rendered below the border."""
    _fig, _f, ax = _axes()
    height = _multiline_box(ax, 0.10, 0.30, 0.40, 0.08, "Heading", note)
    fig = ax.figure
    fig.canvas.draw()

    (x, y, w, h) = _rects(ax)[0]
    assert h == pytest.approx(height)

    for artist in ax.texts:
        y0, y1 = _extent_y(ax, artist)
        assert y0 > y, f"{artist.get_text()[:16]!r} fell below the box: {y0:.4f} < {y:.4f}"
        assert y1 < y + h, f"{artist.get_text()[:16]!r} rose above the box top"

        x0, x1 = _extent_x(ax, artist)
        assert x0 > x and x1 < x + w, f"{artist.get_text()[:16]!r} escaped horizontally"


def _extent_x(ax: plt.Axes, artist) -> tuple[float, float]:
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    bb = artist.get_window_extent()
    (x0, _) = inv.transform((bb.x0, bb.y0))
    (x1, _) = inv.transform((bb.x1, bb.y1))
    return x0, x1


def test_taller_than_needed_box_still_centres_text() -> None:
    """A box given far more height than its text needs must not push text out.

    This was the subtler half of the bug: anchoring the caption to the box bottom
    worked for boxes sized exactly to their content and broke for every box that
    was given extra height.
    """
    _fig, _f, ax = _axes()
    _multiline_box(ax, 0.10, 0.20, 0.40, 0.40, "Heading", "a\nb\nc")
    fig = ax.figure
    fig.canvas.draw()

    (x, y, _w, h) = _rects(ax)[0]
    for artist in ax.texts:
        y0, y1 = _extent_y(ax, artist)
        assert y0 > y and y1 < y + h


def test_text_height_is_measured_not_estimated() -> None:
    """_text_height must report the renderer's extent, not size/72.

    An earlier model derived height from the em box and was ~1.8x too small,
    which is precisely why captions overlapped headings.
    """
    _fig, _f, ax = _axes()
    fig = ax.figure
    for size in (6.0, 6.6, 8.0, 9.0):
        measured = _text_height(ax, size)
        naive = _pt_naive(ax, size)
        assert measured > naive, (
            f"fontsize {size}: measured {measured:.5f} should exceed the naive "
            f"em-box estimate {naive:.5f}"
        )
    plt.close(fig)


def _pt_naive(ax: plt.Axes, size: float) -> float:
    inv = ax.transData.inverted()
    (_, y0) = inv.transform((0.0, 0.0))
    (_, y1) = inv.transform((0.0, 72.0))
    return abs(y1 - y0) * size / 72.0


def test_boxes_never_overlap_in_any_figure() -> None:
    """Every rendered figure: no two boxes overlap.

    Overlap shipped three times while these figures were being built, in three
    different figures, and each one looked fine at a glance.
    """
    violations: list[str] = []
    for name, fn in ALL_FIGURES.items():
        fig, captured = _capture(fn, name)
        rects = _rects(captured)
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if _overlaps(rects[i], rects[j]):
                    violations.append(f"{name}: boxes {i} and {j} overlap")
        plt.close(fig)
    assert not violations, "\n".join(violations)


def _neutral(col: tuple[float, float, float]) -> bool:
    r, g, b = col
    return abs(r - g) < 0.03 and abs(g - b) < 0.03


def _rgb(value) -> tuple[float, float, float] | None:
    """Normalise a matplotlib colour (hex str or RGBA tuple) to RGB, or None."""
    from matplotlib.colors import to_rgb

    if value is None:
        return None
    try:
        r, g, b = to_rgb(value)
    except (ValueError, TypeError):
        return None
    return (round(r, 4), round(g, 4), round(b, 4))


def _capture(fn, name: str):
    """Run a figure function and return the (figure, axes) it drew into."""
    import mosaic.reporting.figures as mod

    original = mod._finish
    holder: dict = {}

    def spy(fig, ax, path):
        holder["fig"], holder["ax"] = fig, ax
        return path

    mod._finish = spy
    try:
        fn(f"docs/figures/{name}.png")
    finally:
        mod._finish = original
    return holder["fig"], holder["ax"]


def _overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    x_overlap = min(ax0 + aw, bx0 + bw) - max(ax0, bx0)
    y_overlap = min(ay0 + ah, by0 + bh) - max(ay0, by0)
    return x_overlap > 0.002 and y_overlap > 0.002


def test_every_figure_is_registered_and_named() -> None:
    """Six figures, descriptive filenames, matching the README."""
    assert set(ALL_FIGURES) == {
        "architecture",
        "research-workflow",
        "data-integration",
        "temporal-leakage",
        "graph-methodology",
        "evidence-lineage",
    }


def test_figure_files_exist_and_are_not_tiny(tmp_path) -> None:
    written = []
    for name, fn in ALL_FIGURES.items():
        out = fn(tmp_path / f"{name}.png")
        assert isinstance(out, str), f"{name} returned {type(out).__name__}, expected str"
        size = Path(out).stat().st_size
        assert size > 10_000, f"{name}.png is only {size} bytes"
        written.append(Path(out).name)
    assert len(written) == 6


def test_legend_marker_is_visible_colour() -> None:
    """A legend whose first swatch lost its colour reads as two grey squares."""
    _fig, _f, ax = _axes()
    from mosaic.reporting.figures import (
        BLUE,
        FORBIDDEN,
        GREY,
        IMPLEMENTED,
        PLANNED,
        RED,
        _legend,
    )

    _legend(
        ax,
        [
            ("implemented", BLUE, IMPLEMENTED),
            ("planned", GREY, PLANNED),
            ("forbidden", RED, FORBIDDEN),
        ],
        loc="lower left",
    )
    leg = ax.get_legend()
    assert leg is not None, "no legend was added"
    handles = leg.legend_handles
    assert len(handles) == 3, f"expected 3 legend markers, found {len(handles)}"
    edges = {tuple(h.get_edgecolor()) for h in handles}
    assert len(edges) == 3, f"legend markers share an edge colour: {edges}"
    fills = {tuple(h.get_facecolor()) for h in handles}
    assert len(fills) == 3, f"legend swatches share one fill: {fills}"
    plt.close(ax.figure)


def test_box_helper_still_draws_a_single_centred_line() -> None:
    _fig, _f, ax = _axes()
    _box(ax, 0.1, 0.1, 0.2, 0.1, "hello")
    fig = ax.figure
    fig.canvas.draw()
    y0, y1 = _extent_y(ax, ax.texts[0])
    assert 0.1 < y0 < y1 < 0.2, f"text not vertically centred: {y0:.4f}..{y1:.4f}"
    plt.close(fig)


def test_palette_names_are_importable() -> None:
    """The figures' palette constants are part of the documented visual contract."""
    from mosaic.reporting import figures as mod

    for name in ("BLUE", "ORANGE", "GREY", "RED", "INK", "BAND"):
        assert re.fullmatch(r"#[0-9a-f]{6}", getattr(mod, name)), f"{name} is not a hex colour"
    assert evidence_lineage is not None


def test_every_edge_colour_in_a_figure_appears_in_its_legend() -> None:
    """A colour used on a box but absent from the legend is unexplained.

    Found in research-workflow: the research question and the feedback arrow were
    orange, and the legend offered only "done" (blue) and "planned" (grey), so the
    one stage that is both done and highlighted had no key.
    """
    unexplained: list[str] = []
    for name, fn in ALL_FIGURES.items():
        fig, captured = _capture(fn, name)
        leg = captured.get_legend()
        if leg is None:
            rects = _rects(captured)
            coded = {
                _rgb(p.get_edgecolor())
                for p in captured.patches
                if isinstance(p, FancyBboxPatch) and p.get_width() <= 0.5
            }
            coded = {c for c in coded if c and not _neutral(c)}
            if len(coded) > 1:
                unexplained.append(f"{name}: uses {len(coded)} coded box colours but has no legend")
        else:
            in_legend = {_rgb(h.get_edgecolor()) for h in leg.legend_handles}
            used: set = set()
            for p in captured.patches:
                if isinstance(p, FancyBboxPatch) and p.get_width() <= 0.5:
                    used.add(_rgb(p.get_edgecolor()))
            for col in used - in_legend:
                if col is None:
                    continue
                r, g, b = col
                if (r, g, b) in {(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)}:
                    continue
                if abs(r - g) < 0.03 and abs(g - b) < 0.03 and r > 0.85:
                    continue  # pale greys: separators and fills
                if abs(r - g) < 0.03 and abs(g - b) < 0.03 and r < 0.45:
                    continue  # dark neutral grey: structural text and arrows
                unexplained.append(f"{name}: colour {col} used but absent from the legend")
        plt.close(fig)
    assert not unexplained, "\n".join(unexplained)


def test_no_arrowhead_points_into_empty_space() -> None:
    """Every arrowhead must land on a box, not mid-path or at a corner.

    Chained _arrow calls each drew a head, which put arrowheads at the elbows of
    the research-workflow loop where there was nothing to point at.
    """
    offenders: list[str] = []
    for name, fn in ALL_FIGURES.items():
        fig, captured = _capture(fn, name)
        rects = _rects(captured)
        for patch in captured.patches:
            if not isinstance(patch, FancyArrowPatch):
                continue
            tip = patch._posA_posB[1]  # arrow tip in data coords
            nearest = min(
                (
                    min(abs(tip[0] - (rx + rw / 2)), abs(tip[1] - (ry + rh / 2)))
                    for rx, ry, rw, rh in rects
                ),
                default=9.9,
            )
            if nearest > 0.30:
                offenders.append(f"{name}: arrow tip {tuple(round(v, 3) for v in tip)} is {nearest:.2f} from every box")
        plt.close(fig)
    assert not offenders, "\n".join(offenders)
