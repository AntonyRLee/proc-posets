"""String-diagram visualiser for cospan signatures and their compositions.

A quick visual checker for ``Sigma`` and ``F(Sigma)``: each generator
(``def:generator-cospan-general``) renders as a labelled box whose left/right
typed :class:`~procposets.cospan.signature.Port` triples are coloured wires; ``;``
(sequential) chains boxes left-to-right joining matched ports; ``⊗`` (parallel)
stacks them.  A legend maps object type -> colour.

Two entry points, both feeding one ``Layout`` so a TikZ backend can be added
later against the same geometry:

  * **term DSL** -- wrap generators and compose with Python operators
    ``>>`` (``;``) and ``@`` (``⊗``)::

        d = D(sig, "register") >> D(sig, "triage", 0) \\
            >> (D(sig, "order_labs") @ D(sig, "order_imaging"))
        render(d, sig).savefig("d.png")

  * **a composite run** straight from ``compose_signature``::

        render(comp, sig)           # comp : CompositeDiagram

Port identity is the ``(src, typ, tgt)`` triple -- exactly the matching
``compose.py`` uses (``g.left <= available``) -- so a wire is drawn between a
right port of one box and a left port of another iff the triples are equal.

The backend-independent geometry (the term DSL, the ``Layout``/``PlacedBox``/
``Wire`` datatypes, :class:`LayoutStyle`, and the two lowering paths ``_finish``
/ ``_layout_composite``) lives in the matplotlib-free :mod:`._layout`; this
module is the matplotlib drawing backend and re-exports those names, so every
``string_diagram.NAME`` keeps resolving unchanged.
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass, field
from math import comb

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from ..cospan.compose import CompositeDiagram
from ..cospan.signature import Port, Signature
from ..cospan.signature_compare import binding_profile
from ..cospan.typebalance import Kappa

# Re-export the backend-independent layout half so `string_diagram.NAME` keeps
# resolving for every name that used to live here (the drawing code below also
# imports these for its own use). LayoutStyle is defined there (matplotlib-free);
# DrawStyle/StringDiagramStyle stay here beside the matplotlib backend.
from ._layout import (  # noqa: F401  (re-exported: public surface + used below)
    _BH,
    _BOX_PAD,
    _BW,
    _HGAP,
    _LANE_BASE,
    _LANE_STEP,
    _MARGIN,
    _PS,
    _RISER,
    _VGAP,
    D,
    Diagram,
    Layout,
    LayoutStyle,
    PlacedBox,
    Wire,
    _box_sub,
    _consumed_later,
    _finish,
    _gen_delta,
    _layout_composite,
    _optimize_ports,
    _route_long_edges,
    _straighten_boxes,
    _count_crossings,
    lower_term,
    _par,
    _ports,
    _seq,
    _Sub,
    gens,
    pick,
)


def _darken(colour, factor: float = 0.7):
    """Darker shade of ``colour`` for high-contrast (non-faded) coloured text."""
    r, g, b = to_rgb(colour)
    return (r * factor, g * factor, b * factor)


# --- style config dataclasses ----------------------------------------------
# The style knobs, grouped into two frozen dataclasses split along the exact
# layout/draw fault line: the LAYOUT knobs (:class:`LayoutStyle`, in the
# matplotlib-free ._layout) are read only by the geometry functions, the DRAW
# knobs below only by this matplotlib backend. Pass style=StringDiagramStyle(...)
# to override the defaults; render(style=None) uses DEFAULT_STYLE.
@dataclass(frozen=True)
class DrawStyle:
    boundary_linestyle: str = "--"
    internal_wire_linestyle: str = "-"
    # How wire crossings are rendered (readability problem (c)):
    #   "plain"  -- draw every wire whole, crossings are bare X's. Correct for a
    #               SYMMETRIC monoidal category (c_{A,B}=c_{B,A}^-1, no over/under
    #               data), maximises continuity, and lets type-colour carry
    #               identity through the crossing. The default.
    #   "casing" -- redraw the over-wire with a background-colour halo behind the
    #               thin coloured wire, one global z-order so a wire is
    #               consistently over/under (no flicker); a depth cue for dense
    #               same-hue crossings, at the cost of breaking the under-wire.
    #   "gap"    -- the legacy braid: erase a constant disk from the under-wire.
    crossing_style: str = "plain"
    crossing_halo_bg: str = "white"  # halo/background colour for "casing"
    label_casing: bool = True  # draw each box label on a patch in the box's own
    # facecolour, so an activity name wider than its box masks (rather than
    # collides with) any wire passing near it (readability problem: label-wire
    # overlap on narrow boxes).
    open_end_markers: bool = True
    box_facecolor: str = "white"
    box_face_overrides: dict[str, str] = field(default_factory=dict)
    box_label_map: dict[str, str] = field(default_factory=dict)
    box_label_fontsize: float = 9
    svg_gids: bool = False
    gid_roles: dict[str, str] = field(default_factory=dict)
    gid_round_offset: int = 0


@dataclass(frozen=True)
class StringDiagramStyle:
    layout: LayoutStyle = LayoutStyle()
    draw: DrawStyle = DrawStyle()


DEFAULT_STYLE = StringDiagramStyle()


# --- bezier wire rendering + braid-style crossing breaks --------------------
_CURVE_N = 120  # samples per rendered curve (dense, so the constant-radius
# break below lands on real geometry, not a coarse polyline)
_GAP = 0.11  # Euclidean radius (data units) of the disk erased from the
# "under" wire at a crossing -- constant, so every break looks
# the same size whatever the crossing angle
_TRIM = 0.06  # ignore crossings within this fraction of either curve's ends

# --- type -> colour ---------------------------------------------------------
# Palette centralized in viz/palette.py (verbatim hexes; the loop-family ports use
# the "Red, Black & Blue" variants). ``_colour_map`` copies ``_PALETTE`` before adding
# fallbacks, so the shared dict is never mutated.
from .palette import STRING_DIAGRAM_TYPE_COLOURS as _PALETTE  # noqa: E402
from .palette import STRING_DIAGRAM_FALLBACK as _FALLBACK  # noqa: E402


def _colour_map(types: set[str | None]) -> dict[str | None, str]:
    out = dict(_PALETTE)
    i = 0
    for t in sorted((x for x in types if x not in out), key=str):
        out[t] = _FALLBACK[i % len(_FALLBACK)]
        i += 1
    return out


def _gid_sanitize(label: str) -> str:
    """Stable id token: non-id characters collapsed to '_' (f2^(m-1) -> f2_m-1)."""
    return _re.sub(r"[^A-Za-z0-9_.:-]+", "_", label).strip("_")


def _semantic_gid(b: PlacedBox, layout: Layout, style: DrawStyle) -> tuple[str, str]:
    """(element id, label id) for one placed box, per the animation SVG spec."""
    role = style.gid_roles.get(b.label, "activity")
    base = _gid_sanitize(b.label)
    if role == "result":
        cols = sorted({round(x.x, 6) for x in layout.boxes
                       if style.gid_roles.get(x.label) == "result"})
        r = cols.index(round(b.x, 6)) + 1 + style.gid_round_offset
        return f"node-{base}-r{r}", f"nodelabel-{base}-r{r}"
    if role == "block":
        return f"block-{base}", f"blocklabel-{base}"
    return f"box-{base}", f"boxlabel-{base}"  # activity and boundary alike


# --- matplotlib backend -----------------------------------------------------
def _violates(box: PlacedBox, kappa: Kappa) -> bool:
    prof = kappa.get(box.label)
    if prof is None:
        return False
    return bool((box.created - prof.creates) or (box.consumed - prof.consumes))


def _bezier_path(points, n: int = _CURVE_N) -> np.ndarray:
    """One Bezier curve of degree ``len(points)-1`` through ``points`` (Bernstein
    form): a single polynomial, so however many through-points shape it there
    are no internal joints/curvature breaks -- the whole wire bends smoothly
    in one piece."""
    pts = np.asarray(points, dtype=float)
    deg = len(pts) - 1
    t = np.linspace(0.0, 1.0, n)
    basis = np.stack(
        [comb(deg, k) * t**k * (1 - t) ** (deg - k) for k in range(deg + 1)], axis=1
    )
    return basis @ pts


def _bezier(
    x1: float, y1: float, x2: float, y2: float, n: int = _CURVE_N
) -> np.ndarray:
    """Cubic bezier from ``(x1,y1)`` to ``(x2,y2)`` with a horizontal tangent at
    *both* ends (control points offset purely along x) -- a wire always leaves
    and enters a port/lane at 0/180 degrees, however much it bends in between.
    Degenerates to a straight horizontal line when ``y1 == y2``."""
    dx = x2 - x1
    return _bezier_path(
        [(x1, y1), (x1 + 0.5 * dx, y1), (x2 - 0.5 * dx, y2), (x2, y2)], n
    )


def _curve_for(w: "Wire") -> np.ndarray:
    if w.waypoints is not None:
        return _bezier_path(list(w.waypoints))
    return _bezier(w.x1, w.y1, w.x2, w.y2)


def _arclen(pts: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _seg_intersect(a1, a2, b1, b2) -> tuple[float, float] | None:
    """``(s_a, s_b)`` arc-length offsets along ``a1-a2`` / ``b1-b2`` at their
    crossing point (``t`` and ``u`` scaled by each segment's length so they add
    up to a position on the whole curve), or ``None`` if they do not cross."""
    d1, d2 = a2 - a1, b2 - b1
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return None
    t = ((b1[0] - a1[0]) * d2[1] - (b1[1] - a1[1]) * d2[0]) / denom
    u = ((b1[0] - a1[0]) * d1[1] - (b1[1] - a1[1]) * d1[0]) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return float(t), float(u)
    return None


def _curve_crossings(pts_a: np.ndarray, pts_b: np.ndarray) -> list[tuple[float, float]]:
    """``(s_a, s_b)`` arc-length position on each curve for every place the two
    sampled polylines cross.  The crossing point in the plane is recovered from
    either curve at that arc-length; here we only need where to break."""
    s_a, s_b = _arclen(pts_a), _arclen(pts_b)
    out = []
    for i in range(len(pts_a) - 1):
        for j in range(len(pts_b) - 1):
            hit = _seg_intersect(pts_a[i], pts_a[i + 1], pts_b[j], pts_b[j + 1])
            if hit is None:
                continue
            t, u = hit
            out.append(
                (s_a[i] + t * (s_a[i + 1] - s_a[i]), s_b[j] + u * (s_b[j + 1] - s_b[j]))
            )
    return out


def _split_at_disks(pts: np.ndarray, centres: list[np.ndarray]) -> list[np.ndarray]:
    """``pts`` with every sample inside a radius-``_GAP`` disk around any
    crossing ``centre`` dropped -- the "braid" break that reads as the wire
    passing under another.  Erasing a *constant Euclidean disk* (rather than a
    fixed arc-length window) keeps the visible gap the same size at every
    crossing angle and never blows up where two wires run nearly parallel."""
    if not centres:
        return [pts]
    keep = np.ones(len(pts), dtype=bool)
    for c in centres:
        keep &= np.linalg.norm(pts - c, axis=1) > _GAP
    runs: list[np.ndarray] = []
    start = None
    for k, alive in enumerate(keep):
        if alive and start is None:
            start = k
        elif not alive and start is not None:
            if k - start >= 2:
                runs.append(pts[start:k])
            start = None
    if start is not None and len(pts) - start >= 2:
        runs.append(pts[start:])
    return runs


def _point_at(pts: np.ndarray, s: float) -> np.ndarray:
    """The plane point at arc-length ``s`` along sampled polyline ``pts``."""
    arc = _arclen(pts)
    k = int(np.searchsorted(arc, s))
    k = max(1, min(k, len(pts) - 1))
    seg = arc[k] - arc[k - 1]
    frac = 0.0 if seg < 1e-12 else (s - arc[k - 1]) / seg
    return pts[k - 1] + frac * (pts[k] - pts[k - 1])


def _draw_wires(ax, wires: list["Wire"], cmap: dict, style: DrawStyle) -> None:
    internal = [w for w in wires if not w.boundary]
    curves = [_curve_for(w) for w in internal]

    def _stamp(line, typ):
        if not style.svg_gids:
            return
        cnt = ax.figure.__dict__.setdefault("_wire_gid_counts", {})
        cnt[typ] = cnt.get(typ, 0) + 1
        line.set_gid(f"wire-{typ}-s{cnt[typ]}")

    mode = getattr(style, "crossing_style", "plain")
    if mode == "gap":
        # legacy braid: erase a constant Euclidean disk from the longer ("under")
        # wire at each genuine (not near-port) crossing.
        lengths = [_arclen(c)[-1] for c in curves]
        breaks: list[list[np.ndarray]] = [[] for _ in internal]
        for i in range(len(internal)):
            for j in range(i + 1, len(internal)):
                for s_i, s_j in _curve_crossings(curves[i], curves[j]):
                    lo_i, hi_i = _TRIM * lengths[i], (1 - _TRIM) * lengths[i]
                    lo_j, hi_j = _TRIM * lengths[j], (1 - _TRIM) * lengths[j]
                    if not (lo_i <= s_i <= hi_i and lo_j <= s_j <= hi_j):
                        continue  # too close to a port -- not a real crossing
                    under = i if lengths[i] >= lengths[j] else j
                    breaks[under].append(
                        _point_at(curves[under], s_i if under == i else s_j)
                    )
        for w, pts, centres in zip(internal, curves, breaks):
            for run in _split_at_disks(pts, centres):
                (ln,) = ax.plot(
                    run[:, 0], run[:, 1], color=cmap[w.typ], lw=2.2,
                    solid_capstyle="butt", zorder=1, ls=style.internal_wire_linestyle,
                )
                _stamp(ln, w.typ)
    elif mode == "casing":
        # background-colour halo behind each coloured wire, one global z-order so
        # the longer wire is consistently UNDER (its halo can't mask a shorter
        # wire, and shorter wires' halos mask it) -- no over/under flicker.
        lengths = [_arclen(c)[-1] for c in curves]
        n = len(internal)
        order = sorted(range(n), key=lambda k: -lengths[k])  # longest first == lowest
        rank = {k: r for r, k in enumerate(order)}
        for k, (w, pts) in enumerate(zip(internal, curves)):
            z = 1.0 + (1.8 * rank[k] / max(1, n - 1))  # in [1.0, 2.8], stays < boxes (3)
            ax.plot(pts[:, 0], pts[:, 1], color=style.crossing_halo_bg, lw=5.0,
                    solid_capstyle="round", zorder=z)
            (ln,) = ax.plot(pts[:, 0], pts[:, 1], color=cmap[w.typ], lw=2.2,
                            solid_capstyle="butt", zorder=z + 0.01,
                            ls=style.internal_wire_linestyle)
            _stamp(ln, w.typ)
    else:  # "plain" -- bare X crossings, every wire continuous (symmetric monoidal)
        for w, pts in zip(internal, curves):
            (ln,) = ax.plot(pts[:, 0], pts[:, 1], color=cmap[w.typ], lw=2.2,
                            solid_capstyle="butt", zorder=1,
                            ls=style.internal_wire_linestyle)
            _stamp(ln, w.typ)

    for w in wires:
        if not w.boundary:
            continue
        (ln,) = ax.plot(
            [w.x1, w.x2],
            [w.y1, w.y2],
            color=cmap[w.typ],
            lw=2.2,
            solid_capstyle="butt",
            zorder=1,
            ls=style.boundary_linestyle,
        )
        _stamp(ln, w.typ)


def render(
    obj,
    signature: Signature | None = None,
    *,
    title: str | None = None,
    labels: bool = False,
    kappa: Kappa | None = None,
    ax=None,
    wire_labels: dict[Port, str] | None = None,
    couplings: list | None = None,
    keys: list | None = None,
    offset: tuple[float, float] | None = None,
    style: StringDiagramStyle | None = None,
):
    """Render a :class:`Diagram` or :class:`CompositeDiagram` to a matplotlib
    figure.  ``signature`` (optional) seeds the legend with all its object types
    so colours stay stable across diagrams.  ``labels`` annotates boundary wires
    with their port.  ``kappa`` (optional) red-outlines any box that creates or
    consumes an object type outside its type-balance licence ``⋈`` -- a visual
    type-balance check.  ``wire_labels`` annotates a boundary stub's port with
    custom text instead of its triple -- for a symbolic multiplicity (e.g.
    ``"2u"``) a single representative port stands in for an unbounded number
    of same-typed wires that can't literally be drawn.  Returns the
    ``style`` overrides the layout/draw knobs; ``None`` (default) uses
    :data:`DEFAULT_STYLE`.  Returns the :class:`~matplotlib.figure.Figure`."""
    if style is None:
        style = DEFAULT_STYLE
    lay, ds = style.layout, style.draw
    if isinstance(obj, Diagram):
        sub = obj._sub(lay)
        if offset is not None:
            sub = sub.shift(*offset)
        layout = lower_term(sub, lay)
    elif isinstance(obj, CompositeDiagram):
        layout = _layout_composite(obj)
    else:
        raise TypeError(f"cannot render {type(obj).__name__}")

    legend_types = set(layout.types)
    if signature is not None:
        legend_types |= {p.typ for p in signature.ports()}
    cmap = _colour_map(legend_types)

    # default labelled view: number every boundary port once and list the
    # number -> (src, type, tgt) lookup in a legend under the diagram -- full
    # triples on the wires collide with each other and the type legend.  An
    # explicit ``wire_labels`` (e.g. from catalogue()) takes precedence.
    numbered_ports: list[Port] | None = None
    if labels and wire_labels is None:
        numbered_ports = sorted(
            {w.port for w in layout.wires if w.boundary and w.port is not None},
            key=lambda p: (str(p.typ), p.src, p.tgt),
        )
        wire_labels = {p: str(k) for k, p in enumerate(numbered_ports)}

    if ax is None:
        # size from the layout's extents: a ⊗-stack of many generators grows
        # tall, a ;-chain grows wide -- the old fixed 4-inch height squashed
        # the former into unreadability.
        bxs = [b.x for b in layout.boxes] or [0.0]
        bys = [y for b in layout.boxes for y in (b.y - b.half_h, b.y + b.half_h)] or [0.0]
        w_in = max(4.0, 1.6 * (max(bxs) - min(bxs)) + 4.0)
        h_in = max(4.0, 0.85 * (max(bys) - min(bys)) + 2.0)
        fig, ax = plt.subplots(figsize=(w_in, h_in))
        owns_fig = True
    else:
        fig = ax.figure
        owns_fig = False

    _draw_wires(ax, layout.wires, cmap, ds)
    for w in layout.wires:
        if not w.boundary:
            continue
        ex, ey = w.open_end if w.open_end is not None else (w.x1, w.y1)
        if ds.open_end_markers:
            ax.plot([ex], [ey], marker="o", ms=4, color=cmap[w.typ], zorder=5)
        # open end at x1 -> box sits to the right -> label extends further
        # left (away from the box); open end at x2 -> the mirror case.
        ha = "right" if (ex, ey) == (w.x1, w.y1) else "left"
        if wire_labels is not None and w.port in wire_labels:
            ax.annotate(
                wire_labels[w.port],
                (ex, ey),
                fontsize=9,
                fontweight="bold",
                color=_darken(cmap[w.typ]),
                ha=ha,
                va="center",
                zorder=5,
                xytext=(-6 if ha == "right" else 6, 0),
                textcoords="offset points",
            )
        elif labels and w.port is not None:
            ax.annotate(
                str(w.port), (ex, ey), fontsize=6, color=_darken(cmap[w.typ]), ha=ha, va="center"
            )

    bad = False
    for b in layout.boxes:
        offending = kappa is not None and _violates(b, kappa)
        bad = bad or offending
        rect = mpatches.FancyBboxPatch(
            (b.x - _BW / 2, b.y - b.half_h),
            _BW,
            2 * b.half_h,
            boxstyle=f"round,pad={_BOX_PAD},rounding_size=0.08",
            linewidth=2.6 if offending else 1.4,
            edgecolor="#d62728" if offending else "black",
            facecolor="#ffecec"
            if offending
            else ds.box_face_overrides.get(b.label, ds.box_facecolor),
            zorder=3,
        )
        ax.add_patch(rect)
        label_face = ("#ffecec" if offending
                      else ds.box_face_overrides.get(b.label, ds.box_facecolor))
        label_bbox = (dict(boxstyle="round,pad=0.12", fc=label_face, ec="none")
                      if ds.label_casing else None)
        txt = ax.text(b.x, b.y, ds.box_label_map.get(b.label, b.label),
                      ha="center", va="center", fontsize=ds.box_label_fontsize,
                      zorder=4, bbox=label_bbox)
        if ds.svg_gids:
            gid, lgid = _semantic_gid(b, layout, ds)
            rect.set_gid(gid)
            txt.set_gid(lgid)

    # graphical N-linear bindings (§43): conservation drawn as a coupling arc over the box
    # linking the equal in/out legs of a type; a shared-key object split drawn as a brace
    # grouping the keyed output legs. (Single-box panels only -- the catalogue view.)
    if (couplings or keys) and len(layout.boxes) == 1:
        from matplotlib.path import Path as _MplPath

        b = layout.boxes[0]
        legpos: dict = {}  # port -> (leg_y, open_x)
        for w in layout.wires:
            if not w.boundary or w.port is None:
                continue
            ox, oy = w.open_end if w.open_end is not None else (w.x1, w.y1)
            legpos[w.port] = (oy, ox)
        top = b.y + b.half_h
        # nest multiple conservation arcs (one per conserved type) at staggered heights so
        # they don't collide with each other or the panel title; each carries an `=` badge.
        ncoup = len(couplings or [])
        for idx, (typ, in_p, out_p) in enumerate(couplings or []):
            if in_p not in legpos or out_p not in legpos:
                continue
            y_in, y_out = legpos[in_p][0], legpos[out_p][0]
            apex = top + 0.16 + 0.17 * idx
            path = _MplPath(
                [(b.x - _BW / 2, y_in), (b.x, apex), (b.x + _BW / 2, y_out)],
                [_MplPath.MOVETO, _MplPath.CURVE3, _MplPath.CURVE3],
            )
            ax.add_patch(mpatches.PathPatch(
                path, fill=False, lw=1.1, ls=":", edgecolor=cmap[typ], alpha=0.85, zorder=6))
            lx = b.x + (idx - (ncoup - 1) / 2) * 0.30  # stagger badges so they don't stack
            ax.text(lx, apex, "=", ha="center", va="center", fontsize=9, fontweight="bold",
                    color=_darken(cmap[typ]), zorder=7,
                    bbox=dict(boxstyle="square,pad=0.04", fc="white", ec="none"))
        for (typ, ports) in (keys or []):
            ys = [legpos[p][0] for p in ports if p in legpos]
            if len(ys) < 2:
                continue
            # brace hugging the box's right edge (clear of the far port-number labels)
            bx = b.x + _BW / 2 + 0.12
            ylo, yhi = min(ys), max(ys)
            ax.plot([bx, bx], [ylo, yhi], color=_darken(cmap[typ]), lw=1.4, zorder=6)
            for y in ys:
                ax.plot([bx - 0.1, bx], [y, y], color=_darken(cmap[typ]), lw=1.4, zorder=6)
            ax.text(bx + 0.06, (ylo + yhi) / 2, "⋈k", ha="left", va="center",
                    fontsize=7, fontweight="bold", color=_darken(cmap[typ]), zorder=7,
                    bbox=dict(boxstyle="square,pad=0.04", fc="white", ec="none"))

    handles = [
        mpatches.Patch(color=cmap[t], label=(t if t is not None else "untyped"))
        for t in sorted(legend_types, key=lambda x: (x is None, str(x)))
    ]
    ax.legend(
        handles=handles,
        title="object type",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        title_fontsize=8,
        frameon=False,
    )

    if owns_fig and numbered_ports:
        # number -> triple lookup under the diagram; offsets in points so the
        # line spacing is size-independent (bbox_inches="tight" picks it up).
        ncols = 1 if len(numbered_ports) <= 12 else 2
        nrows = -(-len(numbered_ports) // ncols)
        ax.annotate("ports  (number: src — type — tgt):", xy=(0, 0),
                    xycoords="axes fraction", xytext=(4, -12), textcoords="offset points",
                    fontsize=9, fontweight="bold", ha="left", va="top",
                    annotation_clip=False)
        for k, p in enumerate(numbered_ports):
            c, r = divmod(k, nrows)
            ax.annotate(f"{k}:  {p.src} — {p.typ} — {p.tgt}", xy=(0, 0),
                        xycoords="axes fraction",
                        xytext=(4 + c * 300, -26 - r * 12), textcoords="offset points",
                        fontsize=8.5, fontweight="bold", color=_darken(cmap[p.typ]),
                        ha="left", va="top", annotation_clip=False)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.1)
    ax.axis("off")
    full_title = title or ""
    if bad:
        full_title = (
            full_title + "  " if full_title else ""
        ) + "⋈ TYPE-BALANCE VIOLATION"
    if full_title:
        ax.set_title(full_title, fontsize=10, color="#d62728" if bad else "black")
    if owns_fig:  # never re-layout a caller-owned figure
        fig.tight_layout()
    return fig


def _binding_graphics(g) -> tuple[list, list]:
    """The generator's multi-leg §32 relations split for graphical rendering (§43):
    ``couplings`` = conservation ``↑X==↓X`` (one in-leg, one out-leg of a type) as
    ``(typ, in_port, out_port)``; ``keys`` = a shared-key object split (>=2 out-legs of a
    type partitioning an input) as ``(typ, [out_ports])``. Single-leg cardinality
    intervals stay on the wire (:func:`_leg_card_by_port`)."""
    couplings: list = []
    keys: list = []
    for c in g.constraints:
        if c.rel != "==" or c.rhs != 0 or len(c.terms) < 2:
            continue
        outs = [p for p, co in c.terms if co > 0 and p in g.right]
        ins = [p for p, co in c.terms if co < 0 and p in g.left]
        if not outs or not ins:
            continue
        typ = outs[0].typ
        if len(outs) == 1 and len(ins) == 1:
            couplings.append((typ, ins[0], outs[0]))
        elif len(outs) >= 2:
            keys.append((typ, sorted(outs, key=str)))
    return couplings, keys


def _leg_card_by_port(g) -> dict:
    """Per-leg §32 binding cardinality keyed by :class:`~procposets.cospan.signature.Port`,
    e.g. ``{order_in: "[2,5]"}`` -- the label drawn *on* that leg's wire. Only
    single-leg unit-coefficient (cardinality) constraints; a leg pinned to the default
    ``(1,1)`` is dropped (it is the implicit 1:1, not a real binding)."""
    by_port: dict = {}
    for c in g.constraints:
        terms = list(c.terms)
        if len(terms) != 1 or terms[0][1] != 1:
            continue
        p = terms[0][0]
        lo, hi = by_port.get(p, [0, None])
        if c.rel in (">=", "=="):
            lo = max(lo, c.rhs)
        if c.rel in ("<=", "=="):  # NB separate `if`: an `==` pins BOTH bounds
            hi = c.rhs if hi is None else min(hi, c.rhs)
        by_port[p] = [lo, hi]
    out: dict = {}
    for p, (lo, hi) in by_port.items():
        if (lo, hi) == (1, 1):
            continue
        out[p] = f"={lo}" if lo == hi else f"[{lo},{'*' if hi is None else hi}]"
    return out


def catalogue(
    sig: Signature,
    path: str | None = None,
    *,
    max_gens: int = 24,
    cols: int = 4,
    title: str | None = None,
    labels: bool = True,
    style: StringDiagramStyle | None = None,
):
    """Render a signature's generator cospans as a grid of string-diagram panels.

    Each panel is one generator ``g_{a,(P,S)}``. With ``labels`` (default
    ``True``) every unique boundary port ``(src, type, tgt)`` is assigned a number
    once; each wire end is tagged with its port number, and a single legend at the
    bottom lists the object-type colours and the number -> triple breakdown -- so
    what differs between same-activity generators is read off the small numbers
    rather than repeated triples. Shows up to ``max_gens`` generators; returns the
    Figure and saves it to ``path`` if given."""
    by_label = gens(sig)
    items = [(label, i) for label in sorted(by_label) for i in range(len(by_label[label]))]
    shown = items[:max_gens]
    n = len(shown)
    ttl = title or f"signature — {len(sig)} generators over {len(by_label)} labels"
    if n < len(items):
        ttl += f"  (showing first {n})"

    if n == 0:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "empty signature", ha="center", va="center")
        fig.suptitle(ttl)
        if path is not None:
            fig.savefig(path, dpi=130, bbox_inches="tight")
        return fig

    diagrams = [(label, i, D(sig, label, i)) for (label, i) in shown]

    # number every unique boundary port that appears in the shown generators
    port_set: set[Port] = set()
    for (label, i) in shown:
        g = pick(sig, label, i)
        port_set |= set(g.left) | set(g.right)
    ports = sorted(port_set, key=lambda p: (str(p.typ), p.src, p.tgt))
    wire_labels = {p: str(k) for k, p in enumerate(ports)} if labels else None
    types = sorted({p.typ for p in ports}, key=lambda x: (x is None, str(x)))
    cmap = _colour_map(set(types))

    cols = min(cols, n)
    rows = -(-n // cols)

    leg_cols = min(3, len(ports)) if labels else 1
    leg_rows = -(-len(ports) // leg_cols) if labels else 0
    leg_ratio = (0.9 + 0.34 * leg_rows) if labels else 0.6  # legend row height vs a panel row (2.8)

    fig = plt.figure(figsize=(cols * 3.6, rows * 2.8 + leg_ratio + 0.4))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[*([2.8] * rows), leg_ratio])

    for k, (label, i, d) in enumerate(diagrams):
        ax = fig.add_subplot(gs[k // cols, k % cols])
        g = pick(sig, label, i)
        # draw each constrained leg's N-linear range *on* its wire (replacing the bare
        # port number with e.g. "3 [2,5]"); unconstrained legs keep their number.
        panel_labels = wire_labels
        card_by_port = _leg_card_by_port(g)
        if wire_labels is not None and card_by_port:
            panel_labels = dict(wire_labels)
            for p, cstr in card_by_port.items():
                panel_labels[p] = f"{wire_labels.get(p, '')} {cstr}".strip()
        couplings, key_splits = _binding_graphics(g)
        render(d, sig, ax=ax, wire_labels=panel_labels, couplings=couplings, keys=key_splits, style=style)
        prof = binding_profile(g)
        # relations (conservation/key) are now drawn graphically on the panel; the title
        # stays a compact label (the long `bind:` text overflowed/overlapped neighbours).
        ax.set_title(f"{label} #{i}", fontsize=9, pad=16,  # pad: clear the conservation arcs
                     color=("#1a5e1a" if not prof.is_trivial() else "black"))
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()  # one shared legend at the bottom instead
    for k in range(n, rows * cols):
        fig.add_subplot(gs[k // cols, k % cols]).axis("off")

    lax = fig.add_subplot(gs[rows, :])
    lax.axis("off")
    tr = lax.transAxes
    x = 0.005
    lax.text(x, 0.9, "object type:", fontsize=9, fontweight="bold", va="center", transform=tr)
    x += 0.085
    for t in types:
        lax.add_patch(mpatches.Rectangle((x, 0.86), 0.012, 0.08, transform=tr,
                                          color=cmap[t], clip_on=False))
        name = t if t is not None else "untyped"
        lax.text(x + 0.016, 0.9, name, fontsize=9, va="center", transform=tr)
        x += 0.016 + 0.011 * len(str(name)) + 0.025
    if labels:
        lax.text(0.005, 0.72, "ports  (number: src — type — tgt):", fontsize=9,
                 fontweight="bold", va="center", transform=tr)
        col_w = 1.0 / leg_cols
        for k, p in enumerate(ports):
            c, r = divmod(k, leg_rows)
            lax.text(0.005 + c * col_w, 0.62 - r * (0.6 / max(1, leg_rows)),
                     f"{k}:  {p.src} — {p.typ} — {p.tgt}", fontsize=8.5,
                     color=_darken(cmap[p.typ]), fontweight="bold", va="center", transform=tr)

    fig.suptitle(ttl, fontsize=12)
    fig.subplots_adjust(top=0.93)  # keep the suptitle clear of row-0 panel titles
    if any(g.constraints for label in by_label for g in by_label[label]):
        fig.text(0.5, 0.004,
                 "bindings (objects per firing): on-wire [min,max]/=k cardinality (no label = 1:1);  "
                 "dotted '=' arc = object conservation (out == in);  '⋈k' brace = shared-key split "
                 "(the grouped legs partition the input)",
                 ha="center", va="bottom", fontsize=8, color="#1a5e1a")
    if path is not None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig
