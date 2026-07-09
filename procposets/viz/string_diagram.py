"""String-diagram visualiser for cospan signatures and their compositions.

A quick visual checker for ``Sigma`` and ``F(Sigma)``: each generator
(``def:generator-cospan-general``) renders as a labelled box whose left/right
typed :class:`~cpm.cospan.signature.Port` triples are coloured wires; ``;``
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
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import comb

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb


def _darken(colour, factor: float = 0.7):
    """Darker shade of ``colour`` for high-contrast (non-faded) coloured text."""
    r, g, b = to_rgb(colour)
    return (r * factor, g * factor, b * factor)

from ..cospan.compose import CompositeDiagram, LoopBox
from ..cospan.signature import Generator, Port, Signature
from ..cospan.signature_compare import binding_profile
from ..cospan.typebalance import Kappa, _delta

# --- style knobs (module-level so a demo can override before rendering) ------
STRAIGHT_SPINE = False  # True: `;` chains stay on one horizontal spine (no
# waterfall drift) and `⊗` stacks are centred on it
BOUNDARY_LINESTYLE = "--"  # linestyle for dangling boundary stubs
INTERNAL_WIRE_LINESTYLE = "-"  # box-to-box wires; presentation renders may dash
# these to expose the gluing seams between composed generators
OPEN_END_MARKERS = True  # circle marker on each dangling boundary tip
BOX_FACECOLOR = "white"  # default box face ("none" -> see-through boxes)
BOX_FACE_OVERRIDES: dict[str, str] = {}  # box label -> facecolor (e.g. to
# flag a compactified-loop power box)
BOX_LABEL_MAP: dict[str, str] = {}  # box label -> display label (e.g. a
# 4-char code; pair with a legend mapping code -> activity)
BOX_LABEL_FONTSIZE: float = 9  # box/result label size; presentation renders bump this
# for legibility once figures are scaled down into slides / scrollytelling
PORT_ORDER_KEY = None  # optional callable Port -> sort key controlling the
# bottom-to-top port order on a box edge; when a demo's wires have a known
# vertical layout (e.g. lab branches above the spine, img below), matching
# the port order to it removes avoidable wire crossings
ALIGN_BOUNDARY_STUBS = False  # True: all dangling boundary stubs on a side
# extend to one common x (the diagram's outermost open end) instead of a
# fixed per-box margin -- open legs then start/end flush left/right
SVG_GIDS = False  # stamp semantic SVG ids (gid) on every artist:
# box-<activity>/boxlabel-<activity>, node-<result>-r<round>/nodelabel-...,
# block-<fN>/blocklabel-<fN>, wire-<type>-s<n>. Roles come from GID_ROLES;
# round numbers from the x-rank of result columns (+ GID_ROUND_OFFSET).
# Enable only for single-diagram figures -- ids must be unique per SVG.
GID_ROLES: dict[str, str] = {}  # box label -> 'activity'|'result'|'block'|'boundary'
GID_ROUND_OFFSET = 0  # added to computed round numbers (multi-call figures)
TYPE_LANES: dict[str | None, float] | None = None  # object type -> fixed
# wire lane (y offset). When set, every port sits exactly on its type's lane
# and boxes centre over their ports, so all wires run straight and
# horizontal. Sound only while no generator carries two same-typed ports on
# one edge (duplicates are nudged apart by _PS as a fallback).

# --- geometry constants -----------------------------------------------------
_BW = 1.4  # box width
_PS = 0.6  # vertical spacing between ports on a box edge
_BH = 0.7  # minimum box height
_HGAP = 1.4  # horizontal gap between sequential columns
_VGAP = 0.8  # vertical gap between parallel rows
_MARGIN = 0.9  # length of dangling boundary wires
_BOX_PAD = 0.02  # FancyBboxPatch boxstyle pad: the drawn border sits this far
# outside the nominal box rect, so ports land on the border path (not the
# nominal edge) and wires finish cleanly on the visible boundary

# --- composite (DAG) layout: layered, Sugiyama-style ------------------------
_LANE_BASE = _BH / 2 + 0.55  # first long-edge lane offset above/below the spine
_LANE_STEP = 0.55  # spacing between stacked lanes on one side
_RISER = _HGAP * 0.45  # horizontal run before a long edge reaches its lane

# --- bezier wire rendering + braid-style crossing breaks --------------------
_CURVE_N = 120  # samples per rendered curve (dense, so the constant-radius
# break below lands on real geometry, not a coarse polyline)
_GAP = 0.11  # Euclidean radius (data units) of the disk erased from the
# "under" wire at a crossing -- constant, so every break looks
# the same size whatever the crossing angle
_TRIM = 0.06  # ignore crossings within this fraction of either curve's ends

# --- type -> colour ---------------------------------------------------------
# Red/black/blue variants from the "Red, Black & Blue" palette
# (color-hex.com/color-palette/26562): one variant of each picked for the
# loop-family ports; _GREY is an unused spare, not from that palette.
_ALIZARIN = "#D2292D"  # red variant
_OLD_BLACK = "#282828"  # black variant
_COTTON_BLUE = "#1761B0"  # blue variant
_GREY = "#9E9E9E"  # unused spare

_PALETTE: dict[str | None, str] = {
    "pat": "#1f77b4",  # patient  -- blue
    "lab": "#2ca02c",  # lab      -- green
    "img": "#ff7f0e",  # imaging  -- orange
    "bed": "#d62728",  # bed      -- red
    "alpha": _ALIZARIN,  # loop-family alpha -- red
    "beta": _COTTON_BLUE,  # loop-family beta  -- blue
    "gamma": _OLD_BLACK,  # loop-family gamma -- black
    None: "#888888",  # untyped  -- grey
}
_FALLBACK = ["#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]


def _colour_map(types: set[str | None]) -> dict[str | None, str]:
    out = dict(_PALETTE)
    i = 0
    for t in sorted((x for x in types if x not in out), key=str):
        out[t] = _FALLBACK[i % len(_FALLBACK)]
        i += 1
    return out


# --- placed geometry (backend-independent) ----------------------------------
@dataclass(frozen=True)
class PlacedBox:
    label: str
    x: float
    y: float
    half_h: float
    created: frozenset = frozenset()  # object types this box mints (right-only)
    consumed: frozenset = frozenset()  # object types this box absorbs (left-only)


@dataclass(frozen=True)
class Wire:
    x1: float
    y1: float
    x2: float
    y2: float
    typ: str | None
    port: Port | None = None  # None for loop-box stubs
    boundary: bool = False
    # extra interior through-points (lane-routed long edges) so the whole wire
    # renders as one smooth higher-degree bezier instead of straight-line legs
    # glued end to end; ``None`` means the plain 2-point S-curve.
    waypoints: tuple[tuple[float, float], ...] | None = None
    # for a boundary stub, which end is the *open* (dangling, away-from-box)
    # tip -- (x1,y1) and (x2,y2) alone can't tell, since both an input stub
    # (box-edge at x2) and an output stub (box-edge at x1) are built with
    # x1<x2. ``None`` for internal wires (not used).
    open_end: tuple[float, float] | None = None


@dataclass
class Layout:
    boxes: list[PlacedBox] = field(default_factory=list)
    wires: list[Wire] = field(default_factory=list)
    types: set = field(default_factory=set)


# --- sub-layout used while composing the term tree --------------------------
@dataclass
class _Sub:
    boxes: list[PlacedBox]
    wires: list[Wire]
    in_eps: list[tuple[Port, float, float]]  # exposed left interface
    out_eps: list[tuple[Port, float, float]]  # exposed right interface
    xmin: float
    xmax: float

    def shift(self, dx: float, dy: float) -> "_Sub":
        return _Sub(
            [
                PlacedBox(b.label, b.x + dx, b.y + dy, b.half_h, b.created, b.consumed)
                for b in self.boxes
            ],
            [
                Wire(
                    w.x1 + dx,
                    w.y1 + dy,
                    w.x2 + dx,
                    w.y2 + dy,
                    w.typ,
                    w.port,
                    w.boundary,
                    open_end=None
                    if w.open_end is None
                    else (w.open_end[0] + dx, w.open_end[1] + dy),
                )
                for w in self.wires
            ],
            [(p, x + dx, y + dy) for (p, x, y) in self.in_eps],
            [(p, x + dx, y + dy) for (p, x, y) in self.out_eps],
            self.xmin + dx,
            self.xmax + dx,
        )


def _gen_delta(g: Generator) -> tuple[frozenset, frozenset]:
    left = {p.typ for p in g.left if p.typ is not None}
    right = {p.typ for p in g.right if p.typ is not None}
    created, consumed = _delta(left, right)
    return frozenset(created), frozenset(consumed)


def _ports(item: "Generator | LoopBox") -> tuple[frozenset, frozenset]:
    """Effective ``(left, right)`` for layout purposes.  A real generator's
    own ports, or -- for an ``f^(n)`` :class:`LoopBox` -- what its first
    iteration needs in and its last iteration produces out, so the collapsed
    repeat is laid out exactly like one n-to-m generator rather than a
    disconnected stub."""
    if isinstance(item, LoopBox):
        return item.body[0].left, item.body[-1].right
    return item.left, item.right


def _box_sub(g: Generator) -> _Sub:
    # TODO(generic): this "tap" suffix is a naming-convention hack tied to
    # loop_family's tap ports, not a generic ordering mechanism. A proper fix
    # would let Generator carry explicit port order (e.g. ordered sequences
    # instead of frozensets) so layout doesn't have to guess intent from
    # port-tgt strings.
    if PORT_ORDER_KEY is not None:
        ins = sorted(g.left, key=lambda p: (PORT_ORDER_KEY(p), str(p)))
        outs = sorted(g.right, key=lambda p: (PORT_ORDER_KEY(p), str(p)))
    else:
        ins = sorted(g.left)
        outs = sorted(g.right, key=lambda p: (p.tgt == "tap", p))
    created, consumed = _gen_delta(g)

    if TYPE_LANES is not None:
        def lane_col(ports, x):
            out, used = [], {}
            for p in sorted(ports, key=lambda q: (TYPE_LANES.get(q.typ, 0.0), str(q))):
                y = TYPE_LANES.get(p.typ, 0.0)
                k = used.get(y, 0)
                used[y] = k + 1
                out.append((p, x, y + k * _PS))  # nudge same-lane duplicates
            return out

        in_eps = lane_col(ins, -_BW / 2 - _BOX_PAD)
        out_eps = lane_col(outs, _BW / 2 + _BOX_PAD)
        ys = [y for (_, _, y) in in_eps + out_eps] or [0.0]
        centre = (max(ys) + min(ys)) / 2
        half_h = max((max(ys) - min(ys)) / 2 + _BH / 2, _BH / 2)
        return _Sub(
            [PlacedBox(g.label, 0.0, centre, half_h, created, consumed)],
            [],
            in_eps,
            out_eps,
            -_BW / 2,
            _BW / 2,
        )

    n = max(len(ins), len(outs), 1)
    half_h = max((n - 1) * _PS / 2 + _BH / 2, _BH / 2)

    def col(ports, x):
        k = len(ports)
        return [(p, x, (i - (k - 1) / 2) * _PS) for i, p in enumerate(ports)]

    in_eps = col(ins, -_BW / 2 - _BOX_PAD)
    out_eps = col(outs, _BW / 2 + _BOX_PAD)
    return _Sub(
        [PlacedBox(g.label, 0.0, 0.0, half_h, created, consumed)],
        [],
        in_eps,
        out_eps,
        -_BW / 2,
        _BW / 2,
    )


def _seq(a: _Sub, b: _Sub) -> _Sub:
    dx = (a.xmax - b.xmin) + _HGAP
    # align b vertically on the bundle of ports it shares with a
    matched = [
        (pa, ya, pb, yb)
        for (pa, _xa, ya) in a.out_eps
        for (pb, _xb, yb) in b.in_eps
        if pa == pb
    ]
    if STRAIGHT_SPINE:
        dy = 0.0
    elif matched:
        raw = (sum(m[1] for m in matched) - sum(m[3] for m in matched)) / len(matched)
        # TODO(generic): the extra +-_PS below doubles the vertical waterfall
        # step for loop_family's a-chain (per request) by adding one more
        # constant unit of drift in the same direction. It must be additive,
        # not a multiplier on `raw` -- `raw` already includes the cumulative
        # drift from earlier boxes in the chain, so scaling it compounds
        # geometrically instead of doubling the step linearly. This is a
        # global tweak to every sequential alignment, not scoped to "a"
        # boxes specifically -- fine while loop_family is the only consumer
        # of this layout path, but a real fix would make the waterfall step
        # an explicit per-chain parameter instead of a blanket offset here.
        dy = raw + (_PS if raw > 0 else (-_PS if raw < 0 else 0.0))
    else:
        dy = 0.0
    b = b.shift(dx, dy)
    used_a = set()
    used_b = set()
    wires = list(a.wires) + list(b.wires)
    for i, (pa, xa, ya) in enumerate(a.out_eps):
        for j, (pb, xb, yb) in enumerate(b.in_eps):
            if j in used_b or i in used_a:
                continue
            if pa == pb:
                wires.append(Wire(xa, ya, xb, yb, pa.typ, pa))
                used_a.add(i)
                used_b.add(j)
                break
    in_eps = list(a.in_eps) + [
        b.in_eps[j] for j in range(len(b.in_eps)) if j not in used_b
    ]
    out_eps = [a.out_eps[i] for i in range(len(a.out_eps)) if i not in used_a] + list(
        b.out_eps
    )
    return _Sub(a.boxes + b.boxes, wires, in_eps, out_eps, a.xmin, b.xmax)


def _par(a: _Sub, b: _Sub) -> _Sub:
    if TYPE_LANES is not None:
        # the type lanes already separate the two blocks vertically; just
        # left-align them without stacking (stacking would pull ports off
        # their lanes and re-introduce wire bends)
        b = b.shift(a.xmin - b.xmin, 0.0)
        return _Sub(
            a.boxes + b.boxes,
            a.wires + b.wires,
            a.in_eps + b.in_eps,
            a.out_eps + b.out_eps,
            min(a.xmin, b.xmin),
            max(a.xmax, b.xmax),
        )
    a_bot = min([by - bx.half_h for bx in a.boxes for by in (bx.y,)], default=0.0)
    b_top = max([by + bx.half_h for bx in b.boxes for by in (bx.y,)], default=0.0)
    dy = a_bot - b_top - _VGAP
    dx = a.xmin - b.xmin  # left-align the two blocks
    b = b.shift(dx, dy)
    out = _Sub(
        a.boxes + b.boxes,
        a.wires + b.wires,
        a.in_eps + b.in_eps,
        a.out_eps + b.out_eps,
        min(a.xmin, b.xmin),
        max(a.xmax, b.xmax),
    )
    if STRAIGHT_SPINE and out.boxes:
        top = max(bx.y + bx.half_h for bx in out.boxes)
        bot = min(bx.y - bx.half_h for bx in out.boxes)
        out = out.shift(0.0, -(top + bot) / 2)  # centre the stack on the spine
    return out


# --- term DSL ---------------------------------------------------------------
class Diagram:
    """A ``;``/``⊗`` term over generators.  Compose with ``>>`` and ``@``."""

    def __init__(self, build):
        self._build = build  # () -> _Sub

    def __rshift__(self, other: "Diagram") -> "Diagram":  # self ; other
        return Diagram(lambda: _seq(self._build(), other._build()))

    def __matmul__(self, other: "Diagram") -> "Diagram":  # self ⊗ other
        return Diagram(lambda: _par(self._build(), other._build()))

    def _sub(self) -> _Sub:
        return self._build()


def pick(sig: Signature, label: str, i: int | None = None) -> Generator:
    """The generator labelled ``label`` (the ``i``-th when there are several,
    ordered deterministically by their string form)."""
    matches = sorted(sig.by_label(label), key=str)
    if not matches:
        raise KeyError(f"no generator labelled {label!r}")
    if i is None:
        if len(matches) > 1:
            opts = "\n".join(f"  [{k}] {g}" for k, g in enumerate(matches))
            raise ValueError(
                f"{label!r} is ambiguous ({len(matches)} generators); pass i=:\n{opts}"
            )
        return matches[0]
    return matches[i]


def D(sig: Signature, label: str, i: int | None = None) -> Diagram:
    """Convenience: a single-box :class:`Diagram` for ``pick(sig, label, i)``."""
    g = pick(sig, label, i)
    return Diagram(lambda: _box_sub(g))


def gens(sig: Signature) -> dict[str, list[Generator]]:
    """Map each label to its generators (ordered as :func:`pick` indexes them)."""
    return {lab: sorted(sig.by_label(lab), key=str) for lab in sorted(sig.labels())}


# --- lowering to a Layout ---------------------------------------------------
def _finish(sub: _Sub) -> Layout:
    wires = list(sub.wires)
    types = {w.typ for w in wires}
    # with ALIGN_BOUNDARY_STUBS all open ends on a side share the diagram's
    # outermost x, so every open leg starts/ends flush left/right
    lx = min((x for _, x, _ in sub.in_eps), default=0.0) - _MARGIN
    rx = max((x for _, x, _ in sub.out_eps), default=0.0) + _MARGIN
    for p, x, y in sub.in_eps:  # dangling boundary inputs
        x0 = lx if ALIGN_BOUNDARY_STUBS else x - _MARGIN
        wires.append(
            Wire(x0, y, x, y, p.typ, p, boundary=True, open_end=(x0, y))
        )
        types.add(p.typ)
    for p, x, y in sub.out_eps:  # dangling boundary outputs
        x1 = rx if ALIGN_BOUNDARY_STUBS else x + _MARGIN
        wires.append(
            Wire(x, y, x1, y, p.typ, p, boundary=True, open_end=(x1, y))
        )
        types.add(p.typ)
    return Layout(sub.boxes, wires, types)


def _layout_composite(comp: CompositeDiagram) -> Layout:
    """Layered (Sugiyama-style) layout of a composite run.

    Three stages: (1) **columns** by longest-path depth on the port-production
    DAG; (2) **barycentre ordering** within each column over a few sweeps to
    cut wire crossings -- the boundary ``▷``/``□`` stacks reorder so their wires
    leave/enter without crossing; (3) **lane routing** for *long* edges (those
    spanning more than one column): instead of one diagonal slashing across
    every box in between, each is laid flat through a reserved lane just
    above/below the spine, lanes packed by greedy interval colouring.  Short
    (adjacent-column) edges stay straight; boundary legs stay dashed stubs."""
    placements = comp.placements
    n = len(placements)

    # (1) columns + edge list, forward over the port-production DAG ----------
    # an ``f^(n)`` LoopBox is laid out exactly like an n-to-m generator (see
    # ``_ports``): it gets a real left edge from whatever produced the port
    # its first iteration consumes, and its own unconsumed right ports become
    # open boundary stubs the same way a generator's would.
    pool: dict[Port, int] = {}  # port -> producing box index
    col = [0] * n
    edges: list[tuple[int, int, Port]] = []  # (src_box, dst_box, port)
    bin_stub: list[tuple[int, Port]] = []  # boundary inputs (dst, port)
    for i, item in enumerate(placements):
        left, right = _ports(item)
        depth = 0
        for p in sorted(left):
            if p in pool:
                s = pool[p]
                edges.append((s, i, p))
                depth = max(depth, col[s] + 1)
            else:
                bin_stub.append((i, p))
        col[i] = depth
        for p in sorted(right):
            pool[p] = i
    bout_stub = [
        (i, p)
        for i, item in enumerate(placements)
        for p in sorted(_ports(item)[1])
        if not _consumed_later(comp, i, p)
    ]

    # each box's height depends on its own port count (an ``f^(n)`` LoopBox can
    # be much taller than its neighbours), so row spacing must stack by actual
    # half-heights rather than a fixed pitch -- otherwise a tall box overlaps
    # the row above/below it.
    half_h_of = [
        max(
            (max(len(_ports(item)[0]), len(_ports(item)[1]), 1) - 1) * _PS / 2
            + _BH / 2,
            _BH / 2,
        )
        for item in placements
    ]

    # (2a) lanes for long edges -- depends only on column ranges, not on y ----
    # ``edges`` is already in deterministic placement order; sort is stable, so
    # ties (equal span) keep that order instead of falling back to a set's
    # hash-seed-dependent iteration (which made lane/side assignment, and hence
    # crossing geometry, vary run-to-run).
    long_edges = [(u, v, p) for (u, v, p) in edges if col[v] - col[u] >= 2]
    lane_of: dict[tuple[int, int, Port], float] = {}
    side_levels: dict[int, list[list[tuple[int, int]]]] = {+1: [], -1: []}
    count = {+1: 0, -1: 0}
    for e in sorted(long_edges, key=lambda e: col[e[1]] - col[e[0]], reverse=True):
        u, v = e[0], e[1]
        side = +1 if count[+1] <= count[-1] else -1
        count[side] += 1
        rng = (col[u], col[v])
        levels = side_levels[side]
        lvl = next(
            (
                k
                for k, occ in enumerate(levels)
                if all(rng[1] <= a or b <= rng[0] for (a, b) in occ)
            ),
            None,
        )
        if lvl is None:
            lvl = len(levels)
            levels.append([])
        levels[lvl].append(rng)
        lane_of[e] = side * (_LANE_BASE + lvl * _LANE_STEP)

    # (2b) barycentre ordering within each column -----------------------------
    by_col: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        by_col[col[i]].append(i)
    order = {c: list(idxs) for c, idxs in by_col.items()}  # top -> bottom

    nbrs: list[list[tuple[str, float]]] = [[] for _ in range(n)]
    for e in edges:  # long edge pulls toward its lane
        u, v = e[0], e[1]
        if e in lane_of:
            nbrs[u].append(("lane", lane_of[e]))
            nbrs[v].append(("lane", lane_of[e]))
        else:
            nbrs[u].append(("box", v))
            nbrs[v].append(("box", u))

    def assign_y() -> list[float]:
        y = [0.0] * n
        for idxs in order.values():
            heights = [half_h_of[i] for i in idxs]
            total = sum(2 * h for h in heights) + _VGAP * (len(idxs) - 1)
            cur = total / 2
            for i, h in zip(idxs, heights):
                cur -= h
                y[i] = cur
                cur -= h + _VGAP
        return y

    y = assign_y()
    for _ in range(8):
        for c in order:
            if len(order[c]) < 2:
                continue

            def key(i: int, _y: list[float] = y) -> float:
                vals = [v if kind == "lane" else _y[int(v)] for kind, v in nbrs[i]]
                return sum(vals) / len(vals) if vals else _y[i]

            order[c] = sorted(order[c], key=key, reverse=True)  # high barycentre on top
        y = assign_y()

    # (3) place boxes and collect port endpoints ------------------------------
    boxes: list[PlacedBox] = []
    left_ep: dict[tuple[int, Port], tuple[float, float]] = {}
    right_ep: dict[tuple[int, Port], tuple[float, float]] = {}
    for i, item in enumerate(placements):
        bx = col[i] * (_BW + _HGAP)
        by = y[i]
        ins, outs = (sorted(side) for side in _ports(item))
        created, consumed = (
            _gen_delta(item)
            if isinstance(item, Generator)
            else (frozenset(), frozenset())
        )
        boxes.append(PlacedBox(item.label, bx, by, half_h_of[i], created, consumed))
        for k, p in enumerate(ins):
            left_ep[(i, p)] = (bx - _BW / 2 - _BOX_PAD, by + (k - (len(ins) - 1) / 2) * _PS)
        for k, p in enumerate(outs):
            right_ep[(i, p)] = (bx + _BW / 2 + _BOX_PAD, by + (k - (len(outs) - 1) / 2) * _PS)

    # wires: long edges get lane through-points (one smooth curve bends out to
    # the lane and back), short edges are a plain point-to-point S-curve -------
    wires: list[Wire] = []
    types: set = set()
    for e in edges:
        u, v, p = e
        x1, y1 = right_ep[(u, p)]
        x2, y2 = left_ep[(v, p)]
        if e in lane_of:
            ly = lane_of[e]
            r = min(_RISER, (x2 - x1) / 4)
            wp = (
                (x1, y1),
                (x1 + r, y1),
                (x1 + 2 * r, ly),
                (x2 - 2 * r, ly),
                (x2 - r, y2),
                (x2, y2),
            )
            wires.append(Wire(x1, y1, x2, y2, p.typ, p, waypoints=wp))
        else:
            wires.append(Wire(x1, y1, x2, y2, p.typ, p))
        types.add(p.typ)
    for i, p in bin_stub:
        x, ey = left_ep[(i, p)]
        wires.append(
            Wire(
                x - _MARGIN,
                ey,
                x,
                ey,
                p.typ,
                p,
                boundary=True,
                open_end=(x - _MARGIN, ey),
            )
        )
        types.add(p.typ)
    for i, p in bout_stub:
        x, ey = right_ep[(i, p)]
        wires.append(
            Wire(
                x,
                ey,
                x + _MARGIN,
                ey,
                p.typ,
                p,
                boundary=True,
                open_end=(x + _MARGIN, ey),
            )
        )
        types.add(p.typ)
    return Layout(boxes, wires, types)


def _consumed_later(comp: CompositeDiagram, i: int, port: Port) -> bool:
    """Whether some placement *strictly after* position ``i`` still needs
    ``port`` -- position-aware so a looping generator that reuses the same
    port identity at an earlier position doesn't falsely "consume" a port
    produced later (e.g. an ``f^(n)`` LoopBox's own output)."""
    for item in comp.placements[i + 1 :]:
        if port in _ports(item)[0]:
            return True
    return False


import re as _re


def _gid_sanitize(label: str) -> str:
    """Stable id token: non-id characters collapsed to '_' (f2^(m-1) -> f2_m-1)."""
    return _re.sub(r"[^A-Za-z0-9_.:-]+", "_", label).strip("_")


def _semantic_gid(b: PlacedBox, layout: Layout) -> tuple[str, str]:
    """(element id, label id) for one placed box, per the animation SVG spec."""
    role = GID_ROLES.get(b.label, "activity")
    base = _gid_sanitize(b.label)
    if role == "result":
        cols = sorted({round(x.x, 6) for x in layout.boxes
                       if GID_ROLES.get(x.label) == "result"})
        r = cols.index(round(b.x, 6)) + 1 + GID_ROUND_OFFSET
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


def _draw_wires(ax, wires: list["Wire"], cmap: dict) -> None:
    internal = [w for w in wires if not w.boundary]
    curves = [_curve_for(w) for w in internal]
    lengths = [_arclen(c)[-1] for c in curves]
    breaks: list[list[np.ndarray]] = [[] for _ in internal]
    for i in range(len(internal)):
        for j in range(i + 1, len(internal)):
            for s_i, s_j in _curve_crossings(curves[i], curves[j]):
                lo_i, hi_i = _TRIM * lengths[i], (1 - _TRIM) * lengths[i]
                lo_j, hi_j = _TRIM * lengths[j], (1 - _TRIM) * lengths[j]
                if not (lo_i <= s_i <= hi_i and lo_j <= s_j <= hi_j):
                    continue  # too close to a port -- not a real crossing
                # the longer-spanning (more "background") wire passes under
                under = i if lengths[i] >= lengths[j] else j
                breaks[under].append(
                    _point_at(curves[under], s_i if under == i else s_j)
                )
    def _stamp(line, typ):
        if not SVG_GIDS:
            return
        cnt = ax.figure.__dict__.setdefault("_wire_gid_counts", {})
        cnt[typ] = cnt.get(typ, 0) + 1
        line.set_gid(f"wire-{typ}-s{cnt[typ]}")

    for w, pts, centres in zip(internal, curves, breaks):
        for run in _split_at_disks(pts, centres):
            (ln,) = ax.plot(
                run[:, 0],
                run[:, 1],
                color=cmap[w.typ],
                lw=2.2,
                solid_capstyle="butt",
                zorder=1,
                ls=INTERNAL_WIRE_LINESTYLE,
            )
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
            ls=BOUNDARY_LINESTYLE,
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
    :class:`~matplotlib.figure.Figure`."""
    if isinstance(obj, Diagram):
        sub = obj._sub()
        if offset is not None:
            sub = sub.shift(*offset)
        layout = _finish(sub)
    elif isinstance(obj, CompositeDiagram):
        layout = _layout_composite(obj)
    else:
        raise TypeError(f"cannot render {type(obj).__name__}")

    legend_types = set(layout.types)
    if signature is not None:
        legend_types |= {p.typ for p in signature.ports()}
    cmap = _colour_map(legend_types)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, 1.6 * (len(layout.boxes) + 1)), 4))
        owns_fig = True
    else:
        fig = ax.figure
        owns_fig = False

    _draw_wires(ax, layout.wires, cmap)
    for w in layout.wires:
        if not w.boundary:
            continue
        ex, ey = w.open_end if w.open_end is not None else (w.x1, w.y1)
        if OPEN_END_MARKERS:
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
            else BOX_FACE_OVERRIDES.get(b.label, BOX_FACECOLOR),
            zorder=3,
        )
        ax.add_patch(rect)
        txt = ax.text(b.x, b.y, BOX_LABEL_MAP.get(b.label, b.label),
                      ha="center", va="center", fontsize=BOX_LABEL_FONTSIZE, zorder=4)
        if SVG_GIDS:
            gid, lgid = _semantic_gid(b, layout)
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
    """Per-leg §32 binding cardinality keyed by :class:`~cpm.cospan.signature.Port`,
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
        render(d, sig, ax=ax, wire_labels=panel_labels, couplings=couplings, keys=key_splits)
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
