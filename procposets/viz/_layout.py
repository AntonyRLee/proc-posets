"""Backend-independent layout for string diagrams (the geometry half of
:mod:`procposets.viz.string_diagram`).

Everything here is **matplotlib-free**: the term DSL (:class:`Diagram`), the
placed-geometry datatypes (:class:`PlacedBox`, :class:`Wire`, :class:`Layout`),
the layout style knobs (:class:`LayoutStyle`), and the two lowering paths
(:func:`_finish` for a term tree, :func:`_layout_composite` for a
``CompositeDiagram`` run).  The matplotlib drawing side lives in
``string_diagram.py`` and imports from here; a TikZ backend could be added
against the same geometry.  ``string_diagram`` re-exports every public (and
externally-referenced private) name from this module, so
``string_diagram.NAME`` keeps resolving unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from ..cospan.compose import CompositeDiagram, LoopBox
from ..cospan.signature import Generator, Port, Signature
from ..cospan.typebalance import _delta


# --- layout style config ----------------------------------------------------
# The layout-affecting subset of the 14 style knobs (the draw knobs live in the
# DrawStyle dataclass in string_diagram.py). Read only by the geometry functions
# below, so this module stays matplotlib-free.
@dataclass(frozen=True)
class LayoutStyle:
    straight_spine: bool = False
    port_order_key: Callable[[Port], object] | None = None
    align_boundary_stubs: bool = False
    type_lanes: dict[str | None, float] | None = None


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


def _box_sub(g: Generator, style: LayoutStyle) -> _Sub:
    # TODO(generic): this "tap" suffix is a naming-convention hack tied to
    # loop_family's tap ports, not a generic ordering mechanism. A proper fix
    # would let Generator carry explicit port order (e.g. ordered sequences
    # instead of frozensets) so layout doesn't have to guess intent from
    # port-tgt strings.
    if style.port_order_key is not None:
        ins = sorted(g.left, key=lambda p: (style.port_order_key(p), str(p)))
        outs = sorted(g.right, key=lambda p: (style.port_order_key(p), str(p)))
    else:
        ins = sorted(g.left)
        outs = sorted(g.right, key=lambda p: (p.tgt == "tap", p))
    created, consumed = _gen_delta(g)

    if style.type_lanes is not None:
        def lane_col(ports, x):
            out, used = [], {}
            for p in sorted(ports, key=lambda q: (style.type_lanes.get(q.typ, 0.0), str(q))):
                y = style.type_lanes.get(p.typ, 0.0)
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


def _seq(a: _Sub, b: _Sub, style: LayoutStyle) -> _Sub:
    dx = (a.xmax - b.xmin) + _HGAP
    # align b vertically on the bundle of ports it shares with a
    matched = [
        (pa, ya, pb, yb)
        for (pa, _xa, ya) in a.out_eps
        for (pb, _xb, yb) in b.in_eps
        if pa == pb
    ]
    if style.straight_spine:
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


def _par(a: _Sub, b: _Sub, style: LayoutStyle) -> _Sub:
    if style.type_lanes is not None:
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
    if style.straight_spine and out.boxes:
        top = max(bx.y + bx.half_h for bx in out.boxes)
        bot = min(bx.y - bx.half_h for bx in out.boxes)
        out = out.shift(0.0, -(top + bot) / 2)  # centre the stack on the spine
    return out


# --- term DSL ---------------------------------------------------------------
class Diagram:
    """A ``;``/``⊗`` term over generators.  Compose with ``>>`` and ``@``."""

    def __init__(self, build):
        self._build = build  # (LayoutStyle) -> _Sub

    def __rshift__(self, other: "Diagram") -> "Diagram":  # self ; other
        return Diagram(lambda st: _seq(self._build(st), other._build(st), st))

    def __matmul__(self, other: "Diagram") -> "Diagram":  # self ⊗ other
        return Diagram(lambda st: _par(self._build(st), other._build(st), st))

    def _sub(self, style: LayoutStyle) -> _Sub:
        return self._build(style)


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
    return Diagram(lambda st: _box_sub(g, st))


def gens(sig: Signature) -> dict[str, list[Generator]]:
    """Map each label to its generators (ordered as :func:`pick` indexes them)."""
    return {lab: sorted(sig.by_label(lab), key=str) for lab in sorted(sig.labels())}


# --- lowering to a Layout ---------------------------------------------------
def _finish(sub: _Sub, style: LayoutStyle) -> Layout:
    wires = list(sub.wires)
    types = {w.typ for w in wires}
    # with align_boundary_stubs all open ends on a side share the diagram's
    # outermost x, so every open leg starts/ends flush left/right
    lx = min((x for _, x, _ in sub.in_eps), default=0.0) - _MARGIN
    rx = max((x for _, x, _ in sub.out_eps), default=0.0) + _MARGIN
    for p, x, y in sub.in_eps:  # dangling boundary inputs
        x0 = lx if style.align_boundary_stubs else x - _MARGIN
        wires.append(
            Wire(x0, y, x, y, p.typ, p, boundary=True, open_end=(x0, y))
        )
        types.add(p.typ)
    for p, x, y in sub.out_eps:  # dangling boundary outputs
        x1 = rx if style.align_boundary_stubs else x + _MARGIN
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
