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
from dataclasses import dataclass, field, replace
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
    box_width: float = 1.4  # term-path box width (== _BW default); shrink it for
    # tight diagrams once labels are abbreviated (:class:`DrawStyle.abbreviate_labels`).
    # Stored per box as ``PlacedBox.half_w`` so the post-passes stay style-free.
    crossing_min: bool = False  # term path: reassign each box's port slots to
    # its neighbours' vertical order (:func:`_optimize_ports`), cutting the
    # port-order-mismatch crossings the static ``sorted`` port order leaves.
    crossing_min_iters: int = 16  # port-reassignment passes (both wire ends move)
    route_long_edges: bool = True  # term path: any wire that would pass *behind*
    # an intervening box is re-routed through a reserved lane in the gap
    # above/below those boxes (:func:`_route_long_edges`), so no wire is ever
    # drawn through/behind a box (readability problem (a)). Default on: a wire
    # behind a box is never intended. No-op on <3-box diagrams (catalogue panels).
    straighten: bool = False  # term path: relax each box's y toward the mean of
    # its wired neighbours (:func:`_straighten_boxes`) so a ;-chain's spine is
    # horizontal and wires stop weaving (continuity, readability problem (d)).
    # OPT-IN: the loop/waterfall figures use *intentional* vertical drift (the
    # ``_seq`` step), which this would flatten -- enable only where wanted.
    straighten_iters: int = 12
    simplify_rounds: int = 4  # term path: how many times to iterate the
    # straighten+re-slot simplification in :func:`lower_term`. Each round's port
    # re-assignment (crossing removal) shifts the geometry, which lets the next
    # straighten level more wires / expose more removable crossings -- so the
    # passes compound. Stops early once the layout stops changing (fixed point).


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
    half_w: float = _BW / 2  # half box width (defaults to the module _BW; the term
    # path sets it from LayoutStyle.box_width so the post-passes read it per box)


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
                PlacedBox(b.label, b.x + dx, b.y + dy, b.half_h, b.created,
                          b.consumed, b.half_w)
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
    hw = style.box_width / 2  # per-figure box half-width (LayoutStyle.box_width)

    if style.type_lanes is not None:
        def lane_col(ports, x):
            out, used = [], {}
            for p in sorted(ports, key=lambda q: (style.type_lanes.get(q.typ, 0.0), str(q))):
                y = style.type_lanes.get(p.typ, 0.0)
                k = used.get(y, 0)
                used[y] = k + 1
                out.append((p, x, y + k * _PS))  # nudge same-lane duplicates
            return out

        in_eps = lane_col(ins, -hw - _BOX_PAD)
        out_eps = lane_col(outs, hw + _BOX_PAD)
        ys = [y for (_, _, y) in in_eps + out_eps] or [0.0]
        centre = (max(ys) + min(ys)) / 2
        half_h = max((max(ys) - min(ys)) / 2 + _BH / 2, _BH / 2)
        return _Sub(
            [PlacedBox(g.label, 0.0, centre, half_h, created, consumed, hw)],
            [],
            in_eps,
            out_eps,
            -hw,
            hw,
        )

    n = max(len(ins), len(outs), 1)
    half_h = max((n - 1) * _PS / 2 + _BH / 2, _BH / 2)

    def col(ports, x):
        k = len(ports)
        return [(p, x, (i - (k - 1) / 2) * _PS) for i, p in enumerate(ports)]

    in_eps = col(ins, -hw - _BOX_PAD)
    out_eps = col(outs, hw + _BOX_PAD)
    return _Sub(
        [PlacedBox(g.label, 0.0, 0.0, half_h, created, consumed, hw)],
        [],
        in_eps,
        out_eps,
        -hw,
        hw,
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


def _optimize_ports(layout: Layout, iters: int = 16) -> Layout:
    """Per-box port-slot reassignment on a finished (compact) Layout: keep box
    positions and the fixed per-edge slot y-values, but permute WHICH wire
    attaches to WHICH slot so each box's ports emerge in the vertical order of
    the boxes they connect to -- cutting the port-order-mismatch crossings the
    term path leaves (``>>``/``@`` composition fixes port order by a static
    ``sorted``). Strictly more powerful than the global
    ``LayoutStyle.port_order_key`` (one order per shared port identity): a box's
    right-slot order and its neighbour's left-slot order are chosen
    independently. Iterated because both ends of a wire move; converges in a few
    passes. Geometry-only and structure-preserving -- only endpoint y within a
    box's own slot set changes. Straight internal wires only: boundary stubs and
    lane-routed (``waypoints``) long edges are left untouched (the compact term
    layout this targets has neither)."""
    boxes = layout.boxes
    wires = list(layout.wires)

    def edge_x(b, side):  # x of b's right (+1) / left (-1) port edge
        return b.x + side * (b.half_w + _BOX_PAD)

    def on_edge(x, y, ex, b):
        return abs(x - ex) < 1e-6 and b.y - b.half_h - 1e-6 <= y <= b.y + b.half_h + 1e-6

    r_idx: dict = {id(b): [] for b in boxes}  # straight wires on each box's right
    l_idx: dict = {id(b): [] for b in boxes}  # ... and left edge
    for i, w in enumerate(wires):
        if w.boundary or w.waypoints is not None:
            continue
        for b in boxes:
            if on_edge(w.x1, w.y1, edge_x(b, +1), b):
                r_idx[id(b)].append(i)
                break
        for b in boxes:
            if on_edge(w.x2, w.y2, edge_x(b, -1), b):
                l_idx[id(b)].append(i)
                break

    for _ in range(max(1, iters)):
        nw = list(wires)
        for b in boxes:
            ri = r_idx[id(b)]
            if len(ri) > 1:  # order this box's right ports by their consumers' y
                slots = sorted(wires[i].y1 for i in ri)
                for s, i in zip(slots, sorted(ri, key=lambda k: (wires[k].y2, k))):
                    nw[i] = replace(nw[i], y1=s)
            li = l_idx[id(b)]
            if len(li) > 1:  # order this box's left ports by their producers' y
                slots = sorted(wires[i].y2 for i in li)
                for s, i in zip(slots, sorted(li, key=lambda k: (wires[k].y1, k))):
                    nw[i] = replace(nw[i], y2=s)
        wires = nw
    return Layout(boxes, wires, layout.types)


def _greedy_switch(layout: Layout) -> Layout:
    """Greedy adjacent-slot switch (Eades-Wormald / ELK greedySwitch) on the FINAL
    (routed) layout: repeatedly apply the first straight-wire slot swap (on any box
    edge) that lowers the TOTAL geometric crossing count -- including crossings
    against the lane-routed arcs, which is why this runs after routing, not inside
    :func:`_optimize_ports` (which optimises the pre-route straight layout). Only
    straight internal legs move (their slot y); boundary stubs and ``waypoints``
    arcs are fixed obstacles. Deterministic (fixed edge+pair scan order),
    never-worse (only strict improvements taken)."""
    boxes = layout.boxes
    wires = list(layout.wires)
    r_idx: dict = {id(b): [] for b in boxes}
    l_idx: dict = {id(b): [] for b in boxes}
    for i, w in enumerate(wires):
        if w.boundary or w.waypoints is not None:
            continue
        for b in boxes:
            if abs(w.x1 - (b.x + b.half_w + _BOX_PAD)) < 1e-6 and \
                    b.y - b.half_h - 1e-6 <= w.y1 <= b.y + b.half_h + 1e-6:
                r_idx[id(b)].append(i)
                break
        for b in boxes:
            if abs(w.x2 - (b.x - b.half_w - _BOX_PAD)) < 1e-6 and \
                    b.y - b.half_h - 1e-6 <= w.y2 <= b.y + b.half_h + 1e-6:
                l_idx[id(b)].append(i)
                break
    edges = [(r_idx[id(b)], "y1") for b in boxes if len(r_idx[id(b)]) > 1]
    edges += [(l_idx[id(b)], "y2") for b in boxes if len(l_idx[id(b)]) > 1]
    if not edges:
        return layout
    cur = _count_crossings(layout)
    guard = 0
    changed = True
    while changed and cur > 0 and guard < 1000:
        changed = False
        guard += 1
        for idxs, attr in edges:
            order = sorted(idxs, key=lambda k: getattr(wires[k], attr))
            for a, c in zip(order, order[1:]):
                ya, yc = getattr(wires[a], attr), getattr(wires[c], attr)
                trial = list(wires)
                trial[a] = replace(trial[a], **{attr: yc})
                trial[c] = replace(trial[c], **{attr: ya})
                tc = _count_crossings(Layout(boxes, trial, layout.types))
                if tc < cur:
                    wires, cur, changed = trial, tc, True
                    break
            if changed:
                break
    return Layout(boxes, wires, layout.types)


def _route_long_edges(layout: Layout) -> Layout:
    """Post-pass on a finished (term-path) :class:`Layout`: any internal straight
    wire that would pass **behind** an intervening box is re-routed through a
    reserved horizontal lane in the gap above/below those boxes and redrawn as a
    ``waypoints`` polyline -- so no wire is ever drawn through/behind a box
    (readability problem (a)). This mirrors the DAG path's lane routing
    (:func:`_layout_composite`) but operates on already-placed geometry, so it
    composes *after* :func:`_optimize_ports` (slots already settled). Boundary
    stubs, wires that already carry waypoints, and wires that clear every box are
    left untouched; the endpoint slots are preserved (only the middle bends out).

    Side + lane level are chosen exactly as the DAG path does: pick the side
    (above/below) with the smaller excursion from the wire's two endpoints, then
    greedily pack non-overlapping column-spans onto the lowest free level so
    parallel long edges never share a lane."""
    boxes = layout.boxes
    if len(boxes) < 3:  # need at least one box strictly between two others
        return layout
    wires = list(layout.wires)

    def grazed(w) -> list:
        """Boxes in an intervening column whose vertical band the wire would
        actually pass behind. Surgical -- only edges that genuinely occlude a box
        (readability problem (a)) are lifted; a long edge that already clears
        every box (e.g. swoops below them) is left straight, so clean branchy
        diagrams are not needlessly arced. (After :func:`_straighten_boxes` puts a
        chain on one baseline, its long edges DO graze the aligned boxes, so
        chains still get flat lanes; unstraightened branchy edges usually don't.)"""
        lo_x, hi_x = min(w.x1, w.x2), max(w.x1, w.x2)
        span = w.x2 - w.x1
        if abs(span) < 1e-9:
            return []
        out = []
        for b in boxes:
            if not (lo_x < b.x < hi_x):  # must be an intervening column
                continue
            ys = [w.y1 + ((x - w.x1) / span) * (w.y2 - w.y1)
                  for x in (b.x - b.half_w, b.x + b.half_w)]
            if min(ys) <= b.y + b.half_h and max(ys) >= b.y - b.half_h:
                out.append(b)
        return out

    longs = []
    for i, w in enumerate(wires):
        if w.boundary or w.waypoints is not None:
            continue
        cb = grazed(w)
        if cb:
            longs.append((i, w, cb))
    if not longs:
        return layout

    # widest span first, for stable greedy lane packing
    longs.sort(key=lambda t: -(max(t[1].x1, t[1].x2) - min(t[1].x1, t[1].x2)))
    side_levels: dict[int, list[list[tuple[float, float]]]] = {+1: [], -1: []}
    for i, w, cb in longs:
        lo_x, hi_x = min(w.x1, w.x2), max(w.x1, w.x2)
        # clear EVERY box in the span (the arc spans the whole gap), not just the
        # grazed ones that triggered routing
        span_boxes = [b for b in boxes if lo_x < b.x < hi_x]
        top = max(b.y + b.half_h for b in span_boxes)
        bot = min(b.y - b.half_h for b in span_boxes)
        cost_up = abs(w.y1 - top) + abs(w.y2 - top)
        cost_dn = abs(w.y1 - bot) + abs(w.y2 - bot)
        side = +1 if cost_up <= cost_dn else -1
        levels = side_levels[side]
        lvl = next(
            (k for k, occ in enumerate(levels)
             if all(hi_x <= a or b <= lo_x for (a, b) in occ)),
            None,
        )
        if lvl is None:
            lvl = len(levels)
            levels.append([])
        levels[lvl].append((lo_x, hi_x))
        base = (top if side > 0 else bot)
        ly = base + side * (_LANE_BASE + lvl * _LANE_STEP)
        x1, y1, x2, y2 = w.x1, w.y1, w.x2, w.y2
        r = min(_RISER, (x2 - x1) / 4)
        wp = (
            (x1, y1),
            (x1 + r, y1),
            (x1 + 2 * r, ly),
            (x2 - 2 * r, ly),
            (x2 - r, y2),
            (x2, y2),
        )
        wires[i] = replace(w, waypoints=wp)
    return Layout(boxes, wires, layout.types)


def _straighten_boxes(layout: Layout, iters: int = 12, damping: float = 0.6) -> Layout:
    """Continuity pass on a finished (term-path) :class:`Layout`: relax each box's
    y toward the mean y of the boxes it is wired to (a barycentre sweep), so a
    ``;``-chain's spine comes out horizontal and its wires stop weaving up and
    down (readability problem (d)). Within-column box overlaps are re-separated
    after every sweep, and every attached wire endpoint (internal legs and
    boundary stubs alike) is shifted with its box, so connectivity and slot
    order are preserved -- only box centres move. **Port-aware**: the sweep targets
    each *wire* being level (its two port endpoints at equal y), not just box
    centres aligned, so wires come out straight wherever the boxes allow. Run
    AFTER :func:`_optimize_ports` (slots assigned) and BEFORE
    :func:`_route_long_edges`."""
    boxes = layout.boxes
    if len(boxes) < 3:
        return layout
    idx = {id(b): k for k, b in enumerate(boxes)}
    ys = [b.y for b in boxes]
    hh = [b.half_h for b in boxes]

    def box_at(x, y, side_sign):
        """Index of the box whose right (``+1``) / left (``-1``) edge holds
        ``(x,y)``, or ``None``."""
        for b in boxes:
            if abs(x - (b.x + side_sign * (b.half_w + _BOX_PAD))) < 1e-6 and (
                b.y - b.half_h - 1e-6 <= y <= b.y + b.half_h + 1e-6
            ):
                return idx[id(b)]
        return None

    # port-AWARE links: for each wire, the offset of its port from each box's
    # centre, so the sweep can target *wire* levelness (endpoints at equal y),
    # not just box-centre alignment -- straight wires wherever the boxes allow.
    links: list[list[tuple[int, float, float]]] = [[] for _ in boxes]
    for w in layout.wires:
        if w.boundary or w.waypoints is not None:
            continue
        u = box_at(w.x1, w.y1, +1)  # right edge == wire source
        v = box_at(w.x2, w.y2, -1)  # left edge == wire dest
        if u is not None and v is not None and u != v:
            du = w.y1 - boxes[u].y  # u's right-port offset from its centre
            dv = w.y2 - boxes[v].y  # v's left-port offset from its centre
            links[u].append((v, du, dv))
            links[v].append((u, dv, du))

    cols: dict[float, list[int]] = defaultdict(list)
    for k, b in enumerate(boxes):
        cols[round(b.x, 6)].append(k)

    def median(xs):
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    for _ in range(max(1, iters)):
        newys = list(ys)
        for k in range(len(boxes)):
            if links[k]:
                # target the MEDIAN wire being exactly level (Brandes-Kopf median
                # alignment) -> maximises the count of dead-straight wires, rather
                # than the mean, which just minimises total slope and leaves many
                # wires slightly bent.
                target = median([ys[nbr] + noff - moff
                                 for (nbr, moff, noff) in links[k]])
                newys[k] = (1 - damping) * ys[k] + damping * target
        ys = newys
        # re-separate boxes that now overlap within a column (top -> down)
        for ks in cols.values():
            order_k = sorted(ks, key=lambda k: ys[k], reverse=True)
            for a in range(1, len(order_k)):
                hi, lo = order_k[a - 1], order_k[a]
                gap = hh[hi] + hh[lo] + _VGAP
                if ys[hi] - ys[lo] < gap:
                    ys[lo] = ys[hi] - gap

    dy = {id(b): ys[idx[id(b)]] - b.y for b in boxes}

    def dy_at(x, y):
        for b in boxes:
            for ex in (b.x + b.half_w + _BOX_PAD, b.x - b.half_w - _BOX_PAD):
                if abs(x - ex) < 1e-6 and (
                    b.y - b.half_h - 1e-6 <= y <= b.y + b.half_h + 1e-6
                ):
                    return dy[id(b)]
        return 0.0

    newboxes = [replace(b, y=ys[idx[id(b)]]) for b in boxes]
    newwires: list[Wire] = []
    for w in layout.wires:
        if w.boundary:  # stub: whole leg translates with its box
            box_end = w.open_end != (w.x1, w.y1)  # True -> (x1,y1) is the box end
            d = dy_at(w.x1, w.y1) if box_end else dy_at(w.x2, w.y2)
            oe = None if w.open_end is None else (w.open_end[0], w.open_end[1] + d)
            newwires.append(replace(w, y1=w.y1 + d, y2=w.y2 + d, open_end=oe))
        elif w.waypoints is not None:  # none yet on the term path, but be safe
            newwires.append(w)
        else:
            newwires.append(replace(w, y1=w.y1 + dy_at(w.x1, w.y1),
                                    y2=w.y2 + dy_at(w.x2, w.y2)))
    return Layout(newboxes, newwires, layout.types)


def _count_crossings(layout: Layout) -> int:
    """Total proper segment-crossings among all wires (waypoint polylines
    included). Geometry-only; used to guard :func:`_straighten_boxes` (adopt it
    per-diagram only when it does not increase crossings)."""
    def segs_of(w):
        pts = list(w.waypoints) if w.waypoints else [(w.x1, w.y1), (w.x2, w.y2)]
        return list(zip(pts, pts[1:]))

    def orient(u, v, p):
        val = (v[0] - u[0]) * (p[1] - u[1]) - (v[1] - u[1]) * (p[0] - u[0])
        return (val > 1e-9) - (val < -1e-9)

    def cross(a, b):
        (p1, p2), (p3, p4) = a, b
        for e in (p1, p2):
            for f in (p3, p4):
                if abs(e[0] - f[0]) < 1e-6 and abs(e[1] - f[1]) < 1e-6:
                    return False
        d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
        d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
        return d1 != d2 and d3 != d4 and 0 not in (d1, d2, d3, d4)

    segs = [s for w in layout.wires for s in segs_of(w)]
    return sum(cross(segs[i], segs[j])
               for i in range(len(segs)) for j in range(i + 1, len(segs)))


def _straight_count(layout: Layout) -> int:
    """How many internal legs are drawn dead level (both port endpoints at equal
    y, so the wire renders horizontal). The straightness objective, used to
    tie-break equal-crossing candidates toward the flatter drawing."""
    return sum(1 for w in layout.wires
               if not w.boundary and w.waypoints is None and abs(w.y1 - w.y2) < 1e-6)


def _geom_key(layout: Layout):
    """Rounded box-y signature -- equal across two layouts iff the boxes stopped
    moving, i.e. the simplification loop has reached a fixed point."""
    return tuple(round(b.y, 5) for b in layout.boxes)


def lower_term(sub: _Sub, style: LayoutStyle) -> Layout:
    """Full term-path lowering with the readability stack applied per
    :class:`LayoutStyle`: ``_finish`` -> ``_optimize_ports`` -> **iterated**
    (port-aware ``_straighten_boxes`` -> ``_optimize_ports``) up to
    ``simplify_rounds`` times -> ``_route_long_edges``. The iteration is the point:
    straightening moves boxes, which lets the port re-assignment remove crossings
    it could not see before, which shifts the geometry so the next straighten
    levels more wires -- the passes compound until a fixed point. Every
    intermediate (and the un-straightened base) is a candidate; the winner
    minimises crossings, tie-broken toward the most dead-straight wires, so a
    ``;``-chain flattens while an already-clean branchy diagram is left untouched.
    Fully deterministic (sorted iteration only)."""
    base = _finish(sub, style)
    if style.crossing_min:
        base = _optimize_ports(base, style.crossing_min_iters)

    def tail(lay: Layout) -> Layout:
        routed = _route_long_edges(lay) if style.route_long_edges else lay
        return _greedy_switch(routed)  # final crossing-min vs the routed arcs

    def better(a: Layout, b: Layout) -> bool:  # a strictly preferred to b
        return (_count_crossings(a), -_straight_count(a)) < \
               (_count_crossings(b), -_straight_count(b))

    best = tail(base)
    if style.straighten:
        cur = base
        seen = {_geom_key(cur)}
        for _ in range(max(1, style.simplify_rounds)):
            s = _straighten_boxes(cur, style.straighten_iters)
            if style.crossing_min:  # re-slot against the just-straightened geometry
                s = _optimize_ports(s, style.crossing_min_iters)
            routed = tail(s)
            if better(routed, best):
                best = routed
            key = _geom_key(s)
            if key in seen:  # fixed point (or a 2-cycle) -> stop
                break
            seen.add(key)
            cur = s
    return best
