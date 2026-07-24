"""Regression: the crossing-minimising port reassignment on the term/compact
layout (``_optimize_ports``, gated by ``LayoutStyle(crossing_min=True)``).

The ``>>``/``@`` term path fixes each box's port order by a static ``sorted``,
leaving port-order-mismatch crossings at box boundaries. ``_optimize_ports``
keeps box positions and each edge's fixed slot y-values but permutes which wire
attaches to which slot, so a box's ports emerge in its neighbours' vertical
order. Pins: it preserves connectivity + slot sets, is deterministic, reaches a
zero-crossing drawing on a solvable twist, and is never worse. The quantitative
win on the order-management OCCN classes (term 1566 -> 570 crossings, 115/122
improved) is measured by the audit-repo probe scratchpad/untangle-probe/probe_portopt.py.
"""
from __future__ import annotations

from procposets.cospan.signature import Generator, Port
from procposets.viz._layout import (
    _BH, _BOX_PAD, _BW, _PS, Diagram, Layout, LayoutStyle, PlacedBox, Wire,
    _box_sub, _finish, _optimize_ports,
)

PLAIN = LayoutStyle()


def _P(s, t, g):
    return Port(s, t, g)


def _box(g):
    return Diagram(lambda st: _box_sub(g, st))


def _term_layout(diagram, style=PLAIN):
    return _finish(diagram._sub(style), style)


def _seg(w):
    pts = list(w.waypoints) if w.waypoints else [(w.x1, w.y1), (w.x2, w.y2)]
    return list(zip(pts, pts[1:]))


def _proper_cross(a, b):
    (p1, p2), (p3, p4) = a, b
    def o(u, v, w):
        val = (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
        return (val > 1e-9) - (val < -1e-9)
    for e in (p1, p2):
        for f in (p3, p4):
            if abs(e[0] - f[0]) < 1e-6 and abs(e[1] - f[1]) < 1e-6:
                return False
    d1, d2, d3, d4 = o(p3, p4, p1), o(p3, p4, p2), o(p1, p2, p3), o(p1, p2, p4)
    return d1 != d2 and d3 != d4 and 0 not in (d1, d2, d3, d4)


def _crossings(layout):
    segs = [s for w in layout.wires for s in _seg(w)]
    return sum(_proper_cross(segs[i], segs[j])
               for i in range(len(segs)) for j in range(i + 1, len(segs)))


def _diamond():
    """G fans to A, B; C joins them. The ``@`` stacks B above A while C's sorted
    left ports put the A-leg on top, so the term path draws the A/B->C legs
    crossed -- a port-slot swap on C (which has 2 left ports) removes it."""
    G = Generator("G", frozenset(), frozenset({_P("G", "t", "A"), _P("G", "t", "B")}))
    A = Generator("A", frozenset({_P("G", "t", "A")}), frozenset({_P("A", "t", "C")}))
    B = Generator("B", frozenset({_P("G", "t", "B")}), frozenset({_P("B", "t", "C")}))
    C = Generator("C", frozenset({_P("A", "t", "C"), _P("B", "t", "C")}), frozenset())
    return _box(G) >> (_box(B) @ _box(A)) >> _box(C)


def _edge_slots(layout):
    """Per box edge (id, side), the sorted set of port-slot y-values -- the
    invariant _optimize_ports may permute assignments within but not change."""
    from collections import defaultdict
    from procposets.viz._layout import _BW, _BOX_PAD
    off = _BW / 2 + _BOX_PAD
    slots = defaultdict(list)
    for w in layout.wires:
        if w.boundary or w.waypoints is not None:
            continue
        for b in layout.boxes:
            if abs(w.x1 - (b.x + off)) < 1e-6 and abs(w.y1 - b.y) <= b.half_h + 1e-6:
                slots[(id(b), "r")].append(round(w.y1, 6))
            if abs(w.x2 - (b.x - off)) < 1e-6 and abs(w.y2 - b.y) <= b.half_h + 1e-6:
                slots[(id(b), "l")].append(round(w.y2, 6))
    return {k: sorted(v) for k, v in slots.items()}


def _crossed_layout():
    """Hand-built Layout with a guaranteed port-slot crossing: producers P (top)
    and Q (bottom) wire into consumer C's two left slots INVERTED (P->lower,
    Q->upper). C is the only box with >1 port on an edge, so _optimize_ports must
    swap its two left slots to untangle."""
    off = _BW / 2 + _BOX_PAD
    hh1, hh2 = _BH / 2, _PS / 2 + _BH / 2
    P = PlacedBox("P", 0.0, 0.6, hh1)
    Q = PlacedBox("Q", 0.0, -0.6, hh1)
    C = PlacedBox("C", 3.0, 0.0, hh2)
    pP, pQ = Port("P", "t", "C"), Port("Q", "t", "C")
    w1 = Wire(off, 0.6, 3.0 - off, -0.3, "t", pP)   # P(top) -> C lower slot
    w2 = Wire(off, -0.6, 3.0 - off, 0.3, "t", pQ)   # Q(bottom) -> C upper slot
    return Layout([P, Q, C], [w1, w2], {"t"})


def test_optimize_ports_removes_crossing():
    crossed = _crossed_layout()
    assert _crossings(crossed) == 1
    assert _crossings(_optimize_ports(crossed)) == 0


def test_diamond_zero_crossing():
    assert _crossings(_optimize_ports(_term_layout(_diamond()))) == 0


def test_render_dispatch_applies_optimizer():
    import matplotlib
    matplotlib.use("Agg")
    from procposets.viz.string_diagram import StringDiagramStyle, render
    d = _diamond()
    plain = render(d)
    tuned = render(d, style=StringDiagramStyle(layout=LayoutStyle(crossing_min=True)))
    import matplotlib.pyplot as plt
    plt.close(plain)
    plt.close(tuned)


def test_slot_sets_and_box_positions_preserved():
    for d in (_diamond(),):
        plain = _term_layout(d)
        opt = _optimize_ports(plain)
        assert [(b.label, b.x, b.y) for b in plain.boxes] == \
               [(b.label, b.x, b.y) for b in opt.boxes]
        assert _edge_slots(plain) == _edge_slots(opt)  # same slots, only reassigned
        assert len(opt.wires) == len(plain.wires)


def test_never_worse():
    for d in (_diamond(),):
        plain = _term_layout(d)
        assert _crossings(_optimize_ports(plain)) <= _crossings(plain)


def test_deterministic():
    plain = _term_layout(_diamond())
    a, b = _optimize_ports(plain), _optimize_ports(plain)
    assert [(w.x1, w.y1, w.x2, w.y2) for w in a.wires] == \
           [(w.x1, w.y1, w.x2, w.y2) for w in b.wires]
