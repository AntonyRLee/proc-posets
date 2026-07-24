"""Regression: the term-path readability stack layered on top of the port
optimizer -- long-edge lane routing (:func:`_route_long_edges`), continuity
straightening (:func:`_straighten_boxes`), the crossing-guarded lowering
(:func:`lower_term`), and the geometry-only crossing counter
(:func:`_count_crossings`).

Pins the behaviour the localisation class figures rely on:
  * a wire that would pass *behind* an intervening box is lifted into a lane
    (gets ``waypoints``) so no straight internal leg crosses a box body;
  * straightening collapses a ``;``-chain's boxes toward a common baseline;
  * the guard never adopts straightening when it would raise the crossing count
    (branchy diagrams stay as clean as the composition made them);
  * the three crossing render styles all dispatch.
"""
from __future__ import annotations

from procposets.cospan.signature import Generator, Port
from procposets.viz._layout import (
    _BH, _BOX_PAD, _BW, Diagram, Layout, LayoutStyle, PlacedBox, Wire, _box_sub,
    _count_crossings, _finish, _optimize_ports, _route_long_edges,
    _straighten_boxes, lower_term,
)


def _P(s, t, g):
    return Port(s, t, g)


def _box(g):
    return Diagram(lambda st: _box_sub(g, st))


def _sub(diagram, style):
    return diagram._sub(style)


# A >> B >> C where A produces a port consumed only by C (span 2, over B); the
# spine port A->B->C stays adjacent.
_A = Generator("A", frozenset(), frozenset({_P("A", "l", "C"), _P("A", "o", "B")}))
_B = Generator("B", frozenset({_P("A", "o", "B")}), frozenset({_P("B", "o", "C")}))
_C = Generator("C", frozenset({_P("A", "l", "C"), _P("B", "o", "C")}), frozenset())
_CHAIN = _box(_A) >> _box(_B) >> _box(_C)


def _behind_any_box(layout):
    """True if any straight internal leg's midpoint lands inside a box body."""
    for w in layout.wires:
        if w.boundary or w.waypoints is not None:
            continue
        mx, my = (w.x1 + w.x2) / 2, (w.y1 + w.y2) / 2
        for b in layout.boxes:
            if (b.x - b.half_w < mx < b.x + b.half_w
                    and b.y - b.half_h < my < b.y + b.half_h):
                return True
    return False


def test_no_leg_runs_behind_a_box():
    # the readability guarantee: after the full pipeline, no straight internal
    # leg passes behind a box body (whether that is achieved by routing it into a
    # lane OR by straightening it clear of the boxes).
    style = LayoutStyle(straighten=True, crossing_min=True)
    layout = lower_term(_sub(_CHAIN, style), style)
    assert not _behind_any_box(layout)


def test_route_lifts_a_grazing_edge_into_a_lane():
    # a hand-built layout with a long wire that genuinely passes THROUGH an
    # intervening box (all three boxes on one baseline) -- routing must lift it.
    hh, off = _BH / 2, _BW / 2 + _BOX_PAD
    A = PlacedBox("A", 0.0, 0.0, hh)
    B = PlacedBox("B", 3.0, 0.0, hh)  # intervening, same baseline
    C = PlacedBox("C", 6.0, 0.0, hh)
    w = Wire(0.0 + off, 0.0, 6.0 - off, 0.0, "t", _P("A", "t", "C"))  # straight through B
    before = Layout([A, B, C], [w], {"t"})
    assert _behind_any_box(before)  # precondition: it does occlude B
    routed = _route_long_edges(before)
    assert any(x.waypoints is not None for x in routed.wires), "must be lane-routed"
    assert not _behind_any_box(routed)


def test_route_is_noop_without_grazing():
    # two adjacent-column edges only (no span-2) -> nothing to route
    a = Generator("a", frozenset(), frozenset({_P("a", "o", "b")}))
    b = Generator("b", frozenset({_P("a", "o", "b")}), frozenset({_P("b", "o", "c")}))
    c = Generator("c", frozenset({_P("b", "o", "c")}), frozenset())
    style = LayoutStyle()
    layout = _finish(_sub(_box(a) >> _box(b) >> _box(c), style), style)
    routed = _route_long_edges(layout)
    assert all(w.waypoints is None for w in routed.wires)


def test_straighten_flattens_a_chain():
    style = LayoutStyle()
    base = _finish(_sub(_CHAIN, style), style)
    spread = lambda lay: (max(b.y for b in lay.boxes) - min(b.y for b in lay.boxes))
    straight = _straighten_boxes(base)
    assert spread(straight) <= spread(base) + 1e-9
    # connectivity + box count preserved
    assert len(straight.boxes) == len(base.boxes)
    assert len(straight.wires) == len(base.wires)


def test_straighten_preserves_box_x_and_labels():
    style = LayoutStyle()
    base = _finish(_sub(_CHAIN, style), style)
    straight = _straighten_boxes(base)
    assert [(b.label, round(b.x, 6)) for b in base.boxes] == \
           [(b.label, round(b.x, 6)) for b in straight.boxes]


def test_guard_never_worse_than_unstraightened():
    style_on = LayoutStyle(straighten=True, crossing_min=True)
    style_off = LayoutStyle(straighten=False, crossing_min=True)
    on = lower_term(_sub(_CHAIN, style_on), style_on)
    off = lower_term(_sub(_CHAIN, style_off), style_off)
    assert _count_crossings(on) <= _count_crossings(off)


def test_count_crossings_matches_optimizer():
    # the diamond has a removable port-slot crossing; optimizing kills it
    G = Generator("G", frozenset(), frozenset({_P("G", "t", "A"), _P("G", "t", "B")}))
    A = Generator("A", frozenset({_P("G", "t", "A")}), frozenset({_P("A", "t", "C")}))
    B = Generator("B", frozenset({_P("G", "t", "B")}), frozenset({_P("B", "t", "C")}))
    C = Generator("C", frozenset({_P("A", "t", "C"), _P("B", "t", "C")}), frozenset())
    d = _box(G) >> (_box(B) @ _box(A)) >> _box(C)
    style = LayoutStyle()
    base = _finish(_sub(d, style), style)
    assert _count_crossings(_optimize_ports(base)) == 0


def test_render_dispatch_all_crossing_styles():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from procposets.viz.string_diagram import (
        DrawStyle, StringDiagramStyle, render,
    )
    lay = LayoutStyle(straighten=True, crossing_min=True)
    for cs in ("plain", "casing", "gap"):
        fig = render(_CHAIN, style=StringDiagramStyle(layout=lay,
                                                      draw=DrawStyle(crossing_style=cs)))
        plt.close(fig)


def test_lower_term_deterministic():
    style = LayoutStyle(straighten=True, crossing_min=True)
    a = lower_term(_sub(_CHAIN, style), style)
    b = lower_term(_sub(_CHAIN, style), style)
    key = lambda lay: [(w.x1, w.y1, w.x2, w.y2, w.waypoints) for w in lay.wires]
    assert key(a) == key(b)


def test_abbreviate_scheme():
    from procposets.viz.string_diagram import abbreviate
    m = abbreviate({"place order", "confirm order", "create package",
                    "send package", "pick item", "item out of stock", "γ₁", "γ₂"})
    assert m["place order"] == "PLOR"       # two words -> first 2 of each
    assert m["confirm order"] == "COOR"
    assert m["create package"] == "CRPA"
    assert m["send package"] == "SEPA"
    assert m["item out of stock"] == "IOOS"  # 4 words -> initials
    assert m["γ₁"] == "γ₁" and m["γ₂"] == "γ₂"  # no ascii-alnum -> pass through
    assert len(set(m.values())) == len(m)   # unique codes


def test_abbreviate_resolves_collisions():
    from procposets.viz.string_diagram import abbreviate
    # two labels whose naive code clashes must still get distinct codes
    m = abbreviate({"pay order", "pay orders"})
    assert m["pay order"] != m["pay orders"]


def test_box_width_narrows_boxes():
    wide = LayoutStyle()                         # box_width 1.4 default
    narrow = LayoutStyle(box_width=0.8)
    w_lay = _finish(_sub(_CHAIN, wide), wide)
    n_lay = _finish(_sub(_CHAIN, narrow), narrow)
    assert all(abs(b.half_w - 0.7) < 1e-9 for b in w_lay.boxes)
    assert all(abs(b.half_w - 0.4) < 1e-9 for b in n_lay.boxes)
    # ports sit on the (narrower) box edge; full pipeline still runs clean
    narrow_full = LayoutStyle(box_width=0.8, straighten=True, crossing_min=True)
    lay = lower_term(_sub(_CHAIN, narrow_full), narrow_full)
    assert not _behind_any_box(lay)
