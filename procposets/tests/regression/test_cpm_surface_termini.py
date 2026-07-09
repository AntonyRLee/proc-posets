"""``engine.surface_termini`` + ``discovery_cleanup.close_gamma2_termini``: the
carrier-drop fix.

A discovered OCPN's ``s`` sends its carrier (container/box) to a bare *final* place; the
order continues to ``r``. Before the fix the engine's AND-combine silently absorbed the
terminating carrier (it contributed ``frozenset()``), so the carrier never reached
``gamma2``. ``surface_termini`` keeps it as a ``(…, gamma2)`` right leg -- but only for a
*mixed* activity (a pure terminus stays zero-right, the explicit-``G2`` running-example
convention), and only when the model has no explicit terminus boundary activity already.
"""
from procposets.cospan.discovery_cleanup import close_gamma2_termini
from procposets.cospan.engine import extract_signature
from procposets.cospan.lmgraph import Kind, LMGraph
from procposets.cospan.signature import Generator, Port, Signature


def _mixed_graph(terminus_label: str | None = None):
    """``X`` consumes order+box, sends order on to ``Y`` (continues) and box to a bare sink
    (terminates) -- the Liss ``s`` shape. ``terminus_label`` optionally adds an explicit
    terminus activity reachable from ``Y`` (to exercise the guard)."""
    g = LMGraph()
    for a in ("X", "Y"):
        g.add_activity(a)
    for p in ("p_oin", "p_bin", "p_xy", "p_bsink"):
        g.add_mediator(p, Kind.XOR)
    g.add_edge("p_oin", "X", "order")   # bare source -> X (order)
    g.add_edge("p_bin", "X", "box")     # bare source -> X (box)
    g.add_edge("X", "p_xy", "order")    # X -> Y (order continues)
    g.add_edge("p_xy", "Y", "order")
    g.add_edge("X", "p_bsink", "box")   # X -> bare sink (box terminates)
    if terminus_label is not None:
        g.add_activity(terminus_label)
        g.add_mediator("p_yend", Kind.XOR)
        g.add_edge("Y", "p_yend", "order")
        g.add_edge("p_yend", terminus_label, "order")
    g.validate()
    return g


def _x_right_types(sig: Signature) -> set:
    (x,) = [g for g in sig.generators if g.label == "X"]
    return {p.typ for p in x.right}


def _x_right_targets(sig: Signature) -> set:
    (x,) = [g for g in sig.generators if g.label == "X"]
    return {(p.typ, p.tgt) for p in x.right}


def test_default_absorbs_terminus():
    """Default (``surface_termini=False``): the terminating box is absorbed, X keeps only the
    continuing order -- the prior behaviour, on which the per-type PN/PT agreement rests."""
    sig = extract_signature(_mixed_graph(), surface_termini=False)
    assert _x_right_types(sig) == {"order"}


def test_surface_keeps_terminating_carrier():
    """``surface_termini=True``: the box terminus survives as a ``(X, box, gamma2)`` leg,
    beside the continuing ``(X, order, Y)`` -- the carrier reaches the final marking."""
    sig = extract_signature(_mixed_graph(), surface_termini=True)
    assert _x_right_types(sig) == {"order", "box"}
    assert ("box", "gamma2") in _x_right_targets(sig)
    assert ("order", "Y") in _x_right_targets(sig)


def test_pure_terminus_stays_zero_right_when_surfaced():
    """A *pure* terminus (Y here ends the order at no further activity) is collapsed to
    zero-right even with surfacing -- it *is* a boundary generator, not a leg into one."""
    sig = extract_signature(_mixed_graph(), surface_termini=True)
    (y,) = [g for g in sig.generators if g.label == "Y"]
    assert y.right == frozenset()  # zero-right, no synthesised gamma2 leg on Y itself


def test_explicit_terminus_activity_disables_surfacing():
    """If the model already carries an explicit terminus *activity* (``gamma2``/``END_<ot>``
    -- e.g. a master spec simulated into a log), surfacing is suppressed so it is not
    double-counted (and the ``gamma2`` label does not collide)."""
    sig = extract_signature(_mixed_graph(terminus_label="gamma2"), surface_termini=True)
    assert _x_right_types(sig) == {"order"}  # box terminus absorbed, not surfaced


def test_close_gamma2_termini_adds_drain_for_unconsumed_leg():
    """An unconsumed ``(…, gamma2)`` terminus leg gets one zero-right ``gamma2`` drain (so a
    closing frontier can empty); a leg already consumed gets none."""
    leg = Port("X", "box", "gamma2")
    producer = Generator("X", frozenset(), frozenset({leg}))
    drained = close_gamma2_termini(Signature(frozenset({producer})))
    g2 = [g for g in drained.generators if g.label == "gamma2"]
    assert len(g2) == 1 and g2[0].left == frozenset({leg}) and g2[0].right == frozenset()

    # already-consumed: a gamma2 generator already takes the leg -> no new drain
    consumer = Generator("gamma2", frozenset({leg}), frozenset())
    already = Signature(frozenset({producer, consumer}))
    assert close_gamma2_termini(already) == already
