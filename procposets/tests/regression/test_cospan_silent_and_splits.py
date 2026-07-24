"""Regression: faithful silent handling is the DEFAULT (tau AND-splits/joins).

A silent (tau) transition with >1 output place (AND-split) or >1 input place
(AND-join) *synchronises* its branches. The faithful default keeps the silent
transparent, so the branches stay AND-combined into ONE generator; the lossy
``remove_silent=True`` opt-in splices them independently through the neighbouring
XOR places and invents spurious single-branch choices (an AND degrading to XOR).

Pins the 2026-07-23 silent-and-splits fix (default flipped True->False). See
``LMGraph.without_silent`` and ``engine.extract_signature``. Complements
``test_cospan_skeleton.test_remove_silent_matches_slow`` (which pins that the two
modes differ and each matches the slow engine) by naming *which* mode is faithful.
"""
from __future__ import annotations

from procposets.cospan.engine import extract_signature
from procposets.cospan.engine_fast import extract_signature_fast
from procposets.cospan.from_ocpn import signature_from_ocpn
from procposets.cospan.lmgraph import Kind, LMGraph
from procposets.cospan.signature_compare import canonical_generators


def _tau_and_split() -> LMGraph:
    """``A`` fans through a silent 1-in/2-out tau into ``B`` and ``C``:
    faithful => A fires to {B, C} jointly; contracted => A chooses B XOR C."""
    g = LMGraph()
    for a in ("A", "B", "C"):
        g.add_activity(a)
    for m in ("pin", "p1", "p2"):
        g.add_mediator(m, Kind.XOR)
    g.add_mediator("tau", Kind.SEQ, silent=True)
    g.add_edge("A", "pin", "t"); g.add_edge("pin", "tau", "t")
    g.add_edge("tau", "p1", "t"); g.add_edge("p1", "B", "t")
    g.add_edge("tau", "p2", "t"); g.add_edge("p2", "C", "t")
    g.validate()
    return g


def _tau_and_join() -> LMGraph:
    """``B`` and ``C`` join through a silent 2-in/1-out tau into ``A``:
    faithful => A is fed by {B, C} jointly; contracted => A fed by B XOR C."""
    g = LMGraph()
    for a in ("A", "B", "C"):
        g.add_activity(a)
    for m in ("p1", "p2", "pout"):
        g.add_mediator(m, Kind.XOR)
    g.add_mediator("tau", Kind.SEQ, silent=True)
    g.add_edge("B", "p1", "t"); g.add_edge("p1", "tau", "t")
    g.add_edge("C", "p2", "t"); g.add_edge("p2", "tau", "t")
    g.add_edge("tau", "pout", "t"); g.add_edge("pout", "A", "t")
    g.validate()
    return g


def _right_tgt_sets(sig, lab):
    return sorted(tuple(sorted(p.tgt for p in g.right)) for g in sig.by_label(lab))


def _left_src_sets(sig, lab):
    return sorted(tuple(sorted(p.src for p in g.left)) for g in sig.by_label(lab))


def test_and_split_synchronises_by_default():
    sig = extract_signature(_tau_and_split())            # default remove_silent=False
    assert _right_tgt_sets(sig, "A") == [("B", "C")]     # ONE generator, both branches


def test_and_split_remove_silent_invents_choice():
    sig = extract_signature(_tau_and_split(), remove_silent=True)
    assert _right_tgt_sets(sig, "A") == [("B",), ("C",)]  # the DEFECT, now opt-in only


def test_and_join_synchronises_by_default():
    sig = extract_signature(_tau_and_join())
    assert _left_src_sets(sig, "A") == [("B", "C")]


def test_and_join_remove_silent_invents_choice():
    sig = extract_signature(_tau_and_join(), remove_silent=True)
    assert _left_src_sets(sig, "A") == [("B",), ("C",)]


def test_default_is_faithful_not_contracted():
    for factory in (_tau_and_split, _tau_and_join):
        g = factory()
        assert extract_signature(g) == extract_signature(g, remove_silent=False)
        assert extract_signature(g) != extract_signature(g, remove_silent=True)


def test_fast_matches_slow_under_faithful_default():
    for factory in (_tau_and_split, _tau_and_join):
        g = factory()
        slow = set(canonical_generators(extract_signature(g)).keys())
        fast = set(canonical_generators(extract_signature_fast(g)).keys())
        assert fast == slow, f"{factory.__name__}: fast={fast} slow={slow}"


def _fake_ocpn_and_split() -> dict:
    """A single-type discovered-style OCPN carrying a tau AND-split: activity
    ``A`` -> tau -> two output places -> ``B``, ``C`` (one typed net)."""
    class _P:
        def __init__(s, name): s.name = name
    class _T:
        def __init__(s, name, label): s.name, s.label = name, label
    class _Arc:
        def __init__(s, src, tgt): s.source, s.target = src, tgt
    class _Net:
        def __init__(s, places, transitions, arcs):
            s.places, s.transitions, s.arcs = places, transitions, arcs
    pin, p1, p2 = _P("pin"), _P("p1"), _P("p2")
    a, tau, b, c = _T("A", "A"), _T("tau", None), _T("B", "B"), _T("C", "C")
    net = _Net({pin, p1, p2}, {a, tau, b, c},
               [_Arc(a, pin), _Arc(pin, tau), _Arc(tau, p1), _Arc(tau, p2),
                _Arc(p1, b), _Arc(p2, c)])
    return {"petri_nets": {"t": (net, None, None)}}


def test_signature_from_ocpn_default_keeps_synchronisation():
    """End-to-end through the OCPN wrapper: the AND-split survives by default on
    both the full and canonical extractors; the lossy opt-in differs."""
    ocpn = _fake_ocpn_and_split()
    faithful = signature_from_ocpn(ocpn, canonical=False)
    assert _right_tgt_sets(faithful, "A") == [("B", "C")]
    assert faithful == signature_from_ocpn(ocpn, canonical=False, remove_silent=False)
    assert faithful != signature_from_ocpn(ocpn, canonical=False, remove_silent=True)
    # canonical (fast) default is likewise faithful and byte-equal to slow keys
    fast_keys = set(canonical_generators(signature_from_ocpn(ocpn, canonical=True)).keys())
    slow_keys = set(canonical_generators(faithful).keys())
    assert fast_keys == slow_keys
