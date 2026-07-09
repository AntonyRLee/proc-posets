"""Tests for trace-language generation from the splice representation
(CLASS_EXTRACTION.md §27c): bounded by a loop cut-off, exact for series-parallel
families."""

from __future__ import annotations

from collections import Counter

from procposets.cospan.class_extraction import ExtractionResult, NamedMorphism, _to_key
from procposets.cospan.signature import Generator, Port
from procposets.cospan.splice import SpliceRepresentation
from procposets.cospan.trace_language import model_traces


def _gen(label, left, right):
    return Generator(label=label, left=frozenset(left), right=frozenset(right))


def _nm(name, body, boundary=frozenset()):
    return NamedMorphism(name=name, boundary=boundary, body=tuple(body))


def P(src, tgt, typ="t"):
    return Port(src=src, typ=typ, tgt=tgt)


def _rep():
    """Baseline gamma1;a;(x@y);z + a loop p;q anchored at the post-a frontier."""
    g1 = _gen("gamma1", [], [P("g1", "a")])
    a = _gen("a", [P("g1", "a")], [P("a", "x"), P("a", "y")])
    x = _gen("x", [P("a", "x")], [P("x", "z")])
    y = _gen("y", [P("a", "y")], [P("y", "z")])
    z = _gen("z", [P("x", "z"), P("y", "z")], [])
    baseline = _nm("M1", [g1, a, frozenset({x, y}), z])
    p = _gen("p", [P("a", "x"), P("a", "y")], [P("p", "q")])
    q = _gen("q", [P("p", "q")], [P("a", "x"), P("a", "y")])
    loop = _nm("L1", [p, q], boundary=_to_key(Counter({P("a", "x"): 1, P("a", "y"): 1})))
    res = ExtractionResult(
        fragments={f.name: f for f in (baseline, loop)}, valid_generators=set(), frontiers_visited=0
    )
    return SpliceRepresentation.from_extraction_result(res, name="m")


def test_baseline_traces_capture_concurrency():
    """m=0: the two linear extensions of the x@y concurrent pair."""
    tl = model_traces(_rep(), max_loops=0)
    assert tl.traces["σ1"] == {
        ("gamma1", "a", "x", "y", "z"),
        ("gamma1", "a", "y", "x", "z"),
    }
    assert not tl.approx_families  # series-parallel -> exact


def test_loop_unrolling_extends_the_language_with_the_cutoff():
    rep = _rep()
    n0 = len(model_traces(rep, max_loops=0).all_traces())
    n1 = len(model_traces(rep, max_loops=1).all_traces())
    n2 = len(model_traces(rep, max_loops=2).all_traces())
    assert n0 == 2 and n1 > n0 and n2 > n1  # bounded growth with the cut-off
    # an m=1 trace splices the loop body p;q after gamma1;a
    one = model_traces(rep, max_loops=1).all_traces()
    assert ("gamma1", "a", "p", "q", "x", "y", "z") in one


def test_traces_are_deterministic():
    rep = _rep()
    assert model_traces(rep, max_loops=2).all_traces() == model_traces(rep, max_loops=2).all_traces()
