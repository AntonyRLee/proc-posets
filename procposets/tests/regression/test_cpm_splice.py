"""Tests for the splice representation (CLASS_EXTRACTION.md §27): the
``M(m,σ)`` family catalogue as concrete pomsets + algebraic skeletons, canonical
and serializable."""

from __future__ import annotations

import json
from collections import Counter

from procposets.cospan.class_extraction import ExtractionResult, NamedMorphism, _to_key
from procposets.cospan.signature import Generator, Port
from procposets.cospan.splice import AlgebraicTerm, SpliceRepresentation


def _gen(label, left, right):
    return Generator(label=label, left=frozenset(left), right=frozenset(right))


def _nm(name, body, boundary=frozenset()):
    return NamedMorphism(name=name, boundary=boundary, body=tuple(body))


def P(src, tgt, typ="t"):
    return Port(src=src, typ=typ, tgt=tgt)


def _result(frags):
    return ExtractionResult(
        fragments={f.name: f for f in frags}, valid_generators=set(), frontiers_visited=0
    )


def _model():
    """A baseline gamma1;a;(x@y);z, a loop L1 anchored at the post-a frontier,
    and the m=2 unrolling that lists L1 explicitly."""
    g1 = _gen("gamma1", [], [P("g1", "a", "t")])
    a = _gen("a", [P("g1", "a", "t")], [P("a", "x", "t"), P("a", "y", "t")])
    x = _gen("x", [P("a", "x", "t")], [P("x", "z", "t")])
    y = _gen("y", [P("a", "y", "t")], [P("y", "z", "t")])
    z = _gen("z", [P("x", "z", "t"), P("y", "z", "t")], [])
    baseline = _nm("M1", [g1, a, frozenset({x, y}), z])
    # loop p;q at the post-a frontier {P(a,x), P(a,y)}
    p = _gen("p", [P("a", "x", "t"), P("a", "y", "t")], [P("p", "q", "t")])
    q = _gen("q", [P("p", "q", "t")], [P("a", "x", "t"), P("a", "y", "t")])
    anchor = _to_key(Counter({P("a", "x", "t"): 1, P("a", "y", "t"): 1}))
    loop = _nm("L1", [p, q], boundary=anchor)
    unrolled = _nm("M2", [g1, a, p, q, frozenset({x, y}), z])
    return baseline, loop, unrolled


def test_splice_rep_builds_one_family_with_a_splice_and_one_loop():
    baseline, loop, unrolled = _model()
    rep = SpliceRepresentation.from_extraction_result(
        _result([baseline, unrolled, loop]), name="m"
    )
    assert len(rep.families) == 1 and len(rep.loops) == 1
    fam = rep.families[0]
    assert fam.spine_id == "σ1" and rep.loops[0].loop_id == "ℓ1"
    # algebraic spine is the boundary-stripped, loop-free skeleton
    assert fam.term.steps == ("gamma1", "a", ("x", "y"), "z")
    # the loop splices after gamma1;a (index 2), referencing the one loop structure
    assert fam.splices == tuple(fam.splices)  # frozen/tuple
    assert [(s.site, s.loop_ids) for s in fam.splices] == [(2, ("ℓ1",))]
    assert fam.loop_free and fam.sp_exact


def test_splice_rep_is_canonically_serializable():
    baseline, loop, unrolled = _model()
    rep = SpliceRepresentation.from_extraction_result(
        _result([baseline, unrolled, loop]), name="m"
    )
    d = rep.to_dict()
    # round-trips through JSON unchanged (byte-stable, all-primitive)
    assert json.loads(json.dumps(d, sort_keys=True)) == d
    assert d["name"] == "m" and d["quotient"] == "forget_provenance"
    fam = d["families"][0]
    assert fam["term"] == ["gamma1", "a", ["x", "y"], "z"]
    assert fam["splices"] == [{"site": 2, "loop_ids": ["ℓ1"]}]
    # concrete companion present: 6 events (gamma1,a,x,y,z,gamma2) -- every closing pomset
    # carries the single γ1/γ2 boundary (§40), so z's open sink is closed at gamma2
    assert len(fam["pomset"]["events"]) == 6


def test_neutral_baseline_exposes_splice_without_an_unrolled_sibling():
    """The master case: a loop fragment + a loop-free baseline whose port
    frontier reaches the anchor. The splice must be found from the baseline
    alone (mechanism (b)), with no M(2) sibling present."""
    baseline, loop, _ = _model()
    rep = SpliceRepresentation.from_extraction_result(_result([baseline, loop]), name="m")
    assert [(s.site, s.loop_ids) for f in rep.families for s in f.splices] == [(2, ("ℓ1",))]


def test_algebraic_term_render():
    assert AlgebraicTerm(("a", ("b", "c"), "d")).render() == "a;(b@c);d"


def test_count_linear_extensions_matches_topological_sorts():
    """Phase-7 #13(D): the guarded ideal-lattice DP swapped in for the width!
    ``nx.all_topological_sorts`` count returns the byte-identical value below
    cap_nodes, and keeps the same ``>cap -> None`` contract that _sp_exact reads."""
    import networkx as nx

    from procposets.cospan.splice import _count_linear_extensions

    def antichain(n):
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        return g

    def chain(n):
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        g.add_edges_from((i, i + 1) for i in range(n - 1))
        return g

    # small corpus where enumerating all_topological_sorts is cheap -> compare directly
    corpus = [antichain(n) for n in range(8)] + [chain(n) for n in range(1, 8)]
    diamond = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
    v = nx.DiGraph([(0, 2), (1, 2)])
    mixed = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)])
    non_closed = nx.DiGraph([(0, 1), (1, 2)])   # covers only, transitive (0,2) omitted
    corpus += [diamond, v, mixed, non_closed]
    for g in corpus:
        assert _count_linear_extensions(g) == sum(1 for _ in nx.all_topological_sorts(g))

    # the wide cases: DP returns n! in O(2^n) states (the enumeration this replaces
    # would be 11! ~ 4e7 sorts); assert the known factorials without enumerating.
    assert _count_linear_extensions(antichain(10)) == 3_628_800     # 10!
    assert _count_linear_extensions(antichain(11)) == 39_916_800    # 11! (= cap_nodes)
    # > cap_nodes stays "not verifiably exact" (None) -- unchanged contract
    assert _count_linear_extensions(antichain(12)) is None
    assert _count_linear_extensions(chain(20)) is None
