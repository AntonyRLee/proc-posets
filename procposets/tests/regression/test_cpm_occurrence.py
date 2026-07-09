"""Tests for the occurrence-net DAG comparison (CLASS_EXTRACTION.md §19)."""

from __future__ import annotations

from procposets.cospan.class_extraction import NamedMorphism
from procposets.cospan.dag_diff import diff_dags
from procposets.cospan.occurrence import (
    IN,
    anchor_types,
    canonical_key,
    history_keys,
    is_isomorphic,
    to_event_dag,
)
from procposets.cospan.signature import Generator, Port


def _gen(label, left, right):
    return Generator(label=label, left=frozenset(left), right=frozenset(right))


def _nm(name, body, boundary=frozenset()):
    return NamedMorphism(name=name, boundary=boundary, body=tuple(body))


def _dag(nm):
    return to_event_dag(nm, {nm.name: nm})


# ports: distinct (src,tgt) so the matcher routes producers to the right
# consumer; all type 't' unless a test needs distinct types.
def P(src, tgt, typ="t"):
    return Port(src=src, typ=typ, tgt=tgt)


def test_n_poset_and_complete_bipartite_are_distinguished():
    """The §19c whole point: ``(a@b);(c@d)`` with N-wiring (a<c, b<c, b<d) is a
    DIFFERENT structure from the complete (2,2) order, even though they share a
    label skeleton. They must get different canonical keys / not be iso."""
    a = _gen("a", [], [P("a", "c")])
    b = _gen("b", [], [P("b", "c"), P("b", "d")])
    c_n = _gen("c", [P("a", "c"), P("b", "c")], [])
    d_n = _gen("d", [P("b", "d")], [])
    n_poset = _nm("N", [frozenset({a, b}), frozenset({c_n, d_n})])

    a2 = _gen("a", [], [P("a", "c"), P("a", "d")])
    b2 = _gen("b", [], [P("b", "c"), P("b", "d")])
    c_f = _gen("c", [P("a", "c"), P("b", "c")], [])
    d_f = _gen("d", [P("a", "d"), P("b", "d")], [])
    complete = _nm("K", [frozenset({a2, b2}), frozenset({c_f, d_f})])

    dn, dk = _dag(n_poset), _dag(complete)
    assert canonical_key(dn) != canonical_key(dk)
    assert not is_isomorphic(dn, dk)


def test_parallel_wires_preserve_multiplicity():
    """Two typed wires between the same node pair must not collapse (the
    DiGraph bug fixed in §19's build): a single shared sink edge keeps the
    full type-multiset."""
    a = _gen("a", [], [P("a", "b", "t1"), P("a", "b", "t2")])
    b = _gen("b", [P("a", "b", "t1"), P("a", "b", "t2")], [])
    two = _dag(_nm("two", [a, b]))

    a1 = _gen("a", [], [P("a", "b", "t1")])
    b1 = _gen("b", [P("a", "b", "t1")], [])
    one = _dag(_nm("one", [a1, b1]))

    # find the a->b edge in each and compare its type-multiset
    def ab_typs(d):
        nodes = {n: dat["label"] for n, dat in d.graph.nodes(data=True)}
        for u, v, dd in d.graph.edges(data=True):
            if nodes.get(u) == "a" and nodes.get(v) == "b":
                return dd["typs"]
        return None

    assert ab_typs(two) == ("t1", "t2")
    assert ab_typs(one) == ("t1",)
    assert not is_isomorphic(two, one)


def test_boundary_wrappers_collapse_to_the_single_gamma_boundary():
    """``START_<ot>``/``END_<ot>`` wrapper activities (§14 B3, §40) are contracted onto the
    single ``gamma1`` source / ``gamma2`` sink the occurrence DAG draws -- not kept as their
    own per-type nodes, and not dropped (which would disconnect the wrapped core). The core
    ``a`` survives, wired ``gamma1 -> a -> gamma2``."""
    start = _gen("START_x", [], [P("s", "a")])
    a = _gen("a", [P("s", "a")], [P("a", "e")])
    end = _gen("END_x", [P("a", "e")], [])
    wrapped = _dag(_nm("w", [start, a, end]))

    lab = {n: d["label"] for n, d in wrapped.graph.nodes(data=True)}
    assert "START_x" not in lab.values() and "END_x" not in lab.values()  # collapsed
    assert set(lab.values()) == {"gamma1", "a", "gamma2"}  # one γ1/γ2 boundary + the core
    assert {(lab[u], lab[v]) for u, v in wrapped.graph.edges()} == {("gamma1", "a"), ("a", "gamma2")}


def test_isomorphic_fragments_from_two_models_merge_in_diff():
    """Two structurally-identical closings from different models land in one
    DagClass whose membership lists both models; a third, different structure
    stays separate and unique to its model."""
    a = _gen("a", [], [P("a", "z")])
    z = _gen("z", [P("a", "z")], [])
    shared = _nm("M1", [a, z])

    b = _gen("b", [], [P("b", "z")])
    z2 = _gen("z", [P("b", "z")], [])
    other = _nm("M2", [b, z2])

    from procposets.cospan.class_extraction import ExtractionResult

    res_a = ExtractionResult(fragments={"M1": shared}, valid_generators=set(), frontiers_visited=0)
    res_b = ExtractionResult(fragments={"M1": shared}, valid_generators=set(), frontiers_visited=0)
    res_c = ExtractionResult(fragments={"M2": other}, valid_generators=set(), frontiers_visited=0)

    rep = diff_dags({"alpha": res_a, "beta": res_b, "gamma": res_c})
    shared_classes = [c for c in rep.closing_classes if c.models == frozenset({"alpha", "beta"})]
    unique_classes = [c for c in rep.closing_classes if c.models == frozenset({"gamma"})]
    assert len(shared_classes) == 1
    assert len(unique_classes) == 1
    assert len(rep.shared_by_all()) == 0


def test_boundary_wrappers_are_fully_absorbed_leaving_a_true_source():
    """The §20 fix: a contracted ``START_`` must not leave a stub wire into the
    first real activity -- that activity becomes a true source (in-degree 0),
    matching a zero-input ``gamma1`` from another adapter. Contracting only the
    wrapper node while keeping its edge was what kept the closings overlay from
    going green."""
    start = _gen("START_x", [], [P("s", "g1")])
    g1 = _gen("gamma1", [P("s", "g1")], [P("g1", "e")])
    e = _gen("e", [P("g1", "e")], [])
    d = _dag(_nm("w", [start, g1, e]))

    nodes = {dd["label"]: n for n, dd in d.graph.nodes(data=True)}
    assert d.graph.in_degree(nodes["gamma1"]) == 0  # true source, no IN stub
    assert not d.graph.has_edge(IN, nodes["gamma1"])

    # ... and it now matches an adapter whose gamma1 is natively zero-input
    g1_native = _gen("gamma1", [], [P("g1", "e")])
    e2 = _gen("e", [P("g1", "e")], [])
    native = _dag(_nm("n", [g1_native, e2]))
    assert is_isomorphic(d, native)


def test_real_loop_anchor_is_kept_not_absorbed():
    """Only *wrapper*-induced boundary wires are dropped; a loop's real anchor
    (``nm.boundary``) stays as IN-rooted edges (it is the splice point, §19e)."""
    from collections import Counter as _C

    from procposets.cospan.class_extraction import _to_key

    # a one-event loop body consuming an anchor token of type 'img'
    body_gen = _gen("x", [P("a", "x", "img")], [P("a", "x", "img")])
    boundary = _to_key(_C({P("a", "x", "img"): 1}))
    d = _dag(_nm("L", [body_gen], boundary=boundary))
    xnode = next(n for n, dd in d.graph.nodes(data=True) if dd["label"] == "x")
    assert d.graph.has_edge(IN, xnode)  # anchor wire kept


def test_history_keys_merge_shared_prefix_but_not_divergent_wiring():
    """The causal-prefix merge (§20d overlay): two events share a history key
    iff their causal cones agree including typed wiring. A shared prefix event
    merges across fragments; an event reached by a different wiring does not."""
    # fragment 1: a -> b (via t), a -> c (via t)
    a1 = _gen("a", [], [P("a", "b"), P("a", "c")])
    b1 = _gen("b", [P("a", "b")], [])
    c1 = _gen("c", [P("a", "c")], [])
    f1 = to_event_dag(_nm("f1", [a1, frozenset({b1, c1})]), {})

    # fragment 2: a -> b (via t), then a different tail
    a2 = _gen("a", [], [P("a", "b"), P("a", "d")])
    b2 = _gen("b", [P("a", "b")], [])
    d2 = _gen("d", [P("a", "d")], [])
    f2 = to_event_dag(_nm("f2", [a2, frozenset({b2, d2})]), {})

    h1 = history_keys(f1)
    h2 = history_keys(f2)
    key_of = lambda h, d, lab: next(h[n] for n, dd in d.graph.nodes(data=True) if dd["label"] == lab)

    # 'a' has the same opening in both -> same history key (merges)
    assert key_of(h1, f1, "a") == key_of(h2, f2, "a")
    # 'b' (a -> b via t) is the same causal cone in both -> merges
    assert key_of(h1, f1, "b") == key_of(h2, f2, "b")
    # 'c' and 'd' are distinct events -> distinct keys (stay forked)
    assert key_of(h1, f1, "c") != key_of(h2, f2, "d")


def test_spine_collapses_loop_unrollings_to_one_family():
    """Round-2 splice-aware comparison (§22h): a closing that traverses a loop
    (M(2,σ)) reduces to the same loop-free spine as the baseline (M(1,σ)), so
    the m-unrolling is recognised as the same family."""
    from procposets.cospan.dag_diff import closing_spine

    # baseline closing:  gamma1 -> a -> z  (a -> z via type t)
    g1 = _gen("gamma1", [], [P("g1", "a", "t")])
    a = _gen("a", [P("g1", "a", "t")], [P("a", "z", "t")])
    z = _gen("z", [P("a", "z", "t")], [])
    baseline = _nm("M1", [g1, a, z])

    # a loop  b;c  anchored somewhere, and an unrolled closing that contains it
    b = _gen("b", [P("a", "z", "t")], [P("b", "c", "t")])
    c = _gen("c", [P("b", "c", "t")], [P("a", "z", "t")])
    loop = _nm("L1", [b, c], boundary=frozenset())  # body = (b, c)
    # unrolled = gamma1 -> a -> [b -> c] -> z   (the loop body spliced before z)
    unrolled = _nm("M2", [g1, a, b, c, z])

    by_name = {"M1": baseline, "M2": unrolled, "L1": loop}
    sp_base = closing_spine(baseline, [loop], by_name)
    sp_unrolled = closing_spine(unrolled, [loop], by_name)
    assert sp_base == sp_unrolled, "M(2) should reduce to the M(1) spine"
    assert sp_base == ("gamma1", "a", "z")


def test_diff_dag_families_folds_unrollings_no_spurious_combo():
    """Family-level overlay diff (§22h/§23b): a model that *lists* an M(2,σ)
    loop unrolling (where another model splices its loop and never lists it)
    must not surface as a spurious model-unique structure. The exact diff keeps
    them separate; ``diff_dag_families`` folds the unrolling onto the baseline
    spine, drawing the loop-free baseline as the family rep."""
    from collections import Counter

    from procposets.cospan.class_extraction import ExtractionResult, _to_key
    from procposets.cospan.dag_diff import diff_dag_families

    g1 = _gen("gamma1", [], [P("g1", "a", "t")])
    a = _gen("a", [P("g1", "a", "t")], [P("a", "z", "t")])
    z = _gen("z", [P("a", "z", "t")], [])
    baseline = _nm("M1", [g1, a, z])
    b = _gen("b", [P("a", "z", "t")], [P("b", "c", "t")])
    c = _gen("c", [P("b", "c", "t")], [P("a", "z", "t")])
    # L1 anchors at the post-a frontier P(a,z,t): non-empty boundary => a loop
    loop = _nm("L1", [b, c], boundary=_to_key(Counter({P("a", "z", "t"): 1})))
    unrolled = _nm("M2", [g1, a, b, c, z])  # baseline with L1 spliced before z

    def result(frags):
        return ExtractionResult(
            fragments={f.name: f for f in frags}, valid_generators=set(), frontiers_visited=0
        )

    # "splicer" lists only the baseline (it would splice L1); "lister" also lists M2
    results = {"splicer": result([baseline, loop]), "lister": result([baseline, unrolled, loop])}

    # exact: M2 is a lister-only closing class
    exact = diff_dags(results)
    assert any(cls.models == frozenset({"lister"}) for cls in exact.closing_classes)

    # family: the unrolling folds onto the shared baseline spine -> a single
    # closing family, shared by both, with no lister-only family
    fam = diff_dag_families(results)
    assert len(fam.closing_classes) == 1
    cls = fam.closing_classes[0]
    assert cls.models == frozenset({"splicer", "lister"})
    # rep is the loop-free baseline (gamma1,a,z), not M2; 4 event nodes incl. the single
    # γ2 sink now closing z's open boundary (§40)
    assert len(cls.rep.event_nodes()) == 4


def test_anchor_types_projects_boundary_to_type_multiset():
    """A loop's anchor (§19e) is its boundary's type-multiset -- comparable
    across adapters, unlike raw ``Port``s."""
    from procposets.cospan.class_extraction import _to_key
    from collections import Counter

    boundary = _to_key(Counter({P("x", "y", "img"): 1, P("u", "v", "lab"): 1}))
    nm = _nm("L1", [], boundary=boundary)
    assert anchor_types(nm) == (((False, "img"), 1), ((False, "lab"), 1))
