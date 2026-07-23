"""Behavioural-equivalence oracle: ``compose_signature`` over a
``FactoredSignature`` must yield **the same composites** as over the
materialised slow ``Signature`` -- the seam proving the lazy fire-time ``><``
join equals the pre-built ``B x F`` product.  The factored path realises the
identical concrete generators the slow engine would have materialised, in the
same deterministic (``str``-sorted) candidate order, so we can assert exact
list equality (placements and all), strictly stronger than the planned
label-multiset check.

Plus the laziness guard: a sequential chain banking k typed ports into one hub
whose every in-arc is an XOR of a live and a dead alternative has ``2^k``
concrete left contexts but exactly ONE ever pool-covered.  At k=20 the slow
engine would have to build ~10^6 hub generators; the factored path composes the
single composite through per-arc coverage alone -- completion of the test at
all is the regression signal (equality with the slow engine is pinned at k=8,
where the 256-generator materialisation is still cheap).
"""

from __future__ import annotations

from procposets.cospan.compose import LoopBox, compose_signature
from procposets.cospan.engine import extract_signature
from procposets.cospan.lmgraph import Kind, LMGraph
from procposets.cospan.skeleton import extract_skeleton
from procposets.cospan.typebalance import Profile

from .test_cospan_engine_running_example import build_running_example
from .test_cospan_extract_fast import _typed_hub
from .test_cospan_surface_termini import _mixed_graph


def _both(g, kappa=None, surface=False, **kw):
    slow = compose_signature(
        extract_signature(g, kappa, surface_termini=surface), **kw
    )
    fact = compose_signature(
        extract_skeleton(g, kappa, surface_termini=surface), **kw
    )
    return slow, fact


def test_running_example_composites_identical():
    """The golden 12: same list (content AND order) from both forms."""
    g = build_running_example()
    slow, fact = _both(g, start_label="G1", end_label="G2")
    assert fact == slow
    assert len(fact) == 12
    for c in fact:
        assert c.placements[0].label == "G1" and not c.placements[0].left
        assert c.placements[-1].label == "G2" and not c.placements[-1].right


def test_typed_hub_composites_identical():
    """The cross-type join: the hub's 16 slow contexts vs the lazy 4x4 pairing
    -- covered lefts and full right family must pair to the same firings."""
    slow, fact = _both(_typed_hub())
    assert fact == slow
    assert fact  # the seam is non-vacuous: some composite completes


def _loop_graph() -> LMGraph:
    """``sp`` seeds x into a self-looping ``L`` (XOR: sp|L -> L); ``sq -> N``
    drains independently -- the compose loop-dedup shape, as an LM-graph, so
    the factored path exercises the LoopBox truncation branch too."""
    g = LMGraph()
    for a in ("sp", "L", "sq", "N"):
        g.add_activity(a)
    g.add_mediator("m1", Kind.XOR)
    g.add_mediator("m2", Kind.XOR)
    g.add_edge("sp", "m1", "t")
    g.add_edge("L", "m1", "t")
    g.add_edge("m1", "L", "t")
    g.add_edge("sq", "m2", "u")
    g.add_edge("m2", "N", "u")
    g.validate()
    return g


def test_loopbox_composites_identical():
    slow, fact = _both(_loop_graph(), unroll=2)
    assert fact == slow
    assert any(
        isinstance(p, LoopBox) for c in fact for p in c.placements
    ), "expected LoopBox composites (the truncation branch must be exercised)"


def test_mixed_graph_composites_identical():
    """Strip mode completes ([X, Y]); surfaced mode dead-ends identically on
    the unconsumed gamma2 leg (both engines return no composites)."""
    slow, fact = _both(_mixed_graph())
    assert fact == slow and fact
    slow_s, fact_s = _both(_mixed_graph(), surface=True)
    assert fact_s == slow_s == []


def test_surfaced_pure_terminus_collapses_at_compose_time():
    """A COMPLETING surfaced compose (not just the dead-end degenerate): the
    running example's ``G2_k`` raw right families are pure ``gamma2`` termini,
    so ``collapse_terminus`` must collapse them to zero-right inside
    ``right_bundles`` when the factored candidates are built -- and the golden
    12 must come out unchanged (surfaced == strip here, every terminus being
    pure and untyped)."""
    g = build_running_example()
    slow, fact = _both(g, surface=True, start_label="G1", end_label="G2")
    assert fact == slow
    assert len(fact) == 12


def test_kappa_joined_at_fire_time_matches_slow():
    """``kappa`` on the FactoredSignature filters pairs at fire-time; the slow
    path filtered them at extraction.  Same composites either way."""
    licensed = {"X": Profile(creates=frozenset({"order"}))}
    slow, fact = _both(_mixed_graph(), kappa=licensed)
    assert fact == slow and fact
    rejecting = {"X": Profile()}
    slow_r, fact_r = _both(_mixed_graph(), kappa=rejecting)
    assert fact_r == slow_r == []


def test_kappa_rejects_mid_run_candidate():
    """The fire-time guard itself (not just source seeding) must bite: ``Y`` is
    a MID-RUN candidate (nonempty left, fired from the pool), so constraining
    it exercises ``admissible`` inside the ready-candidate join -- the seam the
    module docstring advertises.  Rejecting ``Y`` (its context consumes
    ``order`` unlicensed) dead-ends BOTH paths after seeding ``X``; deleting
    the fire-time guard would let the factored path fire ``Y`` anyway and
    return ``[X, Y]`` against the slow ``[]``."""
    rejecting = {"Y": Profile()}
    slow_r, fact_r = _both(_mixed_graph(), kappa=rejecting)
    assert fact_r == slow_r == []
    licensed = {"Y": Profile(consumes=frozenset({"order"}))}
    slow_l, fact_l = _both(_mixed_graph(), kappa=licensed)
    assert fact_l == slow_l and fact_l


def _chain_hub(k: int) -> LMGraph:
    """``P1;...;Pk`` fire in a forced sequence, each also banking a typed port
    into hub ``H``; each ``H`` in-arc ``i`` is an XOR of live ``Pi`` and dead
    ``Ri`` (kept off a never-seeded 2-cycle with ``Si``), so ``|B_H| = 2^k``
    but exactly one bundle is ever pool-covered.  ``H`` drains through ``E``.
    The forced sequencing keeps the composite search linear -- the exponential
    lives ONLY in the left family, i.e. exactly where laziness must win."""
    g = LMGraph()
    g.add_activity("H")
    g.add_activity("E")
    for i in range(1, k + 1):
        for a in (f"P{i}", f"R{i}", f"S{i}"):
            g.add_activity(a)
        g.add_mediator(f"q{i}", Kind.XOR)     # {Pi | Ri} -> H
        g.add_edge(f"P{i}", f"q{i}", f"t{i}")
        g.add_edge(f"R{i}", f"q{i}", f"t{i}")
        g.add_edge(f"q{i}", "H", f"t{i}")
        g.add_mediator(f"rs{i}", Kind.XOR)    # Si -> Ri
        g.add_mediator(f"sr{i}", Kind.XOR)    # Ri -> Si  (the dead 2-cycle)
        g.add_edge(f"S{i}", f"rs{i}", f"t{i}")
        g.add_edge(f"rs{i}", f"R{i}", f"t{i}")
        g.add_edge(f"R{i}", f"sr{i}", f"t{i}")
        g.add_edge(f"sr{i}", f"S{i}", f"t{i}")
        if i < k:
            g.add_mediator(f"c{i}", Kind.XOR)  # Pi -> P(i+1): the forced chain
            g.add_edge(f"P{i}", f"c{i}", "chain")
            g.add_edge(f"c{i}", f"P{i + 1}", "chain")
    g.add_mediator("he", Kind.XOR)
    g.add_edge("H", "he", "t1")
    g.add_edge("he", "E", "t1")
    g.validate()
    return g


def test_chain_hub_k8_composites_identical():
    """Below the cliff (2^8 = 256 hub generators) both engines agree exactly."""
    slow, fact = _both(_chain_hub(8))
    assert fact == slow
    assert len(fact) == 1
    labels = sorted(fact[0].labels())
    assert labels == sorted([f"P{i}" for i in range(1, 9)] + ["H", "E"])


def test_chain_hub_k20_lazy_join_stays_output_sensitive(monkeypatch):
    """2^20 (~10^6) concrete hub contexts, ONE covered: the factored path must
    compose the single composite without ever forming the left family.  The
    natural regression route -- materialise ``left_bundles`` then filter -- is
    pinned by a raising spy (the compose path never legitimately needs it);
    a ``ready_lefts``-internal full product would additionally re-build ~10^6
    twenty-leg bundles per readiness check and drag the suite.  The slow
    engine is (deliberately) never run at this width."""
    from procposets.cospan.skeleton import FactoredGenerator

    def _no_materialise(self):
        raise AssertionError(
            "left_bundles() materialised during factored compose -- the lazy "
            "ready_lefts covered-product has regressed to build-then-filter"
        )

    fs = extract_skeleton(_chain_hub(20))
    monkeypatch.setattr(FactoredGenerator, "left_bundles", _no_materialise)
    fact = compose_signature(fs)
    assert len(fact) == 1
    labels = sorted(fact[0].labels())
    assert labels == sorted([f"P{i}" for i in range(1, 21)] + ["H", "E"])
