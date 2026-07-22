"""Golden cross-checks for the post-discovery *analysis* layer: procposets'
``signature_diff`` / ``typebalance`` / ``class_extraction`` / ``discovery_cleanup``
plus the engine's silent-transition elimination must reproduce the ORIGINAL cpm
value-for-value before the originals are deleted.

These four modules are byte-exact ports (only docstrings/comments differ), so the
checks exist to *lock* that equivalence while ``sim/cpm`` is still the independent
oracle -- closing the WS-1.4 golden gaps (the pre-existing cospan golden covered
``engine``/``compose``/``signature_compare`` but not these).

The type barrier (cpm and procposets carry distinct Generator/Port/Signature
classes) is sidestepped the same way the cospan golden does it: build the *same*
fixture natively in each package via a module-parameterized helper, run the target
function in each, and compare through package-agnostic currency (skeletons are
tuples of plain strings; signatures/violations are keyed by label + port triples).
"""

import importlib
import pathlib
import sys

import pytest

SIM = pathlib.Path("/home/arl/Research/string-diagram-process-mining/sim")


def _cpm(module):
    if not (SIM / "cpm").is_dir():
        pytest.skip("sim/cpm not checked out")
    if str(SIM) not in sys.path:
        sys.path.insert(0, str(SIM))
    return importlib.import_module(module)


def _sig_key(sig):
    """Package-agnostic canonical key for a Signature: label + sorted port triples."""
    gens = []
    for g in sig.generators:
        left = sorted((p.src, p.typ, p.tgt) for p in g.left)
        right = sorted((p.src, p.typ, p.tgt) for p in g.right)
        gens.append((g.label, tuple(left), tuple(right)))
    return sorted(gens)


# ---------------------------------------------------------------------------
# signature_diff: the three-level cross-signature structural diff  ==  cpm
# ---------------------------------------------------------------------------

def _diff_fixture(ce, sig):
    """Two ExtractionResults mirroring cpm's own signature_diff fixtures: a
    label/path partition (shared / only_a / only_b) with a START_/END_ boundary
    wrapper on A, so both the diff partition and the B3 boundary-strip fire."""
    G = sig.Generator

    def frag(name, body):
        return ce.NamedMorphism(name, frozenset(), body)

    def result(fragments, valid_labels):
        valid = {G(lbl, frozenset(), frozenset()) for lbl in valid_labels}
        return ce.ExtractionResult(fragments=fragments, valid_generators=valid, frontiers_visited=0)

    start_t = G("START_t", frozenset(), frozenset())
    shared = G("shared", frozenset(), frozenset())
    end_t = G("END_t", frozenset(), frozenset())
    only_a = G("only_a", frozenset(), frozenset())
    only_b = G("only_b", frozenset(), frozenset())

    frags_a = {"M1": frag("M1", (start_t, shared, end_t)), "M2": frag("M2", (only_a,))}
    frags_b = {"M1": frag("M1", (shared,)), "M2": frag("M2", (only_b,))}
    ra = result(frags_a, {"shared", "only_a", "START_t", "END_t"})
    rb = result(frags_b, {"shared", "only_b"})
    return ra, rb


def test_signature_diff_matches_cpm():
    import procposets.cospan.class_extraction as nce
    import procposets.cospan.signature as nsig
    import procposets.cospan.signature_diff as nsd

    oce = _cpm("cpm.cospan.class_extraction")
    osig = _cpm("cpm.cospan.signature")
    osd = _cpm("cpm.cospan.signature_diff")

    ra_n, rb_n = _diff_fixture(nce, nsig)
    ra_o, rb_o = _diff_fixture(oce, osig)

    for strip in (True, False):
        rn = nsd.diff_signatures(ra_n, rb_n, strip_boundary=strip)
        ro = osd.diff_signatures(ra_o, rb_o, strip_boundary=strip)
        # skeletons are tuples of plain strings -> frozensets compare directly
        assert rn.labels_only_a == ro.labels_only_a
        assert rn.labels_only_b == ro.labels_only_b
        assert rn.labels_in_both == ro.labels_in_both
        assert rn.paths_only_a == ro.paths_only_a
        assert rn.paths_only_b == ro.paths_only_b
        assert rn.paths_in_both == ro.paths_in_both
        assert rn.loops_in_both == ro.loops_in_both
        assert rn.boundary_stripped == ro.boundary_stripped == strip
        # the strongest lock: the full rendered report text is byte-identical
        assert rn.pretty() == ro.pretty()

    # non-vacuity: the fixture actually exercises a label diff, a shared path,
    # and the boundary strip (interior "shared" survives; START_t/END_t stripped)
    rn = nsd.diff_signatures(ra_n, rb_n)
    assert rn.labels_only_a == {"only_a", "START_t", "END_t"}
    assert rn.labels_only_b == {"only_b"}
    assert rn.paths_in_both == {("shared",)}

    # pure skeleton helpers also agree (label_skeleton / strip / loop folding)
    skel = (("START_img", "START_pat"), "gamma1", "examine_1", "gamma2", ("END_pat",))
    assert nsd.strip_boundary_wrapper(skel) == osd.strip_boundary_wrapper(skel)
    base, repeated = ("a", "b"), ("a", "b", "a", "b")
    ln = nsd.label_legend({base, repeated}, prefix="L")
    lo = osd.label_legend({base, repeated}, prefix="L")
    bn, _ = nsd.decompose_loops(ln)
    bo, _ = osd.decompose_loops(lo)
    assert nsd.render_skeleton(bn[ln[repeated]]) == osd.render_skeleton(bo[lo[repeated]])


# ---------------------------------------------------------------------------
# typebalance: admissibility validator + engine enforcement  ==  cpm
# ---------------------------------------------------------------------------

def _conversion_graph(lm):
    """p --lab--> x --bed--> q : activity x 'converts' lab into bed."""
    g = lm.LMGraph()
    for a in ("p", "x", "q"):
        g.add_activity(a)
    g.add_mediator("m1", lm.Kind.SEQ)
    g.add_mediator("m2", lm.Kind.SEQ)
    g.add_edge("p", "m1", "lab")
    g.add_edge("m1", "x", "lab")
    g.add_edge("x", "m2", "bed")
    g.add_edge("m2", "q", "bed")
    g.validate()
    return g


def _vkey(violations):
    return sorted(
        (v.generator.label, tuple(sorted(v.bad_creates)), tuple(sorted(v.bad_consumes)))
        for v in violations
    )


def test_typebalance_matches_cpm():
    import procposets.cospan.engine as neng
    import procposets.cospan.lmgraph as nlm
    import procposets.cospan.typebalance as ntb

    oeng = _cpm("cpm.cospan.engine")
    olm = _cpm("cpm.cospan.lmgraph")
    otb = _cpm("cpm.cospan.typebalance")

    gn, go = _conversion_graph(nlm), _conversion_graph(olm)

    def reject(tb):
        return {"p": tb.Profile(creates=frozenset({"lab"})),
                "x": tb.Profile(),  # no licence -> the conversion is rejected
                "q": tb.Profile(consumes=frozenset({"bed"}))}

    def licensed(tb):
        return {"p": tb.Profile(creates=frozenset({"lab"})),
                "x": tb.Profile(creates=frozenset({"bed"}), consumes=frozenset({"lab"})),
                "q": tb.Profile(consumes=frozenset({"bed"}))}

    # (a) engine acceptance/rejection of an unlicensed type conversion agrees
    assert ("x" in neng.extract_signature(gn).labels()) \
        == ("x" in oeng.extract_signature(go).labels()) is True
    assert ("x" in neng.extract_signature(gn, reject(ntb)).labels()) \
        == ("x" in oeng.extract_signature(go, reject(otb)).labels()) is False
    assert ("x" in neng.extract_signature(gn, licensed(ntb)).labels()) \
        == ("x" in oeng.extract_signature(go, licensed(otb)).labels()) is True

    # (b) the type_balance violation report agrees value-for-value. Extract WITHOUT
    # κ so x survives, then score against a κ that *constrains* x with an empty
    # licence (a label absent from κ is unconstrained, so x must be present in κ):
    # its lab->bed conversion is then the sole unlicensed generator. p/q stay
    # absent from κ, hence unconstrained.
    sig_n, sig_o = neng.extract_signature(gn), oeng.extract_signature(go)
    kap_n, kap_o = {"x": ntb.Profile()}, {"x": otb.Profile()}
    vn, vo = ntb.type_balance(sig_n, kap_n), otb.type_balance(sig_o, kap_o)
    assert _vkey(vn) == _vkey(vo) == [("x", ("bed",), ("lab",))]

    # and a fully-licensed κ yields no violations in either package
    assert ntb.type_balance(sig_n, licensed(ntb)) == otb.type_balance(sig_o, licensed(otb)) == []


# ---------------------------------------------------------------------------
# class_extraction: FULL structural equality  ==  cpm
# (the pre-existing cospan golden only compared len(closing()) -- and the
#  running example has no gamma boundary, so that count was 0 == 0, vacuous)
# ---------------------------------------------------------------------------

def _closing_sig(sig):
    """gamma1 -> a -> b -> gamma2 : one genuine closing instance."""
    G, P = sig.Generator, sig.Port
    return sig.Signature(frozenset({
        G("gamma1", frozenset(), frozenset({P("gamma1", "t", "a")})),
        G("a", frozenset({P("gamma1", "t", "a")}), frozenset({P("a", "t", "b")})),
        G("b", frozenset({P("a", "t", "b")}), frozenset({P("b", "t", "gamma2")})),
        G("gamma2", frozenset({P("b", "t", "gamma2")}), frozenset()),
    }))


def _step_lbls(step):
    if isinstance(step, frozenset):
        return tuple(sorted(g.label for g in step))
    lbl = getattr(step, "label", None)
    return lbl if lbl is not None else getattr(step, "name", str(step))  # Generator | Ref


def _paths(result):
    return sorted(tuple(_step_lbls(s) for s in nm.body) for nm in result.closing())


@pytest.mark.graph
def test_extract_classes_structural_matches_cpm():
    pytest.importorskip("networkx")
    import procposets.cospan.class_extraction as nce
    import procposets.cospan.signature as nsig

    oce = _cpm("cpm.cospan.class_extraction")
    osig = _cpm("cpm.cospan.signature")

    rn = nce.extract_classes(_closing_sig(nsig))
    ro = oce.extract_classes(_closing_sig(osig))

    # full structural equality, not merely the closing COUNT
    assert _paths(rn) == _paths(ro) == [("gamma1", "a", "b", "gamma2")]
    assert len(rn.closing()) == len(ro.closing()) == 1
    assert len(rn.loops()) == len(ro.loops())
    assert rn.pretty() == ro.pretty()  # byte-identical rendered catalogue


# ---------------------------------------------------------------------------
# discovery_cleanup: gamma_normalize / degenerate_filtered (silent elim) /
# forget_provenance  ==  cpm
# ---------------------------------------------------------------------------

def _occn_style(sig):
    """START_order -> a -> b -> END_order (OCCN explicit-boundary encoding)."""
    G, P = sig.Generator, sig.Port
    return sig.Signature(frozenset({
        G("START_order", frozenset(), frozenset({P("START_order", "order", "a")})),
        G("a", frozenset({P("START_order", "order", "a")}), frozenset({P("a", "order", "b")})),
        G("b", frozenset({P("a", "order", "b")}), frozenset({P("b", "order", "END_order")})),
        G("END_order", frozenset({P("b", "order", "END_order")}), frozenset()),
    }))


def _ocpn_style(sig):
    """a (zero-left source) -> b -> z (zero-right sink) (OCPN open-boundary encoding)."""
    G, P = sig.Generator, sig.Port
    return sig.Signature(frozenset({
        G("a", frozenset(), frozenset({P("a", "order", "b")})),
        G("b", frozenset({P("a", "order", "b")}), frozenset({P("b", "order", "z")})),
        G("z", frozenset({P("b", "order", "z")}), frozenset()),
    }))


def _selfbounce_sig(sig):
    """`a` is degree-(1,1): its self-bounce context (succ.tgt == 'a') is a
    discovery artifact degenerate_filtered must drop; its normal a->b context survives."""
    G, P = sig.Generator, sig.Port
    return sig.Signature(frozenset({
        G("a", frozenset({P("x", "t", "a")}), frozenset({P("a", "t", "a")})),  # self-bounce
        G("a", frozenset({P("x", "t", "a")}), frozenset({P("a", "t", "b")})),  # real
        G("b", frozenset({P("a", "t", "b")}), frozenset({P("b", "t", "y")})),
    }))


def _provenance_sig(sig):
    """Two `c` generators differing only in the producer `src` -> forget_provenance
    merges them (fungible-token behavioural quotient)."""
    G, P = sig.Generator, sig.Port
    return sig.Signature(frozenset({
        G("c", frozenset({P("p1", "t", "c")}), frozenset({P("c", "t", "d")})),
        G("c", frozenset({P("p2", "t", "c")}), frozenset({P("c", "t", "d")})),
    }))


def test_discovery_cleanup_matches_cpm():
    import procposets.cospan.discovery_cleanup as ndc
    import procposets.cospan.signature as nsig

    odc = _cpm("cpm.cospan.discovery_cleanup")
    osig = _cpm("cpm.cospan.signature")

    # gamma_normalize on both boundary encodings (relabel + synthesis)
    for build in (_occn_style, _ocpn_style):
        assert _sig_key(ndc.gamma_normalize(build(nsig))) == _sig_key(odc.gamma_normalize(build(osig)))

    # degenerate_filtered (silent/self-bounce elimination) + forget_provenance
    assert _sig_key(ndc.degenerate_filtered(_selfbounce_sig(nsig))) \
        == _sig_key(odc.degenerate_filtered(_selfbounce_sig(osig)))
    assert _sig_key(ndc.forget_provenance(_provenance_sig(nsig))) \
        == _sig_key(odc.forget_provenance(_provenance_sig(osig)))

    # non-vacuity: each transform actually changes its fixture
    assert _sig_key(ndc.gamma_normalize(_occn_style(nsig))) != _sig_key(_occn_style(nsig))
    assert len(ndc.degenerate_filtered(_selfbounce_sig(nsig)).generators) == 2  # 3 -> 2
    assert len(ndc.forget_provenance(_provenance_sig(nsig)).generators) == 1  # 2 -> 1


# ---------------------------------------------------------------------------
# silent-transition (tau) elimination: model -> signature drops silents  ==  cpm
# ---------------------------------------------------------------------------

def _silent_chain_graph(lm):
    """a -> [place] -> tau(silent) -> [place] -> b, all typed 't'."""
    g = lm.LMGraph()
    g.add_activity("a")
    g.add_activity("b")
    g.add_mediator("p1", lm.Kind.XOR)
    g.add_mediator("tau", lm.Kind.SEQ, silent=True)
    g.add_mediator("p2", lm.Kind.XOR)
    g.add_edge("a", "p1", "t")
    g.add_edge("p1", "tau", "t")
    g.add_edge("tau", "p2", "t")
    g.add_edge("p2", "b", "t")
    return g


def test_silent_elimination_matches_cpm():
    import procposets.cospan.engine as neng
    import procposets.cospan.lmgraph as nlm

    oeng = _cpm("cpm.cospan.engine")
    olm = _cpm("cpm.cospan.lmgraph")

    gn, go = _silent_chain_graph(nlm), _silent_chain_graph(olm)

    # without_silent contracts the tau node identically
    hn, ho = gn.without_silent(), go.without_silent()
    assert hn.silent == ho.silent == set()
    assert "tau" not in hn.mediators and "tau" not in ho.mediators

    # the spliced path yields the same signature in both packages ...
    sig_n = neng.extract_signature(hn, remove_silent=False)
    sig_o = oeng.extract_signature(ho, remove_silent=False)
    assert _sig_key(sig_n) == _sig_key(sig_o)

    # ... and default elimination is signature-neutral (on == off) in each
    assert _sig_key(neng.extract_signature(gn)) == _sig_key(neng.extract_signature(gn, remove_silent=False))
    assert _sig_key(oeng.extract_signature(go)) == _sig_key(oeng.extract_signature(go, remove_silent=False))

    # non-vacuity: the type-'t' wire survives the tau splice end to end
    assert ("a", "t", "b") in {(p.src, p.typ, p.tgt) for gen in sig_n for p in gen.right}
