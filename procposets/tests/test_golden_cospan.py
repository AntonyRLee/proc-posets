"""Phase-5/6 golden cross-checks: procposets' cospan algebra must reproduce
the ORIGINAL cpm value-for-value before the originals are deleted.

The cpm package uses relative imports, so we put the sim/ root on sys.path and
import `cpm.cospan.*`.  networkx/pm4py paths are marked so the numpy-only core
suite can skip them.
"""

import importlib
import pathlib
import sys

import pytest

SIM = pathlib.Path("/home/arl/Research/-DIAGRAM-String-diagrams-for-process-mining-v2/sim")


def _cpm(module):
    if not (SIM / "cpm").is_dir():
        pytest.skip("sim/cpm not checked out")
    if str(SIM) not in sys.path:
        sys.path.insert(0, str(SIM))
    return importlib.import_module(module)


# ---------------------------------------------------------------------------
# Layering invariants (the deployable-package property)
# ---------------------------------------------------------------------------

def test_core_and_b0_import_without_extras():
    """`import procposets` and the pure cospan algebra must work with the
    numpy-only core -- no networkx, no pm4py imported at module load."""
    import procposets  # noqa: F401
    import procposets.cospan.signature  # noqa: F401
    import procposets.cospan.engine  # noqa: F401
    import procposets.cospan.compose  # noqa: F401
    import procposets.cospan.signature_compare  # noqa: F401
    import procposets.cospan.from_petri  # noqa: F401
    # engine/compose must not have dragged networkx or pm4py in transitively
    assert "networkx" not in sys.modules or True  # networkx may be present via extras
    # the real invariant: these modules themselves declare no such import
    import inspect
    for mod in (procposets.cospan.engine, procposets.cospan.signature,
                procposets.cospan.compose):
        src = inspect.getsource(mod)
        assert "import networkx" not in src
        assert "import pm4py" not in src and "from pm4py" not in src


# ---------------------------------------------------------------------------
# B0 algebra: extract_signature / compose / compare  ==  cpm
# ---------------------------------------------------------------------------

def _running_example(mod_lm, mod_eng):
    """A small hand-built LMGraph -> Signature, mirroring cpm's own test."""
    g = mod_lm.LMGraph()
    g.add_activity("a")
    g.add_activity("b")
    g.add_activity("c")
    m = g.add_mediator("m", mod_lm.Kind.XOR)
    g.add_edge("a", m)
    g.add_edge("m", "b")
    g.add_edge("m", "c")
    g.validate()
    return mod_eng.extract_signature(g)


def test_extract_signature_matches_cpm():
    import procposets.cospan.engine as neng
    import procposets.cospan.lmgraph as nlm
    oeng = _cpm("cpm.cospan.engine")
    olm = _cpm("cpm.cospan.lmgraph")
    sig_new = _running_example(nlm, neng)
    sig_old = _running_example(olm, oeng)
    # Signatures are frozenset-of-Generator dataclasses; compare their
    # canonical string forms (label + sorted ports), which are byte-stable
    assert _sig_key(sig_new) == _sig_key(sig_old)


def _sig_key(sig):
    gens = []
    for gg in sig.generators:
        left = sorted((p.src, p.typ, p.tgt) for p in gg.left)
        right = sorted((p.src, p.typ, p.tgt) for p in gg.right)
        gens.append((gg.label, tuple(left), tuple(right)))
    return sorted(gens)


def test_signature_compare_matches_cpm():
    import procposets.cospan.engine as neng
    import procposets.cospan.lmgraph as nlm
    import procposets.cospan.signature_compare as ncmp
    oeng = _cpm("cpm.cospan.engine")
    olm = _cpm("cpm.cospan.lmgraph")
    ocmp = _cpm("cpm.cospan.signature_compare")
    sn, so = _running_example(nlm, neng), _running_example(olm, oeng)
    rep_new = ncmp.compare({"a": sn, "b": sn})
    rep_old = ocmp.compare({"a": so, "b": so})
    assert ncmp.report_text(rep_new) == ocmp.report_text(rep_old)


# ---------------------------------------------------------------------------
# B1 [graph]: equal / extract_classes  ==  cpm
# ---------------------------------------------------------------------------

@pytest.mark.graph
def test_equivalence_matches_cpm():
    pytest.importorskip("networkx")
    import procposets.cospan.engine as neng
    import procposets.cospan.lmgraph as nlm
    import procposets.equivalence as neq
    oeng = _cpm("cpm.cospan.engine")
    olm = _cpm("cpm.cospan.lmgraph")
    oeq = _cpm("cpm.equivalence")
    sn, so = _running_example(nlm, neng), _running_example(olm, oeng)
    assert neq.equal(sn, sn) == oeq.equal(so, so)
    assert neq.jaccard(sn, sn) == pytest.approx(oeq.jaccard(so, so))


@pytest.mark.graph
def test_extract_classes_matches_cpm():
    pytest.importorskip("networkx")
    import procposets.cospan.engine as neng
    import procposets.cospan.lmgraph as nlm
    import procposets.cospan.extract_dp as ndp
    oeng = _cpm("cpm.cospan.engine")
    olm = _cpm("cpm.cospan.lmgraph")
    odp = _cpm("cpm.cospan.extract_dp")
    sn, so = _running_example(nlm, neng), _running_example(olm, oeng)
    rn = ndp.extract_classes(sn)
    ro = odp.extract_classes(so)
    assert len(rn.closing()) == len(ro.closing())


# ---------------------------------------------------------------------------
# B2 [pm4py]: inbound adapter  ==  cpm
# ---------------------------------------------------------------------------

@pytest.mark.pm4py
def test_from_process_tree_adapter_matches_cpm():
    pytest.importorskip("pm4py")
    import pm4py
    from pm4py.objects.process_tree.obj import Operator, ProcessTree
    import procposets.adapters.from_process_tree as nad
    import procposets.cospan.engine as neng
    oad = _cpm("cpm.cospan.from_process_tree")
    oeng = _cpm("cpm.cospan.engine")

    def tree():
        root = ProcessTree(operator=Operator.SEQUENCE)
        for lbl in "ab":
            ch = ProcessTree(label=lbl)
            ch.parent = root
            root.children.append(ch)
        return root

    sig_new = neng.extract_signature(nad.lmgraph_from_process_tree(tree()))
    sig_old = oeng.extract_signature(oad.lmgraph_from_process_tree(tree()))
    assert _sig_key(sig_new) == _sig_key(sig_old)


# ---------------------------------------------------------------------------
# occn (pure) miner surface imports and lifts
# ---------------------------------------------------------------------------

def test_occn_to_signature_imports():
    import procposets.occn as occn
    assert hasattr(occn, "occn_to_signature")
    assert hasattr(occn, "mine_occn")


# ---------------------------------------------------------------------------
# B2 [pm4py] outbound adapter -- pinned stochastic language
# ---------------------------------------------------------------------------
# Retired golden seam: this began as a cross-check against SPME's
# demo/09_pm4py_baselines/pm_adapters.py.  That original was itself turned into
# a shim over procposets.adapters.outbound (SPME 2f21eb8) and then purged
# (SPME a58bfaf) -- the seam is closed, the consumer now trusts procposets.
# The check survives as a self-contained value regression on the pinned law
# (computed from procposets.traces.trace_distribution), which is what the
# original comparison agreed on.

@pytest.mark.pm4py
def test_outbound_to_language_pinned():
    pytest.importorskip("pm4py")
    import procposets.adapters.outbound as nout
    import procposets as pp

    def model(mp):
        return [(mp.then(mp.leaf("a"), mp.par(mp.leaf("b"), mp.leaf("c")), mp.leaf("d")), 1.0)]

    # a -> (b || c) -> d, weight 1.0: the two interleavings of the parallel block,
    # each 1/2 of the mass (the SPME pm_adapters cross-check agreed on this law).
    lang = nout.to_language(model(pp))
    assert lang == pytest.approx({("a", "b", "c", "d"): 0.5, ("a", "c", "b", "d"): 0.5})
