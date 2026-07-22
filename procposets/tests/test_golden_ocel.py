"""Golden cross-checks for the ported OCEL bridge: procposets' ``discover`` /
``cospan.simulate`` / ``cospan.faithful_simulate`` must reproduce the ORIGINAL
cpm value-for-value before the originals are deleted.

The type barrier (cpm and procposets have distinct Signature/Generator/Port
classes) is sidestepped by comparing through package-agnostic currency:

* discover  -- both discover from the *same* pm4py OCEL; compare the resulting
  signatures by a structural canonical key.
* simulate  -- discover in each package (canonically equal per the discover
  check), then simulate in each package and compare the OCEL relations frame.
* faithful  -- ``faithful_simulate`` is self-contained and duck-typed, so running
  procposets' copy on the SAME (cpm) signature and comparing the frame proves the
  copied logic did not diverge.
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


def _pkey(p):
    return (p.src, p.typ, p.tgt)


def _gkey(g):
    cons = sorted(
        (c.rel, c.rhs, tuple(sorted((_pkey(pt), co) for pt, co in c.terms)))
        for c in g.constraints
    )
    return (
        g.label,
        tuple(sorted(map(_pkey, g.left))),
        tuple(sorted(map(_pkey, g.right))),
        tuple(cons),
    )


# BPMN node labels are fresh pm4py UUIDs minted per ``convert_to_bpmn`` call, so a
# cross-RUN value comparison is meaningless (two discoveries => two UUID sets). The
# BPMN adapter's byte-exactness is gated in-process by the cospan/viz goldens; here we
# only assert its discover orchestration yields a non-empty signature.
_STABLE = ("PN", "PT", "CN", "OCPN", "OCCN")


def _canon(sigs):
    return {cls: sorted(_gkey(g) for g in sigs[cls].generators) for cls in _STABLE if cls in sigs}


def _ed_master():
    return _cpm("demos.ed_chest_pain.master_signature").SIGNATURE


def _liss_master():
    return _cpm("demos.liss_order_handling.liss_faithful_signature").SIGNATURE


def test_discover_signatures_matches_cpm():
    pytest.importorskip("pm4py")
    pytest.importorskip("pandas")
    csim = _cpm("cpm.simulate")
    cdisc = _cpm("cpm.discover")
    import procposets.discover as ndisc

    ocel = csim.ocel_from_signature(_ed_master(), max_loops=3, faithful=False)
    c_sigs = cdisc.discover_signatures(ocel)
    n_sigs = ndisc.discover_signatures(ocel)
    assert _canon(c_sigs) == _canon(n_sigs)
    # BPMN: UUID labels are cross-run unstable; check only structural presence.
    assert bool(c_sigs["BPMN"].generators) == bool(n_sigs["BPMN"].generators)


def test_simulate_round_trip_matches_cpm():
    pytest.importorskip("pm4py")
    pytest.importorskip("pandas")
    csim = _cpm("cpm.simulate")
    cdisc = _cpm("cpm.discover")
    import procposets.cospan.simulate as nsim
    import procposets.discover as ndisc

    ocel = csim.ocel_from_signature(_ed_master(), max_loops=3, faithful=False)
    c_sig = cdisc.discover_signatures(ocel)["PN"]
    n_sig = ndisc.discover_signatures(ocel)["PN"]
    assert n_sig.generators, "PN discovery unexpectedly empty"

    c_out = csim.ocel_from_signature(c_sig, faithful=False)
    n_out = nsim.ocel_from_signature(n_sig, faithful=False)
    assert c_out.relations.equals(n_out.relations)


def test_faithful_simulate_matches_cpm():
    pytest.importorskip("pm4py")
    pytest.importorskip("pandas")
    cf = _cpm("cpm.faithful_simulate")
    import procposets.cospan.faithful_simulate as nf

    master = _liss_master()  # cpm-typed; faithful_simulate is self-contained + duck-typed
    assert nf.needs_faithful(master) == cf.needs_faithful(master) is True

    a = cf.faithful_ocel_from_signature(master, n_runs=30, seed=3)
    b = nf.faithful_ocel_from_signature(master, n_runs=30, seed=3)
    assert a.relations.equals(b.relations)
