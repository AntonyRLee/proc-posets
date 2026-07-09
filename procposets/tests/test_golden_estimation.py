"""Phase-2 golden cross-checks: procposets' A1 estimation + distance core must
reproduce the ORIGINAL NPMLE fit and SPME smd value-for-value.

Skipped if a sibling repo is absent (procposets stays independently testable).
NPMLE modules form a package (relative imports), so we import them under the
`poset_mixture` package name via its own layout; SPME similarly under `spm`.
"""

import importlib
import pathlib
import random
import sys

import numpy as np
import pytest

RESEARCH = pathlib.Path("/home/arl/Research")
NPMLE = RESEARCH / "poset-mixture-npmle"
SPME = RESEARCH / "stochastic_process_mining" / "experiments"


def _orig(pkg_root: pathlib.Path, module: str):
    if not (pkg_root / module.replace(".", "/")).with_suffix(".py").is_file():
        pytest.skip(f"original not checked out: {pkg_root}/{module}")
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    return importlib.import_module(module)


# ---------------------------------------------------------------------------
# NPMLE fit (numpy)  ==  procposets fit
# ---------------------------------------------------------------------------

def test_fit_matches_npmle_on_demo_truth():
    import procposets as new
    opos = _orig(NPMLE, "poset_mixture.posets")
    osim = _orig(NPMLE, "poset_mixture.simulate")
    onp = _orig(NPMLE, "poset_mixture.npmle")
    olik = _orig(NPMLE, "poset_mixture.likelihood")

    def build(mod_pos, mod_sim, mod_lik, mod_fit):
        P1 = mod_pos.series("a", mod_pos.parallel("b", "c"), "d")
        P2 = mod_pos.series("a", "c", "b", "d")
        truth = mod_sim.TrueMixture(trees=[P1, P2], weights=[0.6, 0.4])
        g, _ = mod_sim.sample_grouped_log(truth, G=100, n_g=4, seed=11)
        log = mod_lik.GroupedLog(g)
        return mod_fit.fit(log)

    r_new = build(new, new, new, new)
    r_old = build(opos, osim, olik, onp)
    assert r_new.loglik == pytest.approx(r_old.loglik, abs=1e-9)
    assert r_new.gap == pytest.approx(r_old.gap, abs=1e-9)
    mn = {rel: w for rel, _, w, _, _, _ in r_new.poset_marginal()}
    mo = {rel: w for rel, _, w, _, _, _ in r_old.poset_marginal()}
    assert set(mn) == set(mo)
    for rel in mn:
        assert mn[rel] == pytest.approx(mo[rel], abs=1e-9)


def test_moment_initialiser_matches_npmle():
    import procposets as new
    onp = _orig(NPMLE, "poset_mixture.npmle")
    osim = _orig(NPMLE, "poset_mixture.simulate")
    opos = _orig(NPMLE, "poset_mixture.posets")
    olik = _orig(NPMLE, "poset_mixture.likelihood")
    P1 = opos.series("a", opos.parallel("b", "c"), "d")
    P2 = opos.series("a", "c", "b", "d")
    g, _ = osim.sample_grouped_log(osim.TrueMixture(trees=[P1, P2], weights=[0.6, 0.4]),
                                   G=80, n_g=4, seed=5)
    log_old = olik.GroupedLog(g)
    log_new = new.GroupedLog(g)
    for order in (2, 3, "auto"):
        assert (new.fit(log_new, init_order=order).loglik
                == pytest.approx(onp.fit(log_old, init_order=order).loglik, abs=1e-9))


# ---------------------------------------------------------------------------
# SPME smd (stdlib)  ==  procposets smd
# ---------------------------------------------------------------------------

def test_smd_matches_spme():
    import procposets as new
    odist = _orig(SPME, "spm.distance")
    opos = _orig(SPME, "spm.poset")
    # a diamond vs a chain, both label sets {a,b,c,d}
    def diamond(mp):
        return mp.then(mp.leaf("a"), mp.par(mp.leaf("b"), mp.leaf("c")), mp.leaf("d"))
    def chain(mp):
        return mp.then(mp.leaf("a"), mp.leaf("c"), mp.leaf("b"), mp.leaf("d"))
    m_new = [(diamond(new), 0.6), (chain(new), 0.4)]
    m_old = [(diamond(opos), 0.6), (chain(opos), 0.4)]
    m_new2 = [(diamond(new), 1.0)]
    m_old2 = [(diamond(opos), 1.0)]
    d_new, _ = new.smd(m_new, m_new2)
    d_old, _ = odist.smd(m_old, m_old2)
    assert d_new == pytest.approx(d_old, abs=1e-12)


def test_rho_mle_matches_spme():
    import procposets as new
    oest = _orig(SPME, "spm.estimate")
    opos = _orig(SPME, "spm.poset")

    def variants(mp):
        return [mp.then(mp.leaf("a"), mp.leaf("b")), mp.then(mp.leaf("b"), mp.leaf("a"))]

    laws_new = new.variant_laws(variants(new))
    laws_old = oest.variant_laws(variants(opos))
    traces = [("a", "b")] * 7 + [("b", "a")] * 3
    r_new = new.rho_mle(traces, laws_new)
    r_old = oest.rho_mle(traces, laws_old)
    assert r_new == pytest.approx(r_old, abs=1e-9)
