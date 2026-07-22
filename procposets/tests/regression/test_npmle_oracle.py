"""Tests for procposets/oracle.py (class Oracle).

Covers, with oracle-style cross-checks rather than snapshots:

* regime selection (enumeration / meet-closure / lattice-heuristic) and the
  ``exact`` flag, including the closure-cap downgrade;
* THE REDUCTION THEOREM: on seeded random grouped logs (m=4, uniform kernel)
  the forced meet-closure oracle prices identically (1e-12 relative) to the
  full 219-poset enumeration oracle for many random mixture densities;
* enumeration atom counts against OEIS A001035 (19 posets on m=3, 219 on
  m=4) and the labeled-N count (219 - 24 = 195 SP posets on m=4), with a
  brute-force decompose() cross-check;
* lattice-heuristic candidate guarantees: all observed chains and all group
  meets are present; the SP class drops exactly the non-SP candidates
  (constructed so that one group meet is the N);
* price() argmax/score consistency with a direct numpy recomputation from
  oracle.logF;
* the meet-closure candidate list contains the empty order even when the
  closure of the observed chains does not.

Everything is deterministic (all rngs seeded).
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import permutations

import numpy as np
import pytest

from procposets import (
    GroupedLog,
    TimedGroupedLog,
    TrueMixture,
    count_linear_extensions,
    decompose,
    enumerate_posets,
    meet,
    meet_closure,
    parallel,
    rel_from_trace,
    respects,
    sample_grouped_log,
    series,
)
from procposets.oracle import Oracle

ELS = ("a", "b", "c", "d")

# The N poset (the canonical non-SP order): a<c, b<c, b<d.  Already
# transitively closed (no composable pairs).
REL_N = frozenset({("a", "c"), ("b", "c"), ("b", "d")})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _mixture() -> TrueMixture:
    return TrueMixture(
        trees=[
            series("a", parallel("b", "c"), "d"),
            parallel("a", series("b", "c", "d")),
        ],
        weights=[0.6, 0.4],
    )


def _random_log(seed: int, eps_sim: float = 0.1, eta_sim: float = 0.15,
                G: int = 12, n_g: int = 3) -> GroupedLog:
    groups, _ = sample_grouped_log(
        _mixture(), G=G, n_g=n_g, seed=seed, eps_sim=eps_sim, eta_sim=eta_sim
    )
    return GroupedLog(groups)


def _basic_log() -> GroupedLog:
    return GroupedLog([
        [("a", "b", "c", "d"), ("a", "c", "b", "d")],
        [("b", "a", "d", "c")],
    ])


def _timed_log() -> TimedGroupedLog:
    return TimedGroupedLog([
        [(("a", "b", "c", "d"), (0.5, 0.3, 0.2, 0.4)),
         (("a", "c", "b", "d"), (0.4, 0.1, 0.6, 0.2))],
        [(("b", "a", "d", "c"), (0.1, 0.2, 0.3, 0.4))],
    ])


def _logsumexp(a: np.ndarray, axis: int = 0) -> np.ndarray:
    mx = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(mx, axis=axis) + np.log(np.exp(a - mx).sum(axis=axis))


def _n_extension_traces():
    """All linear extensions of the N poset, by brute force over S_4."""
    return [p for p in permutations(ELS) if respects(p, REL_N)]


def _n_lattice_log() -> GroupedLog:
    """Three groups: one whose meet is exactly the N (non-SP), one whose meet
    is the SP order (a || b) -> c -> d, one a single chain."""
    exts = _n_extension_traces()
    return GroupedLog([
        exts,                                            # meet = the N
        [("a", "b", "c", "d"), ("b", "a", "c", "d")],    # meet = (a||b)->c->d
        [("d", "c", "b", "a")],                          # meet = the chain itself
    ])


# --------------------------------------------------------------------------- #
# regime selection
# --------------------------------------------------------------------------- #

def test_regime_enumeration_when_m_small():
    orc = Oracle(_basic_log(), [0.1], [0.1])  # m=4 <= default max_exact_m=6
    assert orc.kind == "enumeration"
    assert orc.exact is True


def test_regime_forced_meet_closure():
    orc = Oracle(_basic_log(), [0.1], [0.1], max_exact_m=0)
    assert orc.kind == "meet-closure"
    assert orc.exact is True


def test_regime_swap_kernel_falls_to_heuristic():
    orc = Oracle(_basic_log(), [0.1], [0.1], max_exact_m=0, noise_kernel="swap")
    assert orc.kind == "lattice-heuristic"
    assert orc.exact is False


def test_regime_sp_class_falls_to_heuristic():
    orc = Oracle(_basic_log(), [0.1], [0.1], max_exact_m=0, poset_class="sp")
    assert orc.kind == "lattice-heuristic"
    assert orc.exact is False


def test_regime_timed_log_falls_to_heuristic():
    orc = Oracle(_timed_log(), [0.1], [0.1], max_exact_m=0)
    assert orc.kind == "lattice-heuristic"
    assert orc.exact is False


def test_regime_closure_cap_downgrades_to_heuristic():
    # 3 distinct chains; any fresh pairwise meet pushes past cap=3.
    log = GroupedLog([
        [("a", "b", "c", "d")], [("b", "a", "c", "d")], [("d", "c", "b", "a")],
    ])
    capped = Oracle(log, [0.1], [0.1], max_exact_m=0, closure_cap=3)
    assert capped.kind == "lattice-heuristic"
    assert capped.exact is False
    # same log without the cap is genuinely meet-closure
    free = Oracle(log, [0.1], [0.1], max_exact_m=0)
    assert free.kind == "meet-closure"
    assert free.exact is True


# --------------------------------------------------------------------------- #
# THE REDUCTION THEOREM: meet-closure pricing == enumeration pricing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "seed,eps_sim,eta_sim",
    [(0, 0.0, 0.0), (1, 0.15, 0.2), (2, 0.3, 0.1)],
)
def test_reduction_theorem_meet_closure_is_exact(seed, eps_sim, eta_sim):
    log = _random_log(seed, eps_sim=eps_sim, eta_sim=eta_sim)
    assert log.m == 4

    eps_grid = (0.0, 0.05, 0.25)   # mixed grid: includes the noiseless corner
    eta_grid = (0.0, 0.15)
    enum = Oracle(log, eps_grid, eta_grid)                # full 219-poset class
    mc = Oracle(log, eps_grid, eta_grid, max_exact_m=0)   # data-dependent lattice
    assert enum.kind == "enumeration" and enum.exact
    assert mc.kind == "meet-closure" and mc.exact

    enum_rels = {a.rel for a in enum.atoms}
    mc_rels = {a.rel for a in mc.atoms}
    assert len(enum_rels) == 219          # OEIS A001035, m=4
    assert mc_rels <= enum_rels           # candidates live inside the class
    assert len(mc_rels) >= 3

    rng = np.random.default_rng(1000 + seed)
    finite = enum.logF[np.isfinite(enum.logF).all(axis=1)]
    assert len(finite) > 5
    for j in range(20):
        if j % 2 == 0:
            # realistic case: log-density of a random 5-atom mixture
            sel = rng.choice(len(finite), size=5, replace=False)
            w = rng.dirichlet(np.ones(5))
            log_d = _logsumexp(np.log(w)[:, None] + finite[sel], axis=0)
        else:
            # arbitrary positive densities in a sane range
            log_d = rng.uniform(-14.0, -2.0, size=log.G)
        _, s_enum = enum.price(log_d)
        _, s_mc = mc.price(log_d)
        # the meet-closure atoms are a subset of the enumeration atoms, so
        # its max can never exceed the enumeration max ...
        assert s_mc <= s_enum
        # ... and by the reduction theorem it attains it exactly.
        assert math.isclose(s_enum, s_mc, rel_tol=1e-12), (
            f"log_d #{j}: enumeration score {s_enum!r} != meet-closure {s_mc!r}"
        )


# --------------------------------------------------------------------------- #
# enumeration atom counts
# --------------------------------------------------------------------------- #

def test_enumeration_atom_count_general():
    eps_grid, eta_grid, lam_grid = [0.0, 0.1, 0.2], [0.0, 0.05], [1.0, 2.0]
    n_grid = len(eps_grid) * len(eta_grid) * len(lam_grid)

    # m = 4: 219 labeled posets (OEIS A001035); nothing skipped
    log4 = _basic_log()
    orc4 = Oracle(log4, eps_grid, eta_grid, lam_grid=lam_grid)
    assert orc4.kind == "enumeration"
    assert len(orc4.atoms) == 219 * n_grid
    assert {a.rel for a in orc4.atoms} == set(enumerate_posets(log4.alphabet))
    # every class member gets exactly one atom per grid point
    per_rel = Counter(a.rel for a in orc4.atoms)
    assert set(per_rel.values()) == {n_grid}
    assert orc4.logF.shape == (219 * n_grid, log4.G)

    # m = 3: 19 labeled posets (OEIS A001035)
    log3 = GroupedLog([[("a", "b", "c")], [("c", "b", "a")]])
    orc3 = Oracle(log3, eps_grid, eta_grid, lam_grid=lam_grid)
    assert orc3.kind == "enumeration"
    assert len(orc3.atoms) == 19 * n_grid


def test_enumeration_atom_count_sp():
    eps_grid, eta_grid, lam_grid = [0.0, 0.1, 0.2], [0.0, 0.05], [1.0, 2.0]
    n_grid = len(eps_grid) * len(eta_grid) * len(lam_grid)

    log = _basic_log()
    orc = Oracle(log, eps_grid, eta_grid, poset_class="sp", lam_grid=lam_grid)
    assert orc.kind == "enumeration"
    assert len(orc.atoms) == 195 * n_grid

    # brute-force cross-check: the SP posets on 4 labeled elements are exactly
    # the general posets minus the 24 labeled copies of the N (the unique
    # 4-element non-SP order, which has trivial automorphism group).
    els = frozenset(log.alphabet)
    all_posets = enumerate_posets(log.alphabet)
    brute_sp = {r for r in all_posets if decompose(els, r) is not None}
    assert len(all_posets) == 219
    assert len(brute_sp) == 195
    assert len(all_posets) - len(brute_sp) == 24
    assert {a.rel for a in orc.atoms} == brute_sp
    per_rel = Counter(a.rel for a in orc.atoms)
    assert set(per_rel.values()) == {n_grid}


# --------------------------------------------------------------------------- #
# lattice-heuristic candidates
# --------------------------------------------------------------------------- #

def test_n_poset_construction_is_what_we_think():
    exts = _n_extension_traces()
    assert len(exts) == 5                                   # e(N) = 5
    assert count_linear_extensions(frozenset(ELS), REL_N) == 5
    # Szpilrajn: the meet of all linear extensions recovers the order itself
    assert meet(*(rel_from_trace(t) for t in exts)) == REL_N
    # and the N really is outside the SP class
    assert decompose(frozenset(ELS), REL_N) is None


def test_lattice_candidates_contain_chains_and_group_meets():
    log = _n_lattice_log()
    # swap kernel forces the lattice-heuristic regime for the general class
    orc = Oracle(log, [0.1], [0.05], max_exact_m=0, noise_kernel="swap")
    assert orc.kind == "lattice-heuristic"
    rels = {a.rel for a in orc.atoms}

    chains = {rel_from_trace(t) for t in log.traces}
    meets = {meet(*(rel_from_trace(t) for t in g)) for g in log.groups}
    assert REL_N in meets                # the constructed non-SP meet is there
    assert chains <= rels                # every observed chain is a candidate
    assert meets <= rels                 # every group meet is a candidate


def test_lattice_candidates_sp_class_drops_exactly_non_sp():
    log = _n_lattice_log()
    gen = Oracle(log, [0.1], [0.05], max_exact_m=0, noise_kernel="swap")
    sp = Oracle(log, [0.1], [0.05], max_exact_m=0, poset_class="sp")
    assert sp.kind == "lattice-heuristic"

    els = frozenset(log.alphabet)
    gen_rels = {a.rel for a in gen.atoms}     # the full candidate list
    sp_rels = {a.rel for a in sp.atoms}

    # the non-SP meet (the N) is dropped, not repaired
    assert REL_N in gen_rels
    assert REL_N not in sp_rels
    # observed chains are total orders, hence SP: all kept
    chains = {rel_from_trace(t) for t in log.traces}
    assert chains <= sp_rels
    # SP group meets are kept
    sp_meet = meet(rel_from_trace(("a", "b", "c", "d")),
                   rel_from_trace(("b", "a", "c", "d")))
    assert decompose(els, sp_meet) is not None
    assert sp_meet in sp_rels
    # skipped == exactly the non-SP candidates (same candidate generator)
    assert sp_rels == {r for r in gen_rels if decompose(els, r) is not None}


def test_lattice_candidates_timed_log():
    gaps = (0.3, 0.2, 0.5, 0.1)
    log = _n_lattice_log()
    tlog = TimedGroupedLog([[(t, gaps) for t in g] for g in log.groups])
    orc = Oracle(tlog, [0.1], [0.05], max_exact_m=0)
    assert orc.kind == "lattice-heuristic"
    rels = {a.rel for a in orc.atoms}
    chains = {rel_from_trace(t) for t in tlog.traces}
    meets = {meet(*(rel_from_trace(t) for t in g)) for g in log.groups}
    assert chains <= rels
    assert meets <= rels
    assert REL_N in rels                 # general class keeps the N


# --------------------------------------------------------------------------- #
# price() consistency with logF
# --------------------------------------------------------------------------- #

def test_price_matches_numpy_recomputation():
    log = _random_log(3)
    orc = Oracle(log, [0.0, 0.1], [0.0, 0.1])
    rng = np.random.default_rng(42)
    for _ in range(5):
        log_d = rng.uniform(-12.0, -2.0, size=log.G)
        k, s = orc.price(log_d)
        # independent linear-space reference (ratios are small enough here);
        # price computes the same quantity by log-sum-exp, so agreement is
        # up to float association order, not bit-exact
        scores = np.exp(orc.logF - log_d[None, :]).mean(axis=1)
        assert 0 <= k < len(orc.atoms)
        assert k == int(np.argmax(scores))
        assert math.isclose(s, scores[k], rel_tol=1e-10, abs_tol=0)
        assert np.all(scores <= s * (1 + 1e-10))  # returned score is the max


# --------------------------------------------------------------------------- #
# meet-closure candidate list contains the empty order
# --------------------------------------------------------------------------- #

def test_meet_closure_contains_empty_order():
    # every trace starts with 'a', so every chain (and every meet of chains)
    # contains (a, x) pairs: the raw closure cannot contain the empty order
    log = GroupedLog([
        [("a", "b", "c", "d"), ("a", "c", "b", "d")],
        [("a", "b", "d", "c")],
        [("a", "d", "c", "b")],
    ])
    raw, hit_cap = meet_closure({rel_from_trace(t) for t in log.traces})
    assert not hit_cap
    assert frozenset() not in raw        # the oracle must add it itself

    orc = Oracle(log, [0.0, 0.1], [0.0], max_exact_m=0)
    assert orc.kind == "meet-closure"
    assert frozenset() in {a.rel for a in orc.atoms}
