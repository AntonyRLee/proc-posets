"""Tests for procposets.npmle: corrective step, fit(), certificate,
poset_marginal, refit_weights, polish_nuisances, trivial_chain_loglik.

All randomness is seeded; every fit used by several tests is a module-scoped
fixture.  Oracle-style cross-checks used throughout:

* the 2x2 fully-corrective problem has a calculus closed form (w* = 15/16),
  double-checked by dense grid search;
* e(N poset) is brute-forced over itertools.permutations;
* the duality-gap certificate is re-verified from scratch against *every*
  atom of a freshly rebuilt Oracle;
* refit_weights optimality is attacked with random simplex perturbations;
* trivial_chain_loglik on singleton groups equals the empirical entropy
  sum_t n_t log(n_t / N) analytically, and is unbeatable by any mixture.
"""

from __future__ import annotations

import math
import random
from itertools import permutations

import numpy as np
import pytest

from procposets import (
    GroupedLog,
    TrueMixture,
    fit,
    group_by_key,
    parallel,
    polish_nuisances,
    refit_weights,
    sample_grouped_log,
    sample_keyed_log,
    series,
    tree_relations,
    trivial_chain_loglik,
)
from procposets.npmle import _fully_corrective, _mixture_logd
from procposets.oracle import Oracle
from procposets.rel import is_partial_order, respects, sample_linear_extension

# ---------------------------------------------------------------------------
# Shared ground truth (the adversarial demo mixture: P2 is an extension of P1)
# ---------------------------------------------------------------------------

P1 = series("a", parallel("b", "c"), "d")  # a -> (b || c) -> d, e = 2
P2 = series("a", "c", "b", "d")            # a -> c -> b -> d,  e = 1
TRUTH = TrueMixture(trees=[P1, P2], weights=[0.6, 0.4])
R1, R2 = tree_relations(P1), tree_relations(P2)

CHAIN_TRUTH = TrueMixture(trees=[series("a", "b", "c", "d")], weights=[1.0])

# the N poset: a < c, b < c, b < d (not series-parallel)
REL_N = frozenset({("a", "c"), ("b", "c"), ("b", "d")})
ELS4 = frozenset("abcd")


def _assert_simplex(w: np.ndarray):
    assert np.all(w >= 0.0)
    assert math.isclose(float(w.sum()), 1.0, rel_tol=0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# _mixture_logd
# ---------------------------------------------------------------------------

def test_mixture_logd_matches_bruteforce():
    rng = np.random.default_rng(0)
    logF = rng.normal(size=(5, 7)) * 3.0
    w = rng.dirichlet(np.ones(5))
    got = _mixture_logd(logF, w)
    naive = np.log(np.exp(logF).T @ w)  # safe at this scale
    assert np.allclose(got, naive, rtol=0, atol=1e-12)


def test_mixture_logd_shift_invariant_under_underflow():
    # naive exp would underflow at logF - 900; the stable version must just
    # translate: logd(logF + c, w) == logd(logF, w) + c.
    rng = np.random.default_rng(1)
    logF = rng.normal(size=(4, 6))
    w = rng.dirichlet(np.ones(4))
    base = _mixture_logd(logF, w)
    shifted = _mixture_logd(logF - 900.0, w)
    assert np.all(np.isfinite(shifted))
    assert np.allclose(shifted, base - 900.0, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# _fully_corrective
# ---------------------------------------------------------------------------

# Hand-made 2 atoms x 2 groups problem.  With F = [[.9, .3], [.1, .6]] the
# objective ll(u) = log(.1 + .8u) + log(.6 - .3u) for weight u on atom 0 has
# ll'(u) = 0  <=>  .8(.6 - .3u) = .3(.1 + .8u)  <=>  u* = 0.45/0.48 = 0.9375.
LOGF_2X2 = np.log(np.array([[0.9, 0.3], [0.1, 0.6]]))


def _ll_2x2(u: float) -> float:
    return math.log(0.1 + 0.8 * u) + math.log(0.6 - 0.3 * u)


def test_fully_corrective_reaches_analytic_optimum():
    w = _fully_corrective(LOGF_2X2, np.array([0.5, 0.5]))
    _assert_simplex(w)
    assert abs(w[0] - 0.9375) < 1e-5

    # dense grid search cross-check: nothing on the simplex does better
    grid = np.linspace(0.0, 1.0, 100_001)
    grid_best = np.max(np.log(0.1 + 0.8 * grid) + np.log(0.6 - 0.3 * grid))
    ll = float(_mixture_logd(LOGF_2X2, w).sum())
    assert ll >= grid_best - 1e-9
    # and the analytic optimum is not exceeded either (sanity of the oracle)
    assert ll <= _ll_2x2(0.9375) + 1e-12


def test_fully_corrective_monotone_iterates():
    # step the multiplicative update one iteration at a time (iters=1 performs
    # exactly one update) and track the loglik manually: must never decrease.
    w = np.array([0.999, 0.001])
    prev = -np.inf
    for _ in range(80):
        ll = float(_mixture_logd(LOGF_2X2, w).sum())
        assert ll >= prev - 1e-12
        prev = ll
        w = _fully_corrective(LOGF_2X2, w, iters=1)
        _assert_simplex(w)
    # after 80 monotone steps we must be above the starting loglik and moving
    # toward the analytic optimum
    assert prev > _ll_2x2(0.999)
    assert prev <= _ll_2x2(0.9375) + 1e-12


# ---------------------------------------------------------------------------
# fit() -- clean single component (the N poset exercises the general class)
# ---------------------------------------------------------------------------

def test_fit_single_component_N_poset():
    # oracle check on the ground truth itself: e(N) by brute force
    e_brute = sum(1 for p in permutations(sorted(ELS4)) if respects(p, REL_N))
    assert e_brute == 5
    assert is_partial_order(ELS4, REL_N)

    rng = random.Random(3)
    groups = [
        [sample_linear_extension(ELS4, REL_N, rng) for _ in range(4)]
        for _ in range(200)
    ]
    res = fit(GroupedLog(groups))

    assert res.oracle_kind == "enumeration"
    assert res.exact_oracle
    assert res.gap <= 1e-4  # default gap_tol
    _assert_simplex(res.weights)
    # weights come back sorted descending
    assert np.all(np.diff(res.weights) <= 0)

    marg = res.poset_marginal()
    rel_top, _, w_top, eps_top, eta_top, _ = marg[0]
    assert rel_top == REL_N
    assert w_top >= 0.9  # ~= 0.97 at this seed
    assert eps_top == 0.0 and eta_top == 0.0

    # the fitted N atom carries the brute-forced extension count
    (atom_n,) = [a for a in res.atoms if a.rel == REL_N]
    assert atom_n.e == e_brute
    # every fitted atom is a genuine partial order
    assert all(is_partial_order(ELS4, a.rel) for a in res.atoms)


# ---------------------------------------------------------------------------
# fit() -- adversarial two-component demo truth
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_comp():
    groups, _ = sample_grouped_log(TRUTH, G=300, n_g=5, seed=0)
    log = GroupedLog(groups)
    return log, fit(log)


def test_fit_two_component_recovery(two_comp):
    log, res = two_comp
    assert res.oracle_kind == "enumeration"
    assert res.gap <= 1e-4
    _assert_simplex(res.weights)

    marg = {rel: w for rel, _, w, _, _, _ in res.poset_marginal()}
    assert R1 in marg and R2 in marg  # both rels recovered exactly
    assert abs(marg[R1] - 0.6) <= 0.08
    assert abs(marg[R2] - 0.4) <= 0.08
    assert math.isclose(sum(marg.values()), 1.0, rel_tol=0, abs_tol=1e-9)

    # history: (loglik, gap) per iteration; loglik non-decreasing, and the
    # reported (loglik, gap) are the last history entry
    lls = [ll for ll, _ in res.history]
    assert all(b >= a - 1e-6 for a, b in zip(lls, lls[1:]))
    assert res.history[-1] == (res.loglik, res.gap)
    assert len(res.history) == res.iterations


def test_certificate_holds_over_all_oracle_atoms(two_comp):
    # Rebuild an oracle with the same arguments fit() used and re-score every
    # atom of the class at the fitted mixture: none may beat 1 + gap.
    log, res = two_comp
    oracle = Oracle(log, (0.0,), (0.0,), max_exact_m=6,
                    noise_kernel="uniform", poset_class="general",
                    lam_grid=(1.0,))
    logF_fit = np.stack([log.group_logf(a) for a in res.atoms])
    log_d = _mixture_logd(logF_fit, res.weights)
    # the fitted loglik is reproducible from the returned atoms/weights
    assert math.isclose(float(log_d.sum()), res.loglik, rel_tol=0, abs_tol=1e-6)
    scores = np.exp(np.clip(oracle.logF - log_d[None, :], -700, 700)).mean(axis=1)
    assert float(scores.max()) - 1.0 <= res.gap + 1e-9


# ---------------------------------------------------------------------------
# poset_marginal / marginal_summary
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def swap_noisy_fit():
    # 8% adjacent-swap recording noise + the mechanistic swap kernel on a
    # coarse eps grid {0, 0.2}: the NPMLE brackets the true rate by splitting
    # mass across grid points of the *same* poset, which is exactly what
    # poset_marginal must aggregate.
    groups, _ = sample_grouped_log(TRUTH, G=150, n_g=5, seed=3, eps_sim=0.08)
    log = GroupedLog(groups)
    res = fit(log, eps_grid=(0.0, 0.2), eta_grid=(0.0,), noise_kernel="swap")
    return log, res


def test_poset_marginal_aggregates_grid_mass(swap_noisy_fit):
    _, res = swap_noisy_fit
    marg = res.poset_marginal()

    # grid mass actually split: more atoms than distinct rels
    rels = [rel for rel, *_ in marg]
    assert len(rels) == len(set(rels))          # one entry per rel
    assert len(res.atoms) > len(rels)           # aggregation happened
    assert np.all(np.diff([w for _, _, w, _, _, _ in marg]) <= 0)  # sorted

    # entries match a hand aggregation of (atoms, weights)
    for rel, desc, w, mean_eps, mean_eta, mean_lam in marg:
        idx = [i for i, a in enumerate(res.atoms) if a.rel == rel]
        wsum = float(sum(res.weights[i] for i in idx))
        assert math.isclose(w, wsum, rel_tol=0, abs_tol=1e-12)
        exp_eps = float(sum(res.weights[i] * res.atoms[i].eps for i in idx)) / wsum
        exp_eta = float(sum(res.weights[i] * res.atoms[i].eta for i in idx)) / wsum
        assert math.isclose(mean_eps, exp_eps, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(mean_eta, exp_eta, rel_tol=0, abs_tol=1e-12)
        # mean nuisances live in the convex hull of the grid
        assert 0.0 <= mean_eps <= 0.2
        assert mean_eta == 0.0
        assert mean_lam == 1.0  # untimed: every atom carries lam = 1
    assert math.isclose(sum(w for _, _, w, _, _, _ in marg), 1.0,
                        rel_tol=0, abs_tol=1e-9)

    # the dominant rels bracket the noise rate strictly inside the grid hull
    top_rels = {rel for rel, _, w, _, _, _ in marg if w > 0.2}
    assert R1 in top_rels and R2 in top_rels
    for rel, _, w, mean_eps, _, _ in marg:
        if rel in (R1, R2):
            assert 0.0 < mean_eps < 0.2


def test_marginal_summary_smoke(swap_noisy_fit):
    _, res = swap_noisy_fit
    s = res.marginal_summary()
    assert s.startswith("poset marginal of the mixing measure")
    assert "w = " in s and "mean eps" in s
    # one line per above-threshold marginal entry + the header
    n_shown = sum(1 for _, _, w, _, _, _ in res.poset_marginal() if w >= 1e-3)
    assert len(s.splitlines()) == 1 + n_shown


# ---------------------------------------------------------------------------
# refit_weights
# ---------------------------------------------------------------------------

def test_refit_weights_is_fixed_support_optimum(capsys):
    rows = sample_keyed_log(TRUTH, n_blocks=60, traces_per_block=(2, 6),
                            defect_prob=0.20, seed=5, eps_sim=0.03)
    groups, leftovers = group_by_key(
        rows, declared_assumption="test: same-day blocks share a variant")
    assert leftovers, "calibration requires undersized blocks"
    log = GroupedLog(groups)
    res = fit(log, eps_grid=(0.0, 0.05), eta_grid=(0.0, 0.2))

    res2, gap2 = refit_weights(res, log.groups, leftovers)
    _assert_simplex(res2.weights)
    assert res2.gap == gap2
    assert -1e-9 <= gap2 <= 1e-5  # restricted program solved to near-zero gap
    assert set(res2.atoms) == set(res.atoms)  # support frozen

    # the reported loglik is reproducible on the enlarged group set (under
    # the frozen stage-1 profile: refit_weights keeps the stage-1 pbar, so
    # the re-profiled marginal of a fresh GroupedLog(combined) need not
    # reproduce it)...
    combined = list(log.groups) + list(leftovers)
    log_all = GroupedLog(combined)
    pbar = np.zeros(log_all.T)
    for t, i in log.tidx.items():
        pbar[log_all.tidx[t]] = log.pbar[i]
    log_all.pbar = pbar
    logF = np.stack([log_all.group_logf(a) for a in res2.atoms])
    assert math.isclose(float(_mixture_logd(logF, res2.weights).sum()),
                        res2.loglik, rel_tol=0, abs_tol=1e-9)

    # ...and is the optimum of the fixed-support convex problem: random
    # multiplicative simplex perturbations at four scales never beat it.
    rng = np.random.default_rng(42)
    K = len(res2.weights)
    for scale in (1e-3, 1e-2, 1e-1, 0.5):
        for _ in range(100):
            wp = res2.weights * np.exp(scale * rng.standard_normal(K))
            wp /= wp.sum()
            ll_p = float(_mixture_logd(logF, wp).sum())
            assert ll_p <= res2.loglik + 1e-7


# ---------------------------------------------------------------------------
# polish_nuisances
# ---------------------------------------------------------------------------

def test_polish_nuisances_improves_loglik_and_eps():
    # Chain truth makes the swap-kernel model *exactly* well-specified: every
    # adjacent transposition of a chain trace leaves L(P), so the effective
    # recording-noise rate equals eps_sim = 0.08 -- an unambiguous truth for
    # the fitted eps to approach once polishing frees it from the {0, 0.2}
    # grid.
    groups, _ = sample_grouped_log(CHAIN_TRUTH, G=150, n_g=5, seed=5,
                                   eps_sim=0.08)
    log = GroupedLog(groups)
    res = fit(log, eps_grid=(0.0, 0.2), eta_grid=(0.0,), noise_kernel="swap")
    pol = polish_nuisances(res, log, rounds=1)

    # monotone in the likelihood
    assert pol.loglik >= res.loglik - 1e-9
    _assert_simplex(pol.weights)
    # polish moves nuisances only: the poset support is unchanged
    assert sorted(map(sorted, (a.rel for a in pol.atoms))) == \
        sorted(map(sorted, (a.rel for a in res.atoms)))
    assert all(0.0 <= a.eps <= 0.5 for a in pol.atoms)

    # the polished mean eps is closer to the true rate than the grid fit's
    eps_grid_fit = float(np.dot(res.weights, [a.eps for a in res.atoms]))
    eps_polished = float(np.dot(pol.weights, [a.eps for a in pol.atoms]))
    assert abs(eps_polished - 0.08) < abs(eps_grid_fit - 0.08)
    # sharper: with the kernel exactly well-specified, the MLE tracks the
    # *realized* corruption fraction of this finite sample (0.0973 at seed 5;
    # polished ~0.101, vs ~0.077 when the eps line search ran under a stale
    # uniform tag -- DESIGN_REVIEW W20)
    clean = ("a", "b", "c", "d")
    traces = [t for g in groups for t in g]
    realized = sum(1 for t in traces if t != clean) / len(traces)
    assert abs(eps_polished - realized) < 0.01
    assert pol.oracle_kind.endswith("+ polished nuisances")


# ---------------------------------------------------------------------------
# trivial_chain_loglik
# ---------------------------------------------------------------------------

def test_trivial_chains_lose_on_grouped_signal(two_comp):
    log, res = two_comp
    triv = trivial_chain_loglik(log, eps=0.01)
    assert np.isfinite(triv)
    # groups carry real co-membership signal: the mixture wins by a lot
    assert res.loglik - triv > 100.0


def test_trivial_chains_unbeatable_on_singletons():
    groups, _ = sample_grouped_log(TRUTH, G=200, n_g=1, seed=9)
    log = GroupedLog(groups)  # 200 singleton groups
    assert set(log.n_g) == {1.0}

    triv = trivial_chain_loglik(log, eps=0.0)
    # analytic identity: with singleton groups the trivial mixture density of
    # a trace is its empirical frequency, so its loglik is the (negated,
    # unnormalised) empirical entropy sum_t n_t log(n_t / N).
    counts = log.counts.sum(axis=0)
    n_tot = counts.sum()
    entropy = float(sum(c * math.log(c / n_tot) for c in counts if c > 0))
    assert math.isclose(triv, entropy, rel_tol=0, abs_tol=1e-9)

    # singleton groups carry no grouping information: no mixture over posets
    # can beat the empirical trace distribution
    res = fit(log)
    assert res.loglik <= triv + 1e-6


# ---------------------------------------------------------------------------
# audited-defect guards (DESIGN_REVIEW W9, W15, W16, W19, W20)
# ---------------------------------------------------------------------------

def test_weight_floor_at_or_above_entering_weight_rejected():
    # W16: a floor >= the 1e-3 entering weight prunes every new atom before
    # its corrective step -- silent spin to max_iters.  Must refuse loudly.
    groups, _ = sample_grouped_log(TRUTH, G=10, n_g=3, seed=0)
    log = GroupedLog(groups)
    for floor in (1e-3, 5e-3, 1.0, -1e-9):
        with pytest.raises(ValueError, match="weight_floor"):
            fit(log, weight_floor=floor)


def test_max_iters_exit_returns_certified_mixture():
    # W15: on the max_iters exit the just-appended atom used to be returned
    # at its uncorrected 1e-3 weight while loglik/gap were computed
    # pre-append.  The returned mixture must reproduce its reported loglik.
    groups, _ = sample_grouped_log(TRUTH, G=40, n_g=4, seed=1, eps_sim=0.05)
    log = GroupedLog(groups)
    res = fit(log, eps_grid=(0.0, 0.1), eta_grid=(0.0,), max_iters=1)
    assert res.iterations == 1
    assert res.gap > 1e-4  # genuinely unconverged: an append did happen
    logF = np.stack([log.group_logf(a) for a in res.atoms])
    ll = float(_mixture_logd(logF, res.weights).sum())
    assert math.isclose(ll, res.loglik, rel_tol=0, abs_tol=1e-9)


def test_refit_freezes_stage1_profile():
    # W9.1: "components frozen" means frozen in *density*.  A pure-interloper
    # atom (eta = 1) has density pbar(t); after refit with leftovers that
    # shift the empirical marginal, its density must still use the stage-1
    # pbar = (5/6, 1/6), not the re-profiled (5/8, 3/8).
    from procposets.likelihood import make_atom
    from procposets.npmle import FitResult

    ab, ba = ("a", "b"), ("b", "a")
    groups = [[ab, ab, ab], [ab, ab, ba]]
    leftovers = [[ba], [ba]]
    atom = make_atom(frozenset("ab"), frozenset(), 0.0, 1.0)
    res = FitResult(atoms=[atom], weights=np.ones(1), loglik=0.0, gap=0.0,
                    exact_oracle=True, oracle_kind="test", iterations=0)
    res2, _ = refit_weights(res, groups, leftovers)
    frozen = 5 * math.log(5 / 6) + 3 * math.log(1 / 6)
    reprofiled = 5 * math.log(5 / 8) + 3 * math.log(3 / 8)
    assert math.isclose(res2.loglik, frozen, rel_tol=0, abs_tol=1e-12)
    assert abs(res2.loglik - reprofiled) > 0.1


def test_refit_rejects_unexplainable_leftovers():
    # W9.2: a leftover group no frozen atom explains used to poison the refit
    # into loglik = gap = NaN with only a RuntimeWarning.  Must raise, naming
    # the offending group.
    groups, _ = sample_grouped_log(CHAIN_TRUTH, G=12, n_g=3, seed=0)
    log = GroupedLog(groups)
    res = fit(log)  # default grids: eps = eta = 0, support = the chain
    bad = [[("d", "c", "b", "a")]]
    with pytest.raises(ValueError, match="zero density"):
        refit_weights(res, log.groups, bad)


def test_lattice_candidates_include_empty_order():
    # W9.2: the heuristic-lattice regime must contain the (full-support)
    # empty order, so no group can be priced at zero by every candidate --
    # the path by which fit() itself returned NaN.
    groups, _ = sample_grouped_log(TRUTH, G=8, n_g=3, seed=2)
    log = GroupedLog(groups)
    oracle = Oracle(log, (0.0,), (0.0,), max_exact_m=2, noise_kernel="swap")
    assert oracle.kind == "lattice-heuristic"
    assert frozenset() in {a.rel for a in oracle.atoms}
    assert not np.isneginf(oracle.logF).all(axis=0).any()


def test_unknown_noise_kernel_rejected():
    # W19: "Swap" used to reproduce the uniform-kernel fit bit-for-bit with
    # no notice -- a typo silently changed declared choice N.
    from procposets.likelihood import make_atom

    groups, _ = sample_grouped_log(TRUTH, G=6, n_g=3, seed=0)
    log = GroupedLog(groups)
    with pytest.raises(ValueError, match="noise kernel"):
        fit(log, noise_kernel="Swap")
    # trace_p's dispatch is exhaustive (eps > 0 so the kernel is consulted)
    a = make_atom(frozenset("abcd"), frozenset(), 0.1, 0.0, noise="bogus")
    with pytest.raises(ValueError, match="noise kernel"):
        log.trace_p(a)


def test_declared_kernel_travels_on_eps0_atoms(swap_noisy_fit):
    # W20: eps = 0 atoms of a declared-swap fit carry the swap tag (density-
    # neutral at eps = 0), so a later eps line search -- polish_nuisances --
    # prices under the declared kernel instead of a stale "uniform" tag.
    _, res = swap_noisy_fit
    assert all(a.noise == "swap" for a in res.atoms)


def test_timed_swap_rejected_at_oracle_boundary():
    # W20/W19 companion: timed + swap is a declared non-combination (README
    # row T); it must be refused at fit() time, not depend on which atoms
    # happen to get evaluated.
    from procposets.likelihood import TimedGroupedLog
    from procposets.simulate import sample_timed_grouped_log

    tgroups, _ = sample_timed_grouped_log(TRUTH, G=4, n_g=3,
                                          lams=[1.0, 2.0], seed=0)
    tlog = TimedGroupedLog(tgroups)
    with pytest.raises(ValueError, match="uniform eps kernel"):
        fit(tlog, noise_kernel="swap")


# ---------------------------------------------------------------------------
# audited-defect guards, round 2 (DESIGN_REVIEW W5, W9.3, W12.1, W14, W17, W18)
# ---------------------------------------------------------------------------

def test_price_orders_exactly_in_log_space():
    # W17: exp(clip(.., -700, 700)).mean() collapsed atoms with per-group
    # ratios beyond the clip into exact ties (argmax then picked by index).
    # Row 0 truly dominates; the clipped version scored row 1 higher.
    o = object.__new__(Oracle)
    o.logF = np.array([[800.0, -800.0], [750.0, 750.0]])
    k, score = o.price(np.zeros(2))
    assert k == 0
    assert score > 1.0

    # an all-(-inf) row must price to score 0, never NaN
    o.logF = np.array([[-np.inf, -np.inf], [0.0, 0.0]])
    k, score = o.price(np.zeros(2))
    assert k == 1
    assert math.isclose(score, 1.0, rel_tol=0, abs_tol=1e-12)


def test_converged_flag_and_summary_warning(two_comp):
    # W14/W15: a certified fit says so; a max_iters stop is flagged loudly.
    _, res = two_comp
    assert res.converged
    assert "WARNING" not in res.summary()

    groups, _ = sample_grouped_log(TRUTH, G=40, n_g=4, seed=1, eps_sim=0.05)
    log = GroupedLog(groups)
    stopped = fit(log, eps_grid=(0.0, 0.1), eta_grid=(0.0,), max_iters=1)
    assert not stopped.converged
    assert "WARNING" in stopped.summary()


def test_summary_states_nuisance_grids(two_comp):
    # W5: a printed eps = 0 at a one-point grid is a constraint, not an
    # estimate -- the summary must state the grids it selected from.
    _, res = two_comp
    assert "nuisance grids: eps {0}, eta {0}, lam {1}" in res.summary()


def test_polish_moves_lambda_on_timed_logs():
    # W5(a): lam used to stay grid-quantized forever (the polish loop was
    # literally eps-then-eta).  On timed data with a true rate strictly
    # between grid points, polish must move the weighted-mean lam toward it,
    # and poset_marginal must now report that mean.
    from procposets.likelihood import TimedGroupedLog
    from procposets.simulate import sample_timed_grouped_log

    tgroups, _ = sample_timed_grouped_log(CHAIN_TRUTH, G=40, n_g=3,
                                          lams=[2.5], seed=7)
    tlog = TimedGroupedLog(tgroups)
    # one-point lam grid far from the truth: the fit is pinned at lam = 1
    res = fit(tlog, lam_grid=(1.0,))
    assert all(a.lam == 1.0 for a in res.atoms)
    pol = polish_nuisances(res, tlog, rounds=1)
    assert pol.loglik >= res.loglik - 1e-9

    # single chain component, k_j = 1 at every step, so the lam slice of the
    # loglik is N m log(lam) - lam * sum(all gaps): the polished rate must
    # sit at the analytic MLE (golden section is far tighter than this tol)
    n_traces = sum(len(g) for g in tgroups)
    gap_total = sum(x for g in tgroups for (_, gaps) in g for x in gaps)
    lam_mle = 4 * n_traces / gap_total  # ~2.5, the simulated rate
    (top_atom,) = [a for a, w in zip(pol.atoms, pol.weights) if w > 0.5]
    assert abs(top_atom.lam - lam_mle) < 1e-3
    # the marginal reports the weighted-mean rate (6th column)
    _, _, w_top, _, _, mean_lam = pol.poset_marginal()[0]
    assert w_top > 0.5
    assert abs(mean_lam - lam_mle) < 0.1
    # and marginal_summary surfaces it once lam != 1
    assert "mean lam" in pol.marginal_summary()


def test_refit_full_class_gap():
    # W9.3: the fixed-support gap cannot see a component the leftovers
    # wanted; oracle_params runs the full pricing oracle once on the
    # enlarged frozen-profile log and reports that gap alongside.
    groups, _ = sample_grouped_log(CHAIN_TRUTH, G=12, n_g=3, seed=0)
    log = GroupedLog(groups)
    res = fit(log)

    plain, _ = refit_weights(res, log.groups, [[("a", "b", "c", "d")]])
    assert plain.full_class_gap is None

    upgraded, gap = refit_weights(
        res, log.groups, [[("a", "b", "c", "d")]],
        oracle_params=dict(eps_grid=(0.0,), eta_grid=(0.0,)),
    )
    # chain data + chain leftovers: nothing in the class wants in
    assert upgraded.full_class_gap is not None
    assert -1e-9 <= upgraded.full_class_gap <= 1e-2
    assert f"full-class gap" in upgraded.summary()


def test_oracle_downgrades_on_ideal_budget(monkeypatch):
    # W12.1: a meet-closure candidate too wide for the ideal-state budget is
    # skipped and the certificate downgraded loudly -- never a silent hang.
    import procposets.rel as posets_mod

    monkeypatch.setattr(posets_mod, "MAX_IDEAL_STATES", 10)
    t1 = tuple("abcdefg")            # m = 7 > max_exact_m: meet-closure regime
    t2 = tuple(reversed(t1))
    log = GroupedLog([[t1] * 3, [t2] * 3])
    oracle = Oracle(log, (0.0,), (0.0,))
    # the meet of the two chains (the empty order, 2^7 ideals) was skipped
    assert oracle.budget_skipped >= 1
    assert oracle.kind == "lattice-heuristic"
    assert not oracle.exact
    # the chains themselves (bound 8 <= 10) were kept and price finitely
    assert not np.isneginf(oracle.logF).all(axis=0).any()


def test_meet_closure_regime_via_capability_flags():
    # W18: a *fresh* GeneralPosets() instance used to lose the exact
    # meet-closure certificate to a `cls is GENERAL` identity check.
    from procposets.rel import GeneralPosets

    t1 = tuple("abcdefg")
    t2 = ("a", "b", "c", "d", "e", "g", "f")
    log = GroupedLog([[t1] * 3, [t2] * 3])
    oracle = Oracle(log, (0.0,), (0.0,), poset_class=GeneralPosets())
    assert oracle.kind == "meet-closure"
    assert oracle.exact


# ---------------------------------------------------------------------------
# M2 additions: forced regime, identifiability check, weight bootstrap
# ---------------------------------------------------------------------------

def test_force_regime(two_comp):
    log, auto = two_comp
    assert auto.oracle_kind == "enumeration"
    forced = fit(log, force_regime="lattice-heuristic")
    assert forced.oracle_kind == "lattice-heuristic"
    assert not forced.exact_oracle
    # the lattice program is a restriction: it can never beat the exact fit
    assert forced.loglik <= auto.loglik + 1e-6
    with pytest.raises(ValueError, match="force_regime"):
        fit(log, force_regime="enumeration")  # exact regimes are checked,
        #                                       never asserted


def test_identifiability_report(two_comp):
    from procposets.diagnostics import identifiability_report

    log, res = two_comp
    rep = identifiability_report(res, log)
    assert "sigma_min" in rep and "WARNING" not in rep

    # two copies of the same atom: trace laws exactly collinear
    from procposets.npmle import FitResult
    a = res.atoms[0]
    fake = FitResult(atoms=[a, a], weights=np.array([0.5, 0.5]),
                     loglik=0.0, gap=0.0, exact_oracle=True,
                     oracle_kind="test", iterations=0)
    assert "WARNING" in identifiability_report(fake, log)


def test_bootstrap_weights(two_comp):
    from procposets.diagnostics import bootstrap_weights

    log, res = two_comp
    W = bootstrap_weights(res, log, B=50, seed=1)
    assert W.shape == (50, len(res.atoms))
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(W >= 0.0)
    # resampled weights bracket the point estimate for the dominant atom
    lo, hi = np.quantile(W[:, 0], [0.02, 0.98])
    assert lo <= res.weights[0] <= hi
