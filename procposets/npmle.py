"""Nonparametric MLE over the mixing measure, by Frank-Wolfe.

The estimator
-------------
Maximise, over probability measures nu on the atom class
Theta = posets x eps x eta (x lam for timed logs),

    L(nu) = sum_g log integral f_theta(g) dnu(theta).

This is the Kiefer-Wolfowitz NPMLE: L is concave in nu on a convex set, so

* the fitted mixture density is unique (no initialisation, no local optima),
* the optimum is attained by a discrete nu with at most G atoms (Lindsay),
  so the number of components K is *output*, not input, and
* first-order optimality gives a computable certificate:
  sup_theta (1/G) sum_g f_theta(g)/d_g <= 1 + gap.

Algorithm: Frank-Wolfe / column generation with fully-corrective steps.

* Direction finding = the pricing oracle (oracle.py): the one place all
  combinatorial hardness lives.
* Fully-corrective step = re-optimise the weights over the active atoms.
  That subproblem, max_w sum_g log sum_k w_k F[g,k] on the simplex, is
  itself convex; we solve it with the classical multiplicative update
  w_k <- w_k * (1/G) sum_g F[g,k]/d_g, which is mirror descent on this
  objective with a global convergence guarantee.  (It has the *form* of an
  EM weight step, but nothing here is EM in the objectionable sense: no
  latent component structure is re-estimated, the problem is convex, and
  convergence is to the global optimum.)
* Pruning: atoms whose weight falls below a hard floor are removed; because
  the objective is concave this cannot cycle.

Stopping on the duality gap makes the returned object a *certified*
epsilon-optimal mixture over the oracle's atom class.

Post-processing, both convex or monotone and both optional:

* :func:`refit_weights` -- second-stage weight refinement on a larger group
  set (e.g. the undersized blocks that carry no Kruskal identification but
  do carry weight information), components frozen.  Convex again.
* :func:`polish_nuisances` -- continuous coordinate ascent on each atom's
  (eps, eta; lam on timed logs) from its grid value, alternated with
  fully-corrective weight steps.  Monotone in the likelihood; turns the
  nuisance grids from a discretisation *choice* into a mere initialisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .likelihood import Atom, GroupedLog, TimedGroupedLog
from .oracle import Oracle, _log_mean_exp_rows


def _moment_seed_atom(oracle: Oracle, log: GroupedLog, init_order) -> int:
    """Row index into ``oracle.logF`` to seed column generation from, chosen
    by the M9 moment initialiser (see `poset_mixture.initialiser`).  Ranks
    the distinct candidate posets by order-``init_order`` precedence-moment
    alignment to the data, then returns the best-fitting nuisance atom of the
    top-ranked poset.  Warm start only -- the convex optimum is unchanged."""
    from .initialiser import moment_seed

    rels = list(dict.fromkeys(a.rel for a in oracle.atoms))  # distinct, ordered
    traces = [t for g in log.groups for t in g]
    ranking = moment_seed(frozenset(log.alphabet), rels, traces,
                          order=init_order)
    top_rel = rels[ranking.top]
    # among the top poset's nuisance-grid atoms, seed the best-fitting one
    rowsums = oracle.logF.sum(axis=1)
    idxs = [i for i, a in enumerate(oracle.atoms) if a.rel == top_rel]
    return max(idxs, key=lambda i: rowsums[i])


def _grid_str(grid) -> str:
    return "{" + ", ".join(f"{x:g}" for x in grid) + "}"


@dataclass
class FitResult:
    atoms: List[Atom]
    weights: np.ndarray
    loglik: float
    gap: float
    exact_oracle: bool
    oracle_kind: str
    iterations: int
    history: List[Tuple[float, float]] = field(default_factory=list)  # (loglik, gap)
    converged: bool = True          # False: stopped before gap <= gap_tol
    grids: Optional[dict] = None    # the declared (eps, eta, lam) grids
    full_class_gap: Optional[float] = None  # refit_weights full-oracle pass
    init_order: Optional[object] = None  # M9 moment-initialiser order, if used

    def summary(self, min_weight: float = 1e-6) -> str:
        order = np.argsort(-self.weights)
        lines = [
            f"NPMLE fit: {sum(w > min_weight for w in self.weights)} atoms, "
            f"loglik = {self.loglik:.3f}, duality gap = {self.gap:.2e} "
            f"({self.oracle_kind} oracle certificate)"
        ]
        if self.grids is not None:
            # provenance: a printed eps = 0 at a one-point grid is a
            # constraint of the grid, not an estimate
            lines.append(
                f"  nuisance grids: eps {_grid_str(self.grids['eps'])}, "
                f"eta {_grid_str(self.grids['eta'])}, "
                f"lam {_grid_str(self.grids['lam'])}"
            )
        if self.init_order is not None:
            lines.append(
                f"  moment initialiser: order {self.init_order} (M9) -- a "
                f"warm-start/compute dial only; the fitted optimum is "
                f"initialiser-independent"
            )
        if not self.converged:
            lines.append(
                "  WARNING: stopped before reaching gap_tol -- the reported "
                "gap is honest, but the requested tolerance was not certified"
            )
        if self.full_class_gap is not None:
            lines.append(
                f"  full-class gap on the enlarged data = "
                f"{self.full_class_gap:.2e}"
            )
        for i in order:
            if self.weights[i] <= min_weight:
                continue
            lines.append(f"  w = {self.weights[i]:.4f}   {self.atoms[i].describe()}")
        return "\n".join(lines)

    def poset_marginal(self):
        """Marginalise the mixing measure onto the poset coordinate.

        The nuisance coordinates (eps, eta, lam) live on a grid, and the
        NPMLE of a continuous nuisance typically splits mass across the
        bracketing grid points; the object of interest is the poset marginal,
        with the weighted-mean nuisance values as the point estimates.
        Returns [(rel, desc, total_weight, mean_eps, mean_eta, mean_lam)]
        sorted by weight.
        """
        agg = {}
        for a, w in zip(self.atoms, self.weights):
            d, we, wh, wl, ww = agg.get(a.rel, (a.desc, 0.0, 0.0, 0.0, 0.0))
            agg[a.rel] = (d, we + w * a.eps, wh + w * a.eta,
                          wl + w * a.lam, ww + w)
        out = [
            (rel, d, ww, we / ww, wh / ww, wl / ww)
            for rel, (d, we, wh, wl, ww) in agg.items()
            if ww > 0
        ]
        return sorted(out, key=lambda x: -x[2])

    def marginal_summary(self, min_weight: float = 1e-3) -> str:
        lines = [f"poset marginal of the mixing measure (gap = {self.gap:.2e}):"]
        for rel, desc, w, eps, eta, lam in self.poset_marginal():
            if w < min_weight:
                continue
            tag = "" if lam == 1.0 else f", mean lam = {lam:.3g}"
            lines.append(
                f"  w = {w:.4f}   {desc}   [mean eps = {eps:.3f}, "
                f"mean eta = {eta:.3f}{tag}]"
            )
        return "\n".join(lines)


def _mixture_logd(logF_active: np.ndarray, w: np.ndarray) -> np.ndarray:
    """log d_g = log sum_k w_k exp(logF[k, g]), stable."""
    lw = np.log(np.maximum(w, 1e-300))[:, None]
    z = logF_active + lw
    zmax = z.max(axis=0)
    return zmax + np.log(np.exp(z - zmax).sum(axis=0))


# --- Frank-Wolfe budgets (named so their coupling is visible) ---------------
_INNER_ITERS = 3000        # max fully-corrective (inner simplex) iterations
_INNER_TOL = 1e-13         # inner-solve loglik-increment convergence tolerance
_ENTERING_WEIGHT = 1e-3    # weight a newly column-generated atom enters at; the
#                            next corrective step fixes it.  weight_floor must be
#                            below this or an entering atom is pruned before that
#                            step (see the fit() validation).
_REPRICE_ITERS = 20        # inner tightening rounds before re-pricing a repeated
#                            best atom (DESIGN_REVIEW W14)
_REPRICE_GAP_RATIO = 0.1   # tighten the restricted solve to this fraction of gap_tol


def _fully_corrective(logF_active: np.ndarray, w: np.ndarray,
                      iters: int = _INNER_ITERS, tol: float = _INNER_TOL) -> np.ndarray:
    G = logF_active.shape[1]
    prev = -np.inf
    for _ in range(iters):
        log_d = _mixture_logd(logF_active, w)
        ll = float(log_d.sum())
        if ll - prev < tol * max(1.0, abs(ll)):
            break
        prev = ll
        # multiplicative (mirror-descent) update on the simplex
        resp = np.exp(logF_active + np.log(np.maximum(w, 1e-300))[:, None] - log_d[None, :])
        w = resp.sum(axis=1) / G
        w = np.maximum(w, 0.0)
        w /= w.sum()
    return w


def _tighten_restricted(oracle, logF_active: np.ndarray, w: np.ndarray,
                        gap_tol: float):
    """W14: the best atom re-priced while already active means the *restricted*
    (fixed-support) problem is unconverged.  Push its own fully-corrective solve
    until its gap is in certificate currency, then re-price.  Returns
    ``(w, ll, k_new, score, gap)`` -- the caller keeps the break/converge logic."""
    for _ in range(_REPRICE_ITERS):
        w = _fully_corrective(logF_active, w)
        log_d = _mixture_logd(logF_active, w)
        rgap = float(np.exp(
            _log_mean_exp_rows(logF_active - log_d[None, :]).max()
        ) - 1.0)
        if rgap <= _REPRICE_GAP_RATIO * gap_tol:
            break
    ll = float(log_d.sum())
    k_new, score = oracle.price(log_d)
    return w, ll, k_new, score, score - 1.0


def fit(
    log: GroupedLog,
    eps_grid=(0.0,),
    eta_grid=(0.0,),
    gap_tol: float = 1e-4,
    max_iters: int = 200,
    weight_floor: float = 1e-8,
    max_exact_m: int = 6,
    noise_kernel: str = "uniform",
    poset_class="general",
    lam_grid: Sequence[float] = (1.0,),
    verbose: bool = False,
    force_regime: Optional[str] = None,
    init_order: Optional[object] = None,
) -> FitResult:
    """Fit the NPMLE.  See module docstring; parameters are the *declared*
    residual choices (class, grids, tolerance, noise kernel) discussed in
    the README.  ``poset_class`` defaults to "general" (every partial order
    on the alphabet); pass "sp" for the series-parallel restriction.

    ``weight_floor`` must be < 1e-3: entering atoms are appended at weight
    1e-3 and corrected on the next iteration, so a floor at or above the
    entering weight would prune every new atom before its corrective step
    and spin silently to ``max_iters``.

    ``init_order`` selects the *optional moment initialiser* (ROADMAP M9): an
    int (2, 3, ...) or ``"auto"`` seeds column generation from the candidate
    poset whose order-``k`` precedence-moment best matches the data, instead
    of the default single-best-likelihood atom.  This is a **compute/quality
    dial, not an identifiability parameter** -- the NPMLE is convex over the
    fixed candidate class, so the fitted optimum is identical for every
    ``init_order`` (`docs/miners_as_npmle_specialisations.md` §8; see
    `poset_mixture.initialiser`).  It only changes the warm start, and pays
    off in the large-``m`` regime where per-candidate likelihood is the
    cost.  ``None`` (default) keeps the likelihood-argmax seed."""
    if not 0.0 <= weight_floor < _ENTERING_WEIGHT:
        raise ValueError(
            f"weight_floor must lie in [0, 1e-3): the entering weight is 1e-3, "
            f"and a floor at or above it prunes every new atom before its "
            f"corrective step (got {weight_floor:g})"
        )
    oracle = Oracle(log, eps_grid, eta_grid, max_exact_m=max_exact_m,
                    noise_kernel=noise_kernel, poset_class=poset_class,
                    lam_grid=lam_grid, force_regime=force_regime)
    unexplained = np.isneginf(oracle.logF).all(axis=0)
    if unexplained.any():
        raise ValueError(
            f"groups at indices {np.flatnonzero(unexplained).tolist()} have "
            f"zero density under every candidate atom: no candidate poset "
            f"explains their traces at the given grids.  Widen eps_grid / "
            f"eta_grid (any eps > 0 or eta > 0 atom has full support)."
        )

    # initial atom: best single component (a 1-point mixing measure).  The
    # optional moment initialiser (M9) replaces this likelihood-argmax seed
    # with a cheaper order-k precedence-moment match; it changes only the
    # warm start, never the (convex) optimum reached below.
    if init_order is None:
        k0 = int(np.argmax(oracle.logF.sum(axis=1)))
    else:
        k0 = _moment_seed_atom(oracle, log, init_order)
    active = [k0]
    w = np.ones(1)
    history: List[Tuple[float, float]] = []
    converged = False

    for it in range(1, max_iters + 1):
        logF_active = oracle.logF[active]
        w = _fully_corrective(logF_active, w)
        log_d = _mixture_logd(logF_active, w)
        ll = float(log_d.sum())

        k_new, score = oracle.price(log_d)
        gap = score - 1.0
        history.append((ll, gap))
        if verbose:
            print(f"  iter {it:3d}  loglik {ll:12.4f}  gap {gap:.3e}  atoms {len(active)}")
        if gap <= gap_tol:
            converged = True
            break
        if k_new in active:
            # The best atom is already active while gap > gap_tol: the restricted
            # problem is unconverged and the oracle re-priced the same atom.
            # Tighten the fixed-support solve, then re-price (DESIGN_REVIEW W14).
            w, ll, k_new, score, gap = _tighten_restricted(
                oracle, logF_active, w, gap_tol)
            history[-1] = (ll, gap)
            if gap <= gap_tol:
                converged = True
                break
            if k_new in active:
                # genuinely sublinear endgame on near-collinear atoms: the
                # returned gap stays honest; converged=False flags that the
                # requested tolerance was not certified
                break
            # a genuinely new atom surfaced after tightening: fall through
        active.append(k_new)
        w = np.append(w * (1 - _ENTERING_WEIGHT), _ENTERING_WEIGHT)  # entering atom; corrected next iter

        # prune
        keep = w > weight_floor
        active = [a for a, k in zip(active, keep) if k]
        w = w[keep]
        w /= w.sum()
    else:
        # max_iters exhausted immediately after an append: the entering atom
        # never received its corrective step, and (ll, gap) describe the
        # pre-append mixture.  Drop it so the returned mixture is the one the
        # reported loglik and gap actually certify.
        active.pop()
        w = w[:-1]
        w /= w.sum()

    keep = w > weight_floor
    atoms = [oracle.atoms[a] for a, k in zip(active, keep) if k]
    w = w[keep]
    w /= w.sum()
    order = np.argsort(-w)
    return FitResult(
        atoms=[atoms[i] for i in order],
        weights=w[order],
        loglik=ll,
        gap=gap,
        exact_oracle=oracle.exact,
        oracle_kind=oracle.kind,
        iterations=it,
        history=history,
        converged=converged,
        grids={"eps": tuple(eps_grid), "eta": tuple(eta_grid),
               "lam": tuple(lam_grid)},
        init_order=init_order,
    )


def refit_weights(
    result: FitResult,
    groups: Sequence[Sequence],
    leftovers: Sequence[Sequence],
    timed: bool = False,
    oracle_params: Optional[dict] = None,
) -> Tuple[FitResult, float]:
    """Second-stage weight refinement with the components frozen.

    Undersized blocks (n_g < 3) carry no Kruskal identification for the
    *structure*, but once the components are pinned by the identified stage
    they still carry weight information -- each leftover group contributes a
    convex log-mixture term.  Re-optimising the weights over
    groups + leftovers is convex, so this stage inherits a certificate of
    its own: the returned float is the duality gap of the restricted
    (fixed-support) program on the enlarged data.

    "Frozen" means frozen *in density*, not only in support: the stage-1
    profiled quantities (the empirical trace marginal pbar; the pooled gap
    rate for timed logs) are kept, so every atom's density on the original
    groups is bit-identical between stages.  Novel traces appearing only in
    the leftovers carry zero marginal mass under the frozen profile -- they
    can still be explained by the clean or eps terms of the frozen atoms,
    and a leftover group no frozen atom explains at all raises rather than
    poisoning the refit with NaN.  (Re-profiling over the enlarged trace set
    would silently change every eta > 0 atom's density between stages.)

    The fixed-support gap certifies only that no *frozen* atom deserves more
    weight; a component living predominantly in small blocks is invisible to
    it.  Pass ``oracle_params`` (the same grid / kernel / class keyword
    arguments given to ``fit()``, e.g. ``dict(eps_grid=..., eta_grid=...)``)
    to additionally run the full pricing oracle once on the enlarged
    (frozen-profile) log: the resulting ``full_class_gap`` on the returned
    FitResult upgrades a small value to "no atom in the declared class wants
    in", and prints a large one as the verdict that the leftovers wanted a
    new atom.

    Returns (new FitResult with refined weights, restricted-program gap).
    The loglik reported is on the *enlarged* group set and is therefore not
    comparable to the first-stage loglik.
    """
    combined = list(groups) + list(leftovers)
    log_all = TimedGroupedLog(combined) if timed else GroupedLog(combined)
    # freeze the stage-1 profile (see docstring)
    log_1 = TimedGroupedLog(groups) if timed else GroupedLog(groups)
    pbar = np.zeros(log_all.T)
    for t, i in log_1.tidx.items():
        pbar[log_all.tidx[t]] = log_1.pbar[i]
    log_all.pbar = pbar
    if timed:
        log_all.pooled_rate = log_1.pooled_rate
    logF = np.stack([log_all.group_logf(a) for a in result.atoms])
    unexplained = np.isneginf(logF).all(axis=0)
    if unexplained.any():
        raise ValueError(
            f"combined groups at indices "
            f"{np.flatnonzero(unexplained).tolist()} have zero density under "
            f"every frozen component: the fixed-support refit cannot explain "
            f"them.  Drop the offending leftovers, or refit from grids with "
            f"eps > 0 or eta > 0 (full-support atoms)."
        )
    w = _fully_corrective(logF, result.weights.copy())
    log_d = _mixture_logd(logF, w)
    ll = float(log_d.sum())
    # gap of the fixed-support program, compared in log space (W17):
    # max_k mean_g F[k,g]/d_g - 1
    gap = float(np.exp(_log_mean_exp_rows(logF - log_d[None, :]).max()) - 1.0)
    full_gap = None
    if oracle_params is not None:
        # one full pricing pass over the enlarged frozen-profile log (W9.3):
        # same order of work as first-stage oracle construction, once
        oracle = Oracle(log_all, **oracle_params)
        _, score = oracle.price(log_d)
        full_gap = float(score - 1.0)
    order = np.argsort(-w)
    return (
        FitResult(
            atoms=[result.atoms[i] for i in order],
            weights=w[order],
            loglik=ll,
            gap=gap,
            exact_oracle=False,
            oracle_kind="fixed-support refit",
            iterations=result.iterations,
            history=list(result.history),
            grids=result.grids,
            full_class_gap=full_gap,
        ),
        gap,
    )


def _golden_max(f, lo: float, hi: float, iters: int = 40):
    """Golden-section maximisation of a unimodal-ish 1-D function."""
    phi = (5 ** 0.5 - 1) / 2
    a, b = lo, hi
    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(iters):
        if f1 < f2:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (b - a)
            f2 = f(x2)
        else:
            b, x2, f2 = x2, x1, f1
            x1 = b - phi * (b - a)
            f1 = f(x1)
    x = (a + b) / 2
    return x, f(x)


def polish_nuisances(
    result: FitResult,
    log: GroupedLog,
    rounds: int = 3,
    eps_hi: float = 0.5,
    eta_hi: float = 0.9,
    lam_hi: Optional[float] = None,
) -> FitResult:
    """Continuous refinement of each atom's nuisances off the grid.

    Coordinate ascent: for each atom in turn, maximise the full mixture
    log-likelihood over that atom's eps (then eta, then -- on timed logs --
    lam) by golden section with everything else held fixed, then re-run the
    fully-corrective weight step.  Every accepted move is a line search on
    the true objective (golden section assumes each 1-D slice is
    unimodal-ish; the acceptance check guarantees monotonicity regardless),
    so the likelihood is monotonically non-decreasing; the grids degrade
    from a discretisation choice to an initialisation.

    lam is polished only on timed logs: the untimed density ignores it, so
    the slice would be flat.  ``lam_hi`` defaults to 4x the largest of the
    fitted rates and the pooled rate.  The duality gap is *not* recomputed
    (the certificate remains the fitted one, valid for the grid-restricted
    program); report the polished fit alongside it.
    """
    atoms = list(result.atoms)
    w = result.weights.copy()
    logF = np.stack([log.group_logf(a) for a in atoms])

    coords: List[Tuple[str, float, float]] = [
        ("eps", 0.0, eps_hi), ("eta", 0.0, eta_hi)
    ]
    if isinstance(log, TimedGroupedLog):
        hi = lam_hi if lam_hi is not None else 4.0 * max(
            max(a.lam for a in atoms), log.pooled_rate
        )
        coords.append(("lam", hi * 1e-6, hi))

    def total(k: int, cand: Atom) -> float:
        row = log.group_logf(cand)
        z = logF.copy()
        z[k] = row
        return float(_mixture_logd(z, w).sum())

    for _ in range(rounds):
        for k, a in enumerate(atoms):
            for coord, lo, hi in coords:
                x, _ = _golden_max(
                    lambda v: total(k, replace(a, **{coord: v})), lo, hi
                )
                cand = replace(a, **{coord: x})
                if total(k, cand) >= total(k, a):
                    a = cand
            atoms[k] = a
            logF[k] = log.group_logf(a)
        w = _fully_corrective(logF, w)
    ll = float(_mixture_logd(logF, w).sum())
    order = np.argsort(-w)
    return FitResult(
        atoms=[atoms[i] for i in order],
        weights=w[order],
        loglik=ll,
        gap=result.gap,
        exact_oracle=result.exact_oracle,
        oracle_kind=result.oracle_kind + " + polished nuisances",
        iterations=result.iterations,
        history=list(result.history),
        converged=result.converged,
        grids=result.grids,
        full_class_gap=result.full_class_gap,
    )


def trivial_chain_loglik(log: GroupedLog, eps: float) -> float:
    """Log-likelihood of the *trivial mixture* (empirical chains) on the
    grouped data, at recording-noise level eps.

    This is the falsifiable competitor: on singleton groups it is unbeatable;
    on genuine groups every within-group disagreement costs it a factor
    ~ eps/m!.  Reported so every fit is benchmarked against the degenerate
    solution inside the same likelihood."""
    from .rel import rel_from_trace
    from .likelihood import make_atom

    els = frozenset(log.alphabet)
    lam = log.pooled_rate if isinstance(log, TimedGroupedLog) else 1.0
    atoms = []
    for t, p in zip(log.traces, log.pbar):
        a = make_atom(els, rel_from_trace(t), eps, 0.0, lam=lam)
        atoms.append((a, p))
    logF = np.stack([log.group_logf(a) for a, _ in atoms])
    wts = np.array([p for _, p in atoms])
    return float(_mixture_logd(logF, wts).sum())
