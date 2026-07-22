"""Diagnostics: every fit ships with its own falsification report.

Three reports, all cheap, all derived from quantities the solver already has:

1. **Certificate**: the Frank-Wolfe duality gap, i.e. how far the fit can be
   from the NPMLE optimum over the oracle's atom class.  With the exact
   oracle (small m) this is genuine global optimality.

2. **Trivial-mixture comparison**: the degenerate chain mixture evaluated in
   the *same* grouped likelihood.  On grouped data it is falsifiable -- every
   within-group disagreement costs it ~ log(eps/m!) -- so "did we beat the
   trivial solution, and by how much per group" is the single most honest
   summary of whether the grouping channel carried information.

3. **Recovery score** (simulation only): match fitted atoms to ground-truth
   posets by relation-set equality, report matched weight mass and the
   symmetric-difference of relations for near misses.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .likelihood import GroupedLog
from .npmle import FitResult, trivial_chain_loglik
from .rel import Rel, SPTree, tree_relations


def trivial_report(log: GroupedLog, fit: FitResult, eps: float) -> str:
    """Trivial-mixture falsification report (was ``report_vs_trivial``).

    Compares the NPMLE fit against the degenerate chain mixture in the same grouped
    likelihood -- "did we beat the trivial solution, and by how much per group".
    """
    triv = trivial_chain_loglik(log, eps=max(eps, 1e-3))
    diff = fit.loglik - triv
    verdict = (
        "grouping channel carried information (mixture beats trivial)"
        if diff > 0
        else "WARNING: trivial mixture not beaten -- grouping carried no signal "
        "(singleton groups, or components indistinguishable at this sample size)"
    )
    return (
        f"loglik(NPMLE) = {fit.loglik:.2f}   loglik(trivial chains) = {triv:.2f}   "
        f"delta = {diff:+.2f}  ({diff / log.G:+.3f} per group)\n  -> {verdict}"
    )


def recovery_report(fit: FitResult, true_trees: Sequence[SPTree],
                    true_weights: Sequence[float], min_weight: float = 0.01) -> str:
    """Match the *poset marginal* of the fit to ground truth.

    The nuisance coordinates (eps, eta) are marginalised first: mass split
    across grid points of the same poset is one component, not several.
    Residual mass on other posets is reported with its relation distance to
    the nearest true poset -- 'noise-attributable neighbour' vs 'genuinely
    spurious' is then the analyst's reading, not a hidden threshold's.
    """
    marg = fit.poset_marginal()
    true_rels: List[Rel] = [tree_relations(t) for t in true_trees]
    lines = ["recovery vs ground truth (poset marginal):"]
    used = set()
    for rel, tw, tt in zip(true_rels, true_weights, true_trees):
        hit = next(
            (
                (i, w, eps, eta)
                for i, (r, desc, w, eps, eta, _lam) in enumerate(marg)
                if r == rel and i not in used
            ),
            None,
        )
        if hit is not None:
            i, w, eps, eta = hit
            used.add(i)
            lines.append(
                f"  {str(tt):30s} true w = {tw:.3f}  ->  EXACT, fitted w = {w:.3f} "
                f"(mean eps = {eps:.3f}, mean eta = {eta:.3f})"
            )
        else:
            lines.append(f"  {str(tt):30s} true w = {tw:.3f}  ->  NOT RECOVERED")
    for i, (r, desc, w, eps, eta, _lam) in enumerate(marg):
        if i in used or w < min_weight:
            continue
        d = min(len(r ^ tr) for tr in true_rels)
        lines.append(
            f"  residual: {desc:30s} w = {w:.3f}  "
            f"({d} relation(s) from nearest true poset)"
        )
    return "\n".join(lines)


def identifiability_report(fit: FitResult, log: GroupedLog,
                           min_weight: float = 1e-6) -> str:
    """Runtime check of the identification condition the estimator leans on.

    The three-view (Kruskal / Allman-Matias-Rhodes) argument needs the
    fitted components' *trace laws* to be linearly independent -- a rank
    condition on the (K x distinct traces) matrix of per-component trace
    probabilities, checkable from `likelihood.trace_p` at no modelling cost.
    Reported as the smallest singular value of that matrix (rows normalized
    to unit L2), with a printed verdict: sigma_min near 0 means two fitted
    components are (near-)indistinguishable at the trace level and their
    *weights* are correspondingly ill-determined, whatever the certificate
    says about the mixture density.  This converts the README's
    identifiability *invocation* into a per-fit check; it is a necessary
    condition, not a proof.
    """
    comps = [a for a, w in zip(fit.atoms, fit.weights) if w > min_weight]
    if len(comps) < 2:
        return ("identifiability check: single component above the weight "
                "floor -- rank condition trivially satisfied")
    M = np.stack([log.trace_p(a) for a in comps])
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    sigmas = np.linalg.svd(M, compute_uv=False)
    smin = float(sigmas[-1])
    verdict = (
        "components' trace laws are linearly independent (weights identified "
        "by the three-view argument, given the blocking assumption)"
        if smin > 1e-3
        else "WARNING: near-collinear component trace laws -- the weight "
        "split between them is ill-determined at the trace level"
    )
    return (
        f"identifiability check: K = {len(comps)} components, "
        f"sigma_min = {smin:.3e} (unit-row {len(comps)}x{M.shape[1]} "
        f"trace-law matrix)\n  -> {verdict}"
    )


def bootstrap_weights(fit: FitResult, log: GroupedLog, B: int = 200,
                      seed: int = 0) -> np.ndarray:
    """Fixed-support weight bootstrap: resample groups, rerun the corrective
    step on the *same* atoms.

    Groups are the exchangeable unit (traces within a group are dependent by
    construction), so resample group indices with replacement, reuse the
    already-computed logF rows, and re-solve the convex weight problem per
    resample -- orders of magnitude cheaper than full refits.  Returns a
    (B, K) array of weights aligned with ``fit.atoms``; quantiles of its
    columns are weight *sampling* uncertainty, which the duality-gap
    certificate deliberately does not speak to.  Support is frozen: this
    does not measure structure (support) uncertainty.
    """
    from .npmle import _fully_corrective

    logF = np.stack([log.group_logf(a) for a in fit.atoms])
    rng = np.random.default_rng(seed)
    out = np.empty((B, len(fit.atoms)))
    for b in range(B):
        idx = rng.integers(0, log.G, size=log.G)
        out[b] = _fully_corrective(logF[:, idx], fit.weights.copy())
    return out
