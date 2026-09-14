"""(ε,δ) linear-extension count estimation for posets too WIDE for the exact DP.

The exact ideal-lattice engine (``_extensions.count_extensions``) is exponential
only in the *width* of the poset and refuses -- via ``IdealBudgetExceeded`` /
the chain-cover precheck -- once the ideal count would exceed
``MAX_IDEAL_STATES``.  Wide posets are exactly what typed *parallel* seams
manufacture, and for those the only tool left is a Markov-chain estimator whose
cost is width-indifferent.  This module supplies that tier and a single tiered
entry point that prefers the exact count whenever the budget admits it.

The estimator is the standard **telescoping-product** reduction (Jerrum-Valiant-
Vazirani style) on top of the lazy adjacent-transposition sampler of
Karzanov-Khachiyan (mixing Θ(n³ log n) worst case, Bubley-Dyer 1999 /
Wilson 2004).  Relations are added toward a total order one at a time; each
ratio e(P_{i+1})/e(P_i) is the probability that a uniform linear extension of
P_i already satisfies the added pair, estimated from thinned MCMC samples.  The
telescope terminates at a total order, where e = 1, so
``log e(P) = -Σ_i log p̂_i``.  Every added pair is chosen (from a random subset
of the still-incomparable pairs) to be the one whose empirical satisfaction
probability sits nearest 1/2 -- with the pair oriented so p̂ >= 1/2 -- which
keeps every ratio bounded away from 0 (the 1/3-2/3 regime) and every
delta-method variance finite.

Honesty note on the CI.  The per-ratio variance ``(1-p̂)/(p̂·S_eff)`` treats the
``S_eff`` thinned samples as independent.  They are not exactly independent (a
single chain's thinned draws are correlated, and a burn-in shorter than the true
mixing time is biased), so the reported ``log_halfwidth`` is a *nominal* CI whose
coverage is an empirical claim, not a theorem.  The design's E2 coverage gate --
measuring achieved coverage against exact truth on 200 forced-MCMC trials -- is
the validation of that treatment; the guarantee this module ships is "a CI with
*measured* coverage >= nominal at the gate sizes", and ``delta`` is the nominal
tail the gate certifies (or, on redirection, inflates).

Measured (Check E, 2026-08-03, llm-compositional-architecture repo): coverage
0.865 at nominal 95% over 200 forced-telescope trials at n = 7-8 (0.885 after
the 4x-samples redirection), 7/9 at DP-refused widths n = 22-30; calibration
ratio mean|err| / mean(halfwidth/1.96) = 0.99-1.02, i.e. the halfwidth is the
right *size* and the tails are heavier than normal.  Honest tail at these
sizes: delta ~= 0.115 for a nominal-0.05 request.  Callers needing a true
nominal delta should scale the halfwidth accordingly or budget per-atom deltas
against the measured figure.

Only the numpy layer is pulled here; the module is not part of the numpy-free
import contract of the top-level package (it is imported explicitly by the
estimator's consumers, not eagerly at ``import procposets``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ._extensions import MAX_IDEAL_STATES, count_extensions, ideal_state_bound

# The telescope draws its thinned samples from a fixed pool of parallel chains:
# more chains give more (approximately independent) samples per unit of chain
# length, and the vectorised step advances all of them in one array op, so the
# pool is nearly free.  Held internal because the telescope's sample budget --
# not the chain count -- is the knob that trades runtime for CI width.
_TELESCOPE_CHAINS = 64

# Cap on the number of candidate incomparable pairs scored per stage: computing
# P(a before b) for every remaining pair is O(pairs) per stage and the pair
# nearest 1/2 is found just as well from a random subset (the estimate feeds a
# max-over-pairs argmin, not a sum), so 64 keeps a wide poset's per-stage cost
# from scaling with its (quadratic) incomparable-pair count.
_MAX_CANDIDATES = 64


@dataclass(frozen=True)
class CountEstimate:
    """A point estimate of ``log e(P)`` with a log-scale confidence interval.

    ``log_halfwidth`` is the CI halfwidth on the log scale at confidence
    ``1 - delta``; both are exactly 0 for the exact tier (``method ==
    "exact-ideal"``), where the count is certified, not sampled.  The interval
    is reported as *achieved*: the telescope never silently truncates to meet a
    tolerance, so a caller comparing ``log_halfwidth`` to its ``rel_tol`` sees
    the honest width even when the step cap forced a coarser estimate.
    """

    log_e: float
    log_halfwidth: float
    delta: float
    method: str
    n_ratios: int
    n_samples: int
    n_steps: int

    @property
    def e(self) -> float:
        """The count on the natural scale, ``exp(log_e)``."""
        return math.exp(self.log_e)

    def ci(self) -> Tuple[float, float]:
        """The confidence interval on the ``e`` scale: ``(lo, hi)``."""
        return (math.exp(self.log_e - self.log_halfwidth),
                math.exp(self.log_e + self.log_halfwidth))


# ---------------------------------------------------------------------------
# Index-space order utilities (labels are mapped to 0..n-1 once; the sampler
# lives entirely on integer index arrays so every step is a numpy gather).
# ---------------------------------------------------------------------------

def _closure_full(n: int, pairs: Iterable[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Transitive closure of an index relation (Floyd-Warshall, O(n³), once).

    The exact counter and the sampler both assume a transitively closed order
    (an un-closed pair set makes a genuinely ordered element look minimal and
    corrupts both the DP and the incomparability test); re-closing the input is
    cheap insurance against a caller who passed a Hasse diagram.
    """
    reach = [[False] * n for _ in range(n)]
    for i, j in pairs:
        reach[i][j] = True
    for k in range(n):
        rk = reach[k]
        for i in range(n):
            if reach[i][k]:
                ri = reach[i]
                for j in range(n):
                    if rk[j]:
                        ri[j] = True
    return {(i, j) for i in range(n) for j in range(n) if reach[i][j]}


def _comparability(n: int, rel_idx: Iterable[Tuple[int, int]]) -> np.ndarray:
    """Boolean (n×n) incomparability matrix: True iff the pair may be swapped.

    An adjacent transposition of x (before y) is legal exactly when x and y are
    order-incomparable; against a transitively closed ``rel_idx`` this local
    test is sufficient to keep every state a valid linear extension.
    """
    incomp = np.ones((n, n), dtype=bool)
    np.fill_diagonal(incomp, False)
    for i, j in rel_idx:
        incomp[i, j] = False
        incomp[j, i] = False
    return incomp


def _incomp_pairs(n: int, rel_idx: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """The still-incomparable unordered pairs ``(a, b)`` with ``a < b`` (indices)."""
    out: List[Tuple[int, int]] = []
    for a in range(n):
        for b in range(a + 1, n):
            if (a, b) not in rel_idx and (b, a) not in rel_idx:
                out.append((a, b))
    return out


def _positions(states: np.ndarray, n: int) -> np.ndarray:
    """Rank of each element in each state: ``pos[s, e]`` = index of element e."""
    s = states.shape[0]
    pos = np.empty((s, n), dtype=np.int64)
    pos[np.arange(s)[:, None], states] = np.arange(n)[None, :]
    return pos


def _random_topos(n: int, rel_idx: Set[Tuple[int, int]], n_chains: int,
                  rng: np.random.Generator) -> np.ndarray:
    """``n_chains`` uniform-tie-break topological sorts (the default warm start).

    Kahn's algorithm with a uniformly random choice among currently-available
    elements; the random tie-break decorrelates the chains so that samples drawn
    across chains at one time are approximately independent even before mixing.
    Valid against a transitively closed ``rel_idx`` (indegree counts all
    ancestors, so an element frees only once every ancestor has been emitted).
    """
    succ: List[List[int]] = [[] for _ in range(n)]
    indeg0 = [0] * n
    for i, j in rel_idx:
        succ[i].append(j)
        indeg0[j] += 1
    out = np.empty((n_chains, n), dtype=np.int64)
    for c in range(n_chains):
        indeg = indeg0.copy()
        avail = [i for i in range(n) if indeg[i] == 0]
        for p in range(n):
            k = int(rng.integers(0, len(avail)))
            x = avail[k]
            avail[k] = avail[-1]
            avail.pop()
            out[c, p] = x
            for y in succ[x]:
                indeg[y] -= 1
                if indeg[y] == 0:
                    avail.append(y)
    return out


def _step_chains(states: np.ndarray, incomp: np.ndarray, n: int, n_steps: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Advance every chain ``n_steps`` lazy adjacent-transposition steps in place.

    One step, vectorised over ALL chains: draw a position i in {0..n-2} and a
    laziness coin per chain; swap the elements at i, i+1 wherever the coin says
    "move" AND the two elements are incomparable.  Never loops over chains in
    Python -- each step is a handful of array ops regardless of the chain count.
    """
    n_chains = states.shape[0]
    rows = np.arange(n_chains)
    for _ in range(n_steps):
        i = rng.integers(0, n - 1, size=n_chains)
        move = rng.random(n_chains) >= 0.5
        a = states[rows, i]
        b = states[rows, i + 1]
        ok = move & incomp[a, b]
        rk = rows[ok]
        ik = i[ok]
        states[rk, ik] = b[ok]
        states[rk, ik + 1] = a[ok]
    return states


def sample_extensions_mcmc(elements: Sequence, rel, n_chains: int, n_steps: int,
                           rng: np.random.Generator, init=None) -> List[Tuple]:
    """Sample linear extensions of ``(elements, rel)`` by the lazy KK chain.

    A vectorised Karzanov-Khachiyan lazy adjacent-transposition Markov chain over
    linear extensions, run for ``n_steps`` steps on ``n_chains`` parallel chains
    and returning their final states as label tuples.  Works at ANY width -- the
    per-step cost is independent of the poset's ideal count -- which is exactly
    when the exact ideal-DP sampler is unusable.  ``rel`` must be a transitively
    closed relation set (the same Rel view as ``rel.py``), since the swap
    legality test is the local incomparability test.

    ``init`` warm-starts the chains: an iterable of states (each a sequence of
    labels that is itself a valid linear extension of ``rel``); it is
    resampled with replacement to fill ``n_chains`` (or copied when it already
    supplies exactly that many).  Each supplied state is asserted to respect
    ``rel``.  With ``init=None`` the chains start from independent uniform-tie-
    break topological sorts.
    """
    elements = list(elements)
    n = len(elements)
    if n <= 1:
        return [tuple(elements) for _ in range(n_chains)]
    idx = {e: k for k, e in enumerate(elements)}
    rel_idx = {(idx[a], idx[b]) for (a, b) in rel}
    incomp = _comparability(n, rel_idx)
    if init is None:
        states = _random_topos(n, rel_idx, n_chains, rng)
    else:
        init = list(init)
        base = np.array([[idx[x] for x in s] for s in init], dtype=np.int64)
        pos = _positions(base, n)
        for (i, j) in rel_idx:
            assert bool((pos[:, i] < pos[:, j]).all()), "init state violates rel"
        if base.shape[0] == n_chains:
            states = base.copy()
        else:
            sel = rng.integers(0, base.shape[0], size=n_chains)
            states = base[sel].copy()
    _step_chains(states, incomp, n, n_steps, rng)
    return [tuple(elements[i] for i in states[c]) for c in range(n_chains)]


def approx_count_extensions(elements: Iterable, rel, *, rel_tol: float = 0.1,
                            delta: float = 0.05,
                            rng: Optional[np.random.Generator] = None,
                            exact_budget: Optional[int] = None,
                            max_total_steps: int = 50_000_000) -> CountEstimate:
    """Tiered (ε,δ) estimate of ``e(P)`` for ``P = (elements, rel)``.

    Exact tier: if the chain-cover ideal bound fits ``exact_budget`` (default
    ``MAX_IDEAL_STATES``), return the certified ideal-DP count with
    ``log_halfwidth = delta = 0``.  ``exact_budget=0`` forces the MCMC telescope
    even on a small poset -- how the validation gates measure the estimator's
    coverage against exact truth.

    MCMC telescope (the wide-poset tier): add relations toward a total order one
    at a time, each ratio estimated from thinned samples of the lazy KK chain,
    each added pair the one nearest the 1/2 regime, oriented so ``p̂ >= 1/2``.
    Since ``e(total order) = 1``, ``log e(P) = -Σ log p̂_i``; per-ratio
    delta-method variances sum to a normal CI on the log scale at confidence
    ``1 - delta``.  Per-stage sampling is allocated evenly against the projected
    stage count ``T_est ≈ log₂(n!)`` so the propagated halfwidth targets
    ``rel_tol``; the ACHIEVED halfwidth is reported honestly whether or not it
    meets ``rel_tol``.  ``max_total_steps`` (counted as chain-steps, not
    chain-steps × chains) is a hard cap: on hitting it, remaining stages run at
    reduced sample counts and the (larger) achieved halfwidth is returned rather
    than the run hanging or being silently truncated.

    ``rng`` must be an explicit numpy Generator (``np.random.default_rng(seed)``):
    a silent default seed would make results irreproducible by construction, so
    ``rng=None`` raises ``ValueError``.
    """
    if rng is None:
        raise ValueError(
            "approx_count_extensions requires an explicit numpy Generator "
            "(e.g. np.random.default_rng(seed)); a silent default seed would "
            "make the estimate irreproducible"
        )
    elements = list(elements)
    if len(set(elements)) != len(elements):
        raise ValueError("elements must be distinct hashable labels")
    n = len(elements)
    idx = {e: k for k, e in enumerate(elements)}
    rel_idx = _closure_full(n, {(idx[a], idx[b]) for (a, b) in rel})

    budget = MAX_IDEAL_STATES if exact_budget is None else exact_budget
    label_pairs = frozenset((elements[i], elements[j]) for (i, j) in rel_idx)

    # Trivial / exact tiers.  n <= 1 has e = 1; otherwise, unless the telescope
    # is forced (budget == 0), defer to the certified count whenever its width
    # fits the ideal-state budget.
    if n <= 1:
        return CountEstimate(0.0, 0.0, 0.0, "exact-ideal", 0, 0, 0)
    if budget and ideal_state_bound(elements, label_pairs, budget) <= budget:
        e = count_extensions(elements, label_pairs, max_states=budget)
        return CountEstimate(math.log(e), 0.0, 0.0, "exact-ideal", 0, 0, 0)

    incomp_list = _incomp_pairs(n, rel_idx)
    if not incomp_list:  # already a total order (forced path): e = 1 exactly
        return CountEstimate(0.0, 0.0, 0.0, "exact-ideal", 0, 0, 0)

    z = NormalDist().inv_cdf(1.0 - delta / 2.0)
    # Projected number of ratios: log e(P) <= log(n!), and each -log p̂ >= log 2,
    # so log₂(n!) upper-bounds the stage count -- a conservative even split of
    # the total variance budget across stages.
    t_est = max(1.0, math.lgamma(n + 1) / math.log(2.0))
    burnin_base = max(2000, n ** 3)
    thinning = max(50, n * n // 2)
    per_ratio_var = (rel_tol / z) ** 2 / t_est
    # Worst-case p̂ = 1/2 gives (1-p̂)/p̂ = 1, so this many independent samples
    # per ratio caps each per-ratio variance at its share of the budget for ANY
    # p̂ >= 1/2; skewed ratios then come in comfortably under.
    n_chains = _TELESCOPE_CHAINS
    s_target = max(n_chains, math.ceil(1.0 / per_ratio_var))

    states = _random_topos(n, rel_idx, n_chains, rng)
    sum_neg_log = 0.0
    sum_var = 0.0
    n_ratios = 0
    total_steps = 0

    while incomp_list:
        incomp = _comparability(n, rel_idx)
        remaining = max_total_steps - total_steps
        burnin = burnin_base
        rounds = math.ceil(s_target / n_chains)
        if burnin + rounds * thinning > remaining:  # step cap: degrade, don't hang
            burnin = min(burnin, max(0, remaining // 2))
            rounds = max(1, min(rounds, max(1, (remaining - burnin) // thinning)))

        _step_chains(states, incomp, n, burnin, rng)
        total_steps += burnin
        samp_rows = []
        for _ in range(rounds):
            _step_chains(states, incomp, n, thinning, rng)
            total_steps += thinning
            samp_rows.append(states.copy())
        samps = np.concatenate(samp_rows, axis=0)
        pos = _positions(samps, n)
        s_eff = samps.shape[0]

        cand = incomp_list
        if len(cand) > _MAX_CANDIDATES:
            sel = rng.choice(len(cand), size=_MAX_CANDIDATES, replace=False)
            cand = [cand[k] for k in sel]
        best_score = math.inf
        best_pair = cand[0]
        best_p = 0.5
        for (a, b) in cand:
            q = float((pos[:, a] < pos[:, b]).mean())
            if q >= 0.5:
                dpair, p = (a, b), q
            else:
                dpair, p = (b, a), 1.0 - q
            score = p - 0.5  # distance from the balanced 1/2 regime
            if score < best_score:
                best_score, best_pair, best_p = score, dpair, p

        # Clamp p̂ off {0, 1} at the sample resolution: a finite-sample unanimous
        # ratio (p̂ = 1) must not claim zero variance / zero log contribution.
        p_hat = min(max(best_p, 1.0 / (s_eff + 1)), s_eff / (s_eff + 1))
        sum_neg_log += -math.log(p_hat)
        sum_var += (1.0 - p_hat) / (p_hat * s_eff)
        n_ratios += 1

        di, dj = best_pair
        # Incremental closure: rel_idx is already closed, so adding di < dj only
        # creates x < y for x in preds(di)∪{di}, y in succs(dj)∪{dj}.
        ua = {di} | {x for (x, y) in rel_idx if y == di}
        vb = {dj} | {y for (x, y) in rel_idx if x == dj}
        rel_idx |= {(x, y) for x in ua for y in vb}

        # Warm-start the next stage from current states that already satisfy the
        # added pair (they are valid extensions of the grown order for free);
        # resample them to refill the pool, or fall back to fresh topo sorts if
        # a coarse stage left none.
        fpos = _positions(states, n)
        good = states[fpos[:, di] < fpos[:, dj]]
        if good.shape[0] == 0:
            states = _random_topos(n, rel_idx, n_chains, rng)
        else:
            sel = rng.integers(0, good.shape[0], size=n_chains)
            states = good[sel].copy()

        incomp_list = _incomp_pairs(n, rel_idx)

    log_halfwidth = z * math.sqrt(sum_var)
    return CountEstimate(sum_neg_log, log_halfwidth, delta, "mcmc-telescope",
                         n_ratios, s_target, total_steps)
