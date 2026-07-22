"""Component densities for the grouped poset mixture model.

Model
-----
A component (atom of the mixing measure) is a triple

    theta = (P, eps, eta)

* ``P``   -- a partial order (general by default; SP if that class is
             declared); the clean per-trace law is uniform on its linear
             extensions, p_P(sigma) = 1[sigma in L(P)] / e(P).
* ``eps`` -- *recording* noise: a trace is, with probability eps, replaced by
             a uniform permutation.  p^eps = (1-eps) p_P + eps / m!.
* ``eta`` -- *grouping* noise (interloper rate): each trace in a group is,
             with probability eta, an independent draw from the population
             marginal pbar rather than from the group's component.

Group log-density of theta on group g = (sigma_1, ..., sigma_{n_g}):

    log f_theta(g) = sum_j log[ (1-eta) p^eps_P(sigma_j) + eta pbar(sigma_j) ]

Two deliberate design points, documented because they are the residual
modelling choices (see README):

1. **No compression to Q_g.**  Under contamination the hard intersection
   Q_g = meet(traces) is *not* sufficient, and softening it would introduce a
   threshold.  We therefore evaluate the exact per-trace product; the
   intersection orders survive only inside the oracle, as candidate
   generators, where thresholds cannot leak into the likelihood.

2. **pbar is profiled out.**  The interloper term makes the density depend on
   the mixture marginal, which would destroy linearity in the mixing measure
   (and with it, convexity).  The trace marginal is nonparametrically
   identified by the empirical trace distribution irrespective of the
   mixture, so we plug the empirical marginal in.  This is a pseudo-
   likelihood: consistent, mildly inefficient, and it restores exact
   linearity of log f in theta -- the property the whole NPMLE architecture
   rests on.

Everything here is vectorised over the *distinct* traces of the log (there
are at most m! of them, typically far fewer), so the cost of evaluating an
atom on all groups is one small matrix product.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial, log as _ln
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._extensions import preds as _preds
from .rel import (
    GENERAL,
    Rel,
    describe,
    get_poset_class,
    is_partial_order,
    respects,
)


@dataclass(frozen=True)
class Atom:
    """One candidate component of the mixing measure.

    ``noise`` names the recording-noise kernel the eps mass is spread over:

    * ``"uniform"`` -- agnostic: eps / m! on every permutation.  Weakly
      identified against structural absorption (a near-miss trace is priced
      at 1/m!, so a mildly permissive neighbour poset usually explains it
      better; the NPMLE then reports noise as small neighbour atoms).
    * ``"swap"``    -- mechanistic: eps uniform on the swap-neighbourhood
      N1(P) = permutations one adjacent transposition away from L(P).  If
      recording errors really are local jitter, this kernel prices near
      misses correctly (1/|N1| >> 1/m!) and eps becomes identified.

    The kernel is a *declared* model of the logging mechanism -- one of the
    three residual choices of the pipeline (see README).
    """

    rel: Rel
    e: int          # extension count e(P)
    eps: float
    eta: float
    noise_kernel: str = "uniform"
    desc: str = ""  # human-readable form of rel (SP tree string or Hasse covers)
    lam: float = 1.0  # completion rate for timed traces (ignored on untimed logs)

    def describe(self) -> str:
        # the printed token stays ``noise=`` (byte-pinned in a downstream golden);
        # only the attribute it reads was renamed noise -> noise_kernel.
        tag = "" if self.noise_kernel == "uniform" else f", noise={self.noise_kernel}"
        tag += "" if self.lam == 1.0 else f", lam={self.lam:g}"
        return f"{self.desc}  [e={self.e}, eps={self.eps:g}, eta={self.eta:g}{tag}]"


class GroupedLog:
    """A grouped event log, indexed for fast likelihood evaluation.

    Parameters
    ----------
    groups : list of lists of traces (tuples of activity labels).
    """

    def __init__(self, groups: Sequence[Sequence[Tuple[str, ...]]]):
        self.groups = [list(g) for g in groups]
        self.alphabet = sorted({a for g in self.groups for t in g for a in t})
        self.m = len(self.alphabet)
        self.m_fact = factorial(self.m)

        # The likelihood assumes complete, duplicate-free traces everywhere
        # (the 1[t in L(P)] indicator, the 1/m! price, lam^m, one gap per
        # activity).  A trace that is not a permutation of the full alphabet
        # used to produce KeyErrors or silently wrong densities *with a
        # clean certificate*; partial traces are a
        # redesign, not an input.
        alpha_set = set(self.alphabet)
        for gi, g in enumerate(self.groups):
            for t in g:
                if len(t) != self.m or set(t) != alpha_set:
                    kind = ("repeated activity labels" if len(set(t)) < len(t)
                            else "missing activities")
                    raise ValueError(
                        f"trace {t!r} in group {gi} is not a permutation of "
                        f"the full alphabet {self.alphabet} ({kind}): the "
                        f"declared likelihood requires complete, "
                        f"duplicate-free traces"
                    )

        # distinct-trace index
        self.traces: List[Tuple[str, ...]] = sorted({t for g in self.groups for t in g})
        self.tidx: Dict[Tuple[str, ...], int] = {t: i for i, t in enumerate(self.traces)}
        self.T = len(self.traces)
        self.G = len(self.groups)

        # counts[g, t] = multiplicity of distinct trace t in group g
        self.counts = np.zeros((self.G, self.T), dtype=float)
        for gi, g in enumerate(self.groups):
            for t in g:
                self.counts[gi, self.tidx[t]] += 1.0
        self.n_g = self.counts.sum(axis=1)

        # empirical (profiled) trace marginal pbar over distinct traces
        tot = self.counts.sum()
        self.pbar = self.counts.sum(axis=0) / tot

        self._n1_cache: Dict[Rel, tuple] = {}

    # -- vectorised densities ------------------------------------------------

    def in_L(self, rel: Rel) -> np.ndarray:
        """Indicator vector over distinct traces: is trace an extension of rel?"""
        return np.array([1.0 if respects(t, rel) else 0.0 for t in self.traces])

    def swap_kernel(self, rel: Rel):
        """(indicator over distinct traces of membership in N1(rel), |N1(rel)|).

        N1(rel) = permutations *outside* L(rel) that are one adjacent
        transposition away from some extension.  |N1| is computed by full
        enumeration of S_m -- fine for m <= 8 (the enforced wall below);
        for larger alphabets fall back to the uniform kernel.
        """
        if rel not in self._n1_cache:
            if self.m > 8:
                raise ValueError("swap kernel requires m <= 8 (full S_m enumeration)")
            from itertools import permutations

            n1 = set()
            for perm in permutations(self.alphabet):
                if respects(perm, rel):
                    continue
                for i in range(self.m - 1):
                    q = list(perm)
                    q[i], q[i + 1] = q[i + 1], q[i]
                    if respects(tuple(q), rel):
                        n1.add(perm)
                        break
            ind = np.array([1.0 if t in n1 else 0.0 for t in self.traces])
            self._n1_cache[rel] = (ind, max(len(n1), 1))
        return self._n1_cache[rel]

    def trace_p(self, atom: Atom, in_L: np.ndarray | None = None) -> np.ndarray:
        """Per-distinct-trace probability under atom (with eps and eta mixed in)."""
        if in_L is None:
            in_L = self.in_L(atom.rel)
        if atom.eps == 0.0:
            contamination = 0.0  # kernel is density-neutral at eps = 0:
            #                      skip it (avoids the N1 enumeration)
        elif atom.noise_kernel == "swap":
            n1_ind, n1_size = self.swap_kernel(atom.rel)
            contamination = n1_ind / n1_size
        elif atom.noise_kernel == "uniform":
            contamination = 1.0 / self.m_fact
        else:
            raise ValueError(
                f"unknown noise kernel {atom.noise_kernel!r}: the declared kernels "
                f"are 'uniform' and 'swap'"
            )
        clean = (1.0 - atom.eps) * in_L / atom.e + atom.eps * contamination
        return (1.0 - atom.eta) * clean + atom.eta * self.pbar

    def group_logf(self, atom: Atom, in_L: np.ndarray | None = None) -> np.ndarray:
        """Vector over groups: log f_theta(g)."""
        p = self.trace_p(atom, in_L)
        with np.errstate(divide="ignore"):
            logp = np.log(p)
        out = self.counts @ np.where(np.isfinite(logp), logp, -1e30)
        # exact -inf when a required trace has zero probability
        dead = (p == 0.0)
        if dead.any():
            hit = (self.counts[:, dead].sum(axis=1) > 0)
            out[hit] = -np.inf
        return out


def make_atom(
    elements: frozenset,
    rel: Rel,
    eps: float,
    eta: float,
    noise_kernel: str = "uniform",
    poset_class=GENERAL,
    lam: float = 1.0,
) -> Optional[Atom]:
    """Build an atom from a relation set.

    Returns None if the declared hypothesis class does not contain the order
    (only possible for the SP class; the general class contains everything).
    """
    cls = get_poset_class(poset_class)
    assert is_partial_order(elements, rel), (
        f"relation set is not a transitively closed partial order on "
        f"{sorted(elements)}: {sorted(rel)}"
    )  # __debug__-gated: a malformed rel gives silently wrong e(P) downstream
    if not cls.contains(elements, rel):
        return None
    return Atom(
        rel=rel,
        e=cls.extension_count(elements, rel),
        eps=eps,
        eta=eta,
        noise_kernel=noise_kernel,
        desc=describe(elements, rel),
        lam=lam,
    )


class TimedGroupedLog(GroupedLog):
    """Grouped log of *timestamped* traces under racing-clock semantics.

    Enrichment (README section 6): each trace is a pair (sigma, gaps) where
    ``gaps[j]`` is the time between the (j-1)-th and j-th completions.  The
    declared component model: once all its P-predecessors are complete, an
    activity runs an independent Exp(lam) clock (memoryless, so at every
    completion the k currently-enabled activities restart fresh clocks in
    distribution).  The j-th gap is then Exp(lam * k_j) and the finisher is
    uniform on the k_j enabled, giving the clean density

        f(sigma, gaps | P, lam) = prod_j lam * exp(-lam * k_j(sigma, P) * gaps[j])
                                = lam^m * exp(-lam * <k(sigma, P), gaps>)

    for sigma in L(P) (else 0), where k_j = #minimal elements of P restricted
    to the not-yet-completed set.  Timestamps therefore carry concurrency
    information the ordinal trace does not: many-enabled steps have
    systematically shorter gaps, so chains and parallel blocks separate even
    where their extension sets coincide.  Note the *ordinal marginal* of this
    model is prod_j 1/k_j, not the uniform 1/e(P) of the untimed model: using
    timestamps is a (declared) change of component model, not just extra data.

    Noise channels, declared analogously to the untimed model:

    * eta -- interloper: with prob eta the pair is drawn from the profiled
      marginal, plugged in as (empirical ordinal marginal) x (iid Exp(pooled
      rate) gaps).  The gap factor is a crude but declared profile: the
      pooled-rate exponential is the maximum-entropy density matching the
      global mean gap, and it keeps the density linear in theta.
    * eps -- garbled record: uniform ordinal (1/m!) x the same pooled gap
      density.  A mechanistic timed kernel (swap + time perturbation) is
      future work; "swap" is therefore rejected here.

    lam is a mixing coordinate on a grid, like eps and eta: the NPMLE selects
    per-component rates, so heterogeneous service speeds are estimated, not
    assumed away.
    """

    def __init__(self, groups: Sequence[Sequence[Tuple[Tuple[str, ...], Sequence[float]]]]):
        super().__init__([[t for (t, _) in g] for g in groups])
        self.timed_groups = [
            [(tuple(t), tuple(float(x) for x in gaps)) for (t, gaps) in g] for g in groups
        ]
        for g in self.timed_groups:
            for t, gaps in g:
                if len(gaps) != len(t):
                    raise ValueError("need one completion gap per activity")
                for x in gaps:
                    if not (0.0 < x < float("inf")):  # also catches NaN
                        raise ValueError(
                            f"completion gaps must be strictly positive and "
                            f"finite, got {x!r} in trace {t!r}: zero gaps "
                            f"(same-timestamp ties) violate the continuous "
                            f"racing-clock model outright"
                        )
        all_gaps = [x for g in self.timed_groups for (_, gaps) in g for x in gaps]
        self.pooled_rate = len(all_gaps) / sum(all_gaps)
        self._k_cache: Dict[Rel, List[np.ndarray]] = {}

    def _k_vectors(self, rel: Rel) -> List[np.ndarray]:
        """Per distinct ordinal trace: the vector of enabled-counts k_j."""
        if rel not in self._k_cache:
            preds = _preds(self.alphabet, rel)  # {e: frozenset(predecessors)}
            out = []
            for t in self.traces:
                rem = set(t)
                ks = []
                for x in t:
                    ks.append(sum(1 for y in rem if not (preds[y] & rem)))
                    rem.discard(x)
                out.append(np.array(ks, dtype=float))
            self._k_cache[rel] = out
        return self._k_cache[rel]

    def group_logf(self, atom: Atom, in_L: np.ndarray | None = None) -> np.ndarray:
        # Evaluated entirely in log space: the previous linear-space form
        # (lam**m, exp(-lam <k, gaps>)) overflowed for large m log(lam) and
        # underflowed to an exact 0.0 -- hence a spurious -inf group density
        # -- for lam <k, gaps> beyond ~745.
        if atom.noise_kernel == "swap":
            raise ValueError("timed traces support only the uniform eps kernel")
        if in_L is None:
            in_L = self.in_L(atom.rel)
        ks = self._k_vectors(atom.rel)
        lam, lbar, m = atom.lam, self.pooled_rate, self.m
        neg_inf = float("-inf")

        def ln0(x: float) -> float:
            return _ln(x) if x > 0.0 else neg_inf

        log_w_clean = ln0(1.0 - atom.eta) + ln0(1.0 - atom.eps)
        log_w_eps = ln0(1.0 - atom.eta) + ln0(atom.eps) - _ln(self.m_fact)
        log_eta = ln0(atom.eta)
        log_lam_m = m * ln0(lam)
        log_lbar_m = m * _ln(lbar)

        out = np.empty(self.G)
        for gi, g in enumerate(self.timed_groups):
            tot = 0.0
            for t, gaps in g:
                ti = self.tidx[t]
                log_gbar = log_lbar_m - lbar * sum(gaps)  # profiled gap density
                terms = [
                    log_w_eps + log_gbar,
                    log_eta + ln0(self.pbar[ti]) + log_gbar,
                ]
                if in_L[ti]:
                    terms.append(
                        log_w_clean + log_lam_m - lam * float(np.dot(ks[ti], gaps))
                    )
                hi = max(terms)
                if hi == neg_inf:
                    tot = neg_inf
                    break
                tot += hi + _ln(sum(exp(x - hi) for x in terms))
            out[gi] = tot
        return out
