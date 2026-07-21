"""Estimating the mixture weights from an event log -- the estimation/conformance track, live.

The structure (the variants and their tilings) is GIVEN; the log enters the comparison object
only through the estimated weight vector: the empirical block matrix is the analytic matrix
evaluated at the estimated weights, E_N = P(rho_hat_N). So consistency of the matrix IS
consistency of the weight estimator, and the regime table (scratch section, tab:regimes) is a
statement about the attribution rule a_j(omega). This module supplies the forward sampler and
the two estimators:

    rho_counting  assign-then-average with the hard attribution 1[omega in supp D_j], split
                  uniformly when several supports contain the trace. On trace-disjoint variants
                  (Regime 1) this is exact 0/1 counting and consistent by the SLLN; under
                  overlap it is biased, with a.s. limit `counting_limit`.
    rho_mle       the mixing-weight maximum-likelihood estimate via EM (posterior attribution).
                  The component laws are KNOWN -- fixed by the variants -- so only the weights
                  are estimated, a convex problem. Regime 2 (overlap, linearly independent
                  laws): consistent, and it degenerates to `rho_counting` exactly at the
                  disjoint boundary. Regime 3 (linearly dependent laws): the likelihood is flat
                  along a ridge, so the fit is initialisation-dependent -- the
                  non-identifiability is visible, not hidden.

Pure stdlib, like the rest of the core.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from .poset import Model, Poset
from .traces import linear_extensions, trace_distribution

Trace = tuple[str, ...]
Law = dict[Trace, float]


def variant_laws(variants: list[Poset]) -> list[Law]:
    """Each variant's trace law P(. | D_j): uniform over its linear extensions (Assumption 3)."""
    return [trace_distribution([(P, 1.0)]) for P in variants]


def reweight(variants: list[Poset], rho: list[float]) -> Model:
    """Pair each variant poset with its weight -> a ``Model``.  Despite the name
    this is a Model *constructor* (it zips ``variants`` with the given ``rho``), not
    an in-place reweighting: ``rho`` is analytic if true, empirical if estimated."""
    return list(zip(variants, rho))


def mixture_law(laws: list[Law], rho: list[float]) -> Law:
    """The mixture trace law P_rho = sum_j rho_j P(. | D_j)."""
    out: dict[Trace, float] = defaultdict(float)
    for w_j, law in zip(rho, laws):
        for t, p in law.items():
            out[t] += w_j * p
    return dict(out)


def sample_traces(variants: list[Poset], rho: list[float], n: int, rng: random.Random) -> list[Trace]:
    """An i.i.d. event log of n traces from the forward model: variant ~ rho, then a uniform
    linear extension of its poset (enumeration is fine at sandbox scale; Bubley-Dyer at real scale)."""
    les = [linear_extensions(P) for P in variants]
    picks = rng.choices(range(len(variants)), weights=rho, k=n)
    return [rng.choice(les[j]) for j in picks]


def rho_counting(traces: list[Trace], laws: list[Law]) -> list[float]:
    """Assign-then-average (Eq. rhohat) with the hard attribution: a trace is split uniformly over
    the variants whose support contains it. Exact counting when supports are disjoint."""
    r = len(laws)
    acc = [0.0] * r
    for t in traces:
        sup = [j for j in range(r) if t in laws[j]]
        if not sup:
            raise ValueError(f"trace {t} lies outside every variant's support (misspecified variant set)")
        for j in sup:
            acc[j] += 1.0 / len(sup)
    return [v / len(traces) for v in acc]


def counting_limit(laws: list[Law], rho: list[float]) -> list[float]:
    """The a.s. limit of rho_counting under true weights rho (its bias is visible here: the limit
    equals rho iff supports are disjoint or rho happens to match the uniform split)."""
    r = len(laws)
    acc = [0.0] * r
    for t, p in mixture_law(laws, rho).items():
        sup = [j for j in range(r) if t in laws[j]]
        for j in sup:
            acc[j] += p / len(sup)
    return acc


def rho_mle(traces: list[Trace], laws: list[Law], init: list[float] | None = None,
            iters: int = 2000, tol: float = 1e-13) -> list[float]:
    """The mixing-weight MLE by EM (posterior attribution, Regime 2). Known components make the
    log-likelihood concave in rho; on linearly dependent laws (Regime 3) the maximiser is a ridge
    and the returned point depends on `init` -- deliberately so."""
    counts = Counter(traces)
    n = len(traces)
    r = len(laws)
    rho = list(init) if init is not None else [1.0 / r] * r
    for _ in range(iters):
        acc = [0.0] * r
        for t, c in counts.items():
            post = [rho[j] * laws[j].get(t, 0.0) for j in range(r)]
            tot = sum(post)
            if tot <= 0.0:
                raise ValueError(f"trace {t} has zero probability under the current mixture")
            for j in range(r):
                acc[j] += c * post[j] / tot
        new = [v / n for v in acc]
        done = max(abs(a - b) for a, b in zip(rho, new)) < tol
        rho = new
        if done:
            break
    return rho


def log_likelihood(traces: list[Trace], laws: list[Law], rho: list[float]) -> float:
    """Log-likelihood of the log under mixture weights rho (equal along a Regime-3 ridge)."""
    law = mixture_law(laws, rho)
    return sum(c * math.log(law[t]) for t, c in Counter(traces).items())
