"""The regime table as unit tests: what each attribution rule can and cannot estimate."""
import math
import random

from procposets.distance import smd
from procposets.estimate import (counting_limit, log_likelihood, mixture_law, reweight,
                          rho_counting, rho_mle, sample_traces, variant_laws)
from procposets.poset import leaf, par, then

# Regime 1: trace-disjoint variants sharing the prefix a (supports {abc,acb} / {ade} / {aed})
DISJOINT = [then(leaf("a"), par(leaf("b"), leaf("c"))),
            then(leaf("a"), leaf("d"), leaf("e")),
            then(leaf("a"), leaf("e"), leaf("d"))]
# Regime 2: overlapping supports (ab), linearly independent laws
OVERLAP = [par(leaf("a"), leaf("b")), then(leaf("a"), leaf("b"))]
# Regime 3: linearly dependent laws -- law(a(x)b) = 1/2 law(a;b) + 1/2 law(b;a)
DEPENDENT = [par(leaf("a"), leaf("b")), then(leaf("a"), leaf("b")), then(leaf("b"), leaf("a"))]

ABC, ADE, AED = ("a", "b", "c"), ("a", "d", "e"), ("a", "e", "d")
AB, BA = ("a", "b"), ("b", "a")


def test_disjoint_counting_recovers_exact_frequencies():
    laws = variant_laws(DISJOINT)
    log = [ABC] * 5 + [ADE] * 3 + [AED] * 2
    assert rho_counting(log, laws) == [0.5, 0.3, 0.2]


def test_disjoint_mle_degenerates_to_counting():
    # at the disjoint boundary the posterior is 0/1, so the MLE IS the counting estimate
    laws = variant_laws(DISJOINT)
    log = [ABC] * 5 + [ADE] * 3 + [AED] * 2
    fit = rho_mle(log, laws)
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(fit, [0.5, 0.3, 0.2]))


def test_overlap_mle_consistent_counting_biased():
    # exact log at rho = (0.3, 0.7): P(ab) = 0.85, P(ba) = 0.15
    laws = variant_laws(OVERLAP)
    log = [AB] * 85 + [BA] * 15
    fit = rho_mle(log, laws)
    assert all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(fit, [0.3, 0.7]))
    # hard counting splits the ambiguous ab half-half: limit (0.15 + 0.425, 0.425)
    cnt = rho_counting(log, laws)
    lim = counting_limit(laws, [0.3, 0.7])
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(cnt, [0.575, 0.425]))
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(cnt, lim))


def test_counting_limit_is_exact_on_disjoint_supports():
    laws = variant_laws(DISJOINT)
    rho = [0.5, 0.3, 0.2]
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(counting_limit(laws, rho), rho))


def test_dependent_laws_ridge_is_init_dependent_but_law_identifiable():
    # exact log at rho = (0.5, 0.25, 0.25): 50 ab, 50 ba
    laws = variant_laws(DEPENDENT)
    log = [AB] * 50 + [BA] * 50
    fit_a = rho_mle(log, laws, init=[0.9, 0.05, 0.05])
    fit_b = rho_mle(log, laws, init=[0.2, 0.7, 0.1])
    # different ridge points ...
    assert max(abs(a - b) for a, b in zip(fit_a, fit_b)) > 0.1
    # ... with the same identifiable quotient (the trace law) and the same likelihood
    for fit in (fit_a, fit_b):
        q = (fit[0] / 2 + fit[1], fit[0] / 2 + fit[2])
        assert math.isclose(q[0], 0.5, abs_tol=1e-6) and math.isclose(q[1], 0.5, abs_tol=1e-6)
    assert math.isclose(log_likelihood(log, laws, fit_a), log_likelihood(log, laws, fit_b), abs_tol=1e-9)


def test_sampler_stays_in_support_and_is_seeded():
    rho = [0.5, 0.3, 0.2]
    support = set(mixture_law(variant_laws(DISJOINT), rho))
    t1 = sample_traces(DISJOINT, rho, 100, random.Random(7))
    t2 = sample_traces(DISJOINT, rho, 100, random.Random(7))
    assert t1 == t2
    assert set(t1) <= support


def test_error_confined_to_branch_rows():
    # regime 1: deterministic rows (START, the (b * c) block) are exact for every N;
    # only the branch rows a, d, e can carry statistical error
    laws = variant_laws(DISJOINT)
    rho = [0.5, 0.3, 0.2]
    traces = sample_traces(DISJOINT, rho, 50, random.Random(3))
    rho_hat = rho_counting(traces, laws)
    _, per_block = smd(reweight(DISJOINT, rho), reweight(DISJOINT, rho_hat))
    noisy = {s for s, ang in per_block.items() if ang > 1e-12}
    assert noisy <= {"a", "d", "e"}


def test_misspecified_variant_set_raises():
    laws = variant_laws(DISJOINT)
    try:
        rho_counting([("z",)], laws)
    except ValueError:
        return
    raise AssertionError("a trace outside every support must raise")
