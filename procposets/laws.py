"""The channel map: every density branch an atom can carry, in one place.

An atom's per-trace density is a mixture of three **channels**, and until this
module they were three unrelated conventions scattered through
``likelihood.trace_p``, ``oracle`` and the README's declared-choices ledger:

.. code-block:: text

    p(t) = (1 - eta) [ (1 - eps) * COMPONENT(t | P)      <- the law on L(P)
                       +    eps  * CONTAMINATION(t | P) ]  <- mass off L(P)
           +   eta   * INTERLOPER(t)                       <- another process

Each channel is a **declared statement about a mechanism**, not a convenience
(the README house rule), so each carries its mechanism string here rather than
in a docstring somewhere downstream.

Why the map is worth having: ``refinement_monotone``
-----------------------------------------------------

The meet-closure oracle regime is exact because of one property, and only one:

    **Refinement monotonicity.**  If ``Q`` refines ``P`` (``Q`` contains every
    relation of ``P``) then ``p(t | Q) >= p(t | P)`` for every ``t`` in
    ``L(Q)``.

Given that, replacing any ``P`` by the meet of the observed traces extending it
never lowers the pricing score, so the supremum over *all* posets is attained
on the meet-closure of the observed chains.

The condition previously written down for this — "the density depends on ``P``
only through the extension indicator and ``1/e(P)``" — is **sufficient but not
necessary**, and assuming it was necessary made the exactness look far more
fragile than it is.  Measured over all 3002 strictly-refining poset pairs on
four activities:

===========================  ==========
channel                      violations
===========================  ==========
component: uniform                    0
component: race (any rates)           0
component: Mallows (any theta)        0
contamination: uniform                0
contamination: swap               11136
===========================  ==========

So a **non-uniform component law keeps the exact regime**; it is the *swap
contamination* that loses it, because its mass sits outside ``L(P)`` on a
neighbourhood ``N1(P)`` that a refinement can evict a trace from entirely
(measured: ``cbad`` priced at 0.1667 under ``a<b``, at 0 under the refinement
``a<b, a<c``).  Restrict-and-renormalise laws are monotone for free —
``Z(Q) <= Z(P)`` since ``L(Q)`` is a subset — and the sequential-choice ``race``
law is monotone because refining shrinks the enabled sets, hence its
denominators.

That is why the regime decision is derived here (:func:`meet_closure_is_exact`)
instead of hard-coded against a kernel name.

Two sufficient conditions, so a new channel usually needs no fresh proof
---------------------------------------------------------------------------

1. **Restrict-and-renormalise.**  ``p(t | P) = f(t) / Z(P)`` with ``f``
   independent of ``P`` and ``Z(P) = sum over L(P) of f``.  Refining shrinks
   ``L(P)``, hence ``Z``, hence raises every surviving density.  The uniform
   law (``f = 1``) and a Mallows law (``f = theta^inv``) are both of this form.

2. **Stratified mixtures of (1), with parameter-independent stratum weights.**
   ``p(t | P) = w_k * p_k(t | P)`` for ``t`` in stratum ``k``, where each
   ``p_k`` satisfies (1) and ``w_k`` does not depend on ``P``.  Monotonicity is
   inherited stratum by stratum.  This is the loop law -- body
   ``M``, truncation ``K``, depth weights ``w_k(q, K)`` -- and it is why loops
   will carry the flag: measured, zero violations over all 60 strictly-refining
   body pairs on a three-activity body at three ``(q, K)`` settings.

The sequential-choice ``race`` law satisfies neither form and is monotone for
its own reason (refining shrinks the enabled sets, hence its denominators), so
the conditions above are sufficient rather than exhaustive -- a channel outside
them must be measured, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

__all__ = [
    "COMPONENT_LAWS",
    "CONTAMINATION_KERNELS",
    "Channel",
    "INTERLOPER",
    "get_component_law",
    "get_contamination_kernel",
    "meet_closure_is_exact",
]


@dataclass(frozen=True)
class Channel:
    """One declared density branch."""

    name: str
    role: str
    """``"component"`` | ``"contamination"`` | ``"interloper"``."""
    mechanism: str
    """The modelling statement this channel makes.  Goes in the ledger."""
    refinement_monotone: bool
    """Does ``p(t|Q) >= p(t|P)`` hold for ``Q`` refining ``P``, ``t`` in
    ``L(Q)``?  Meet-closure exactness needs every active channel to say yes."""
    max_m: Optional[int] = None
    """Alphabet ceiling, where the channel enumerates ``S_m``."""
    profiles: Tuple[str, ...] = field(default_factory=tuple)
    """Quantities profiled from the data (plugged in, not optimised)."""


COMPONENT_LAWS: Dict[str, Channel] = {
    "uniform": Channel(
        name="uniform",
        role="component",
        mechanism=(
            "Uniform on the linear extensions: p(t|P) = 1/e(P).  The "
            "maximum-entropy law given the poset, and what makes the poset the "
            "entire structural parameter.  Cannot express a preference AMONG "
            "extensions, so a genuinely lopsided concurrency reads as plain "
            "concurrency."
        ),
        refinement_monotone=True,
    ),
    "race": Channel(
        name="race",
        role="component",
        mechanism=(
            "Racing clocks / Plackett-Luce: at each step the next activity is "
            "drawn from the currently ENABLED set with probability proportional "
            "to its rate.  Mechanistic and self-normalising (no Z), and it is "
            "the ordinal projection of the timed model's semantics, so the "
            "ordinal and timed paths finally share one law.  Note equal rates "
            "do NOT reproduce the uniform law -- they give prod_j 1/k_j, which "
            "differs whenever the enabled-set sizes differ along an extension."
        ),
        refinement_monotone=True,
        profiles=("rates",),
    ),
}

CONTAMINATION_KERNELS: Dict[str, Channel] = {
    "uniform": Channel(
        name="uniform",
        role="contamination",
        mechanism=(
            "Agnostic: eps/M on every order of the base measure.  Expect "
            "structural absorption of structured noise (near-miss traces are "
            "cheaper to explain with a permissive neighbour poset)."
        ),
        refinement_monotone=True,
    ),
    "swap": Channel(
        name="swap",
        role="contamination",
        mechanism=(
            "Local recording jitter: eps uniform on N1(P), the orders one "
            "adjacent transposition outside L(P).  Prices near misses "
            "correctly and identifies eps."
        ),
        refinement_monotone=False,
        max_m=8,
    ),
}

INTERLOPER: Dict[str, Channel] = {
    "pbar": Channel(
        name="pbar",
        role="interloper",
        mechanism=(
            "Declared pseudo-likelihood: the empirical trace marginal plugged "
            "into the eta term to preserve convexity.  P-independent, so it "
            "never threatens refinement monotonicity."
        ),
        refinement_monotone=True,
        profiles=("pbar",),
    ),
}


def _lookup(table: Dict[str, Channel], name: str, what: str) -> Channel:
    try:
        return table[name]
    except KeyError:
        raise ValueError(
            f"unknown {what} {name!r}: the declared channels are "
            f"{sorted(table)}"
        ) from None


def get_component_law(name: str) -> Channel:
    """The declared component law, or a ValueError naming the alternatives."""
    return _lookup(COMPONENT_LAWS, name, "component law")


def get_contamination_kernel(name: str) -> Channel:
    """The declared contamination kernel, or a ValueError naming alternatives."""
    return _lookup(CONTAMINATION_KERNELS, name, "noise kernel")


def meet_closure_is_exact(component_law: str, noise_kernel: str,
                          timed: bool) -> bool:
    """Can the meet-closure regime claim exactness for this channel combination?

    True iff every active channel is refinement-monotone.  Derived from the
    map rather than hard-coded, so adding a monotone law (``race``) inherits
    the exact regime automatically and adding a non-monotone one cannot claim
    it by accident.

    ``timed`` is excluded pending its own monotonicity result: the timed clean
    density depends on ``P`` through the enabled counts *and* carries gap
    factors, so it is conservatively treated as not established.
    """
    if timed:
        return False
    return (get_component_law(component_law).refinement_monotone
            and get_contamination_kernel(noise_kernel).refinement_monotone)
