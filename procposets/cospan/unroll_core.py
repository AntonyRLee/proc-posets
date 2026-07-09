"""Bounded grounding of a constrained signature (§38) -- the notation-agnostic
core.

The symbolic constrained ``Signature`` (constraints on generator legs) is the
bound-independent comparison invariant.  To *execute* it -- check feasibility,
exhibit witness runs -- we **ground** it: per generator, enumerate the concrete
leg-multiplicity assignments its own constraints admit up to an order ``k``
(capping ``cmax = *`` at ``k``), one **weighted** generator per assignment
(:attr:`Generator.weights`).  The expanded signature then runs through the
ordinary :func:`procposets.cospan.compose.compose_signature`.

This module holds only the OCCN-**independent** grounding of an arbitrary
constrained signature; the OCCN faithful per-firing grounding
(``ground_occn`` / ``ground_run``) lives in :mod:`procposets.occn.unroll_occn`,
which imports *down* into this module -- so ``cospan`` never depends on
``occn``.
"""
from __future__ import annotations

from .feasibility import all_solutions
from .signature import Generator, Signature


def unroll_generator(g: Generator, *, order: int = 8, max_assignments: int = 200_000) -> set[Generator]:
    """Ground one generator into its weighted concrete instances (§38).

    A generator with no constraints is already grounded (every leg weight 1) and is
    returned unchanged. Otherwise each satisfying assignment of ``g.constraints`` (over
    ``[0, order]``, ``cmax = *`` capped at ``order``) becomes one weighted generator;
    the assignment's per-leg counts populate :attr:`Generator.weights`. Legs not
    mentioned by any constraint keep weight 1 (they are absent from the assignment).
    An infeasible generator yields the empty set (it drops out of the grounding)."""
    if not g.constraints:
        return {g}
    out: set[Generator] = set()
    for assign in all_solutions(g.constraints, bound=order, lo=0, max_assignments=max_assignments):
        weights = frozenset((p, int(n)) for p, n in assign.items())
        out.add(Generator(g.label, g.left, g.right, frozenset(), weights))
    return out


def unroll_signature(sig: Signature, *, order: int = 8, max_assignments: int = 200_000) -> Signature:
    """Ground every generator of ``sig`` (see :func:`unroll_generator`). Weighted
    variants dedup automatically: a :class:`Generator`'s ``weights`` is a frozenset of
    ``(port, count)`` pairs, so the boundary multiset is canonical and two distinct
    groundings never collapse while symmetric ones never duplicate (the §36
    ``canonical_key`` multiset requirement, met by construction)."""
    gens: set[Generator] = set()
    for g in sig:
        gens |= unroll_generator(g, order=order, max_assignments=max_assignments)
    return Signature(frozenset(gens))
