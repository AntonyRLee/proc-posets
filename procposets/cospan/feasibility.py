"""Integer feasibility for leg-multiplicity constraint systems (§32).

Given a set of :class:`~cpm.cospan.signature.LinearConstraint` over leg variables
``n_p`` (one per leg port), decide whether a satisfying non-negative integer
assignment exists, exhibit a witness, and report each variable's feasible range.

Two backends, by the user's call:

1. **Bounded enumeration (default, lightweight).** Each variable ranges over a small
   integer box ``[lo, bound]`` (tightened first by the unary ``>=``/``<=``/``==``
   constraints); non-unary constraints filter the product. The user bounds the
   problem size; ``bound=1`` (the default) pins every variable to 1 -- the 1-1
   baseline -- so an unconstrained signature behaves exactly as before. If the box
   is too large, :class:`FeasibilityTooLarge` is raised (the signal to use backend 2).
2. **External ILP solver (TODO -- see CLASS_EXTRACTION.md §34).** Pass ``solver=`` a
   callable to dispatch to e.g. PuLP / OR-Tools for unbounded or large systems. Not
   implemented here; the hook is in place so callers don't change.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable

from .signature import LinearConstraint, Port


class FeasibilityTooLarge(Exception):
    """The bounded-enumeration box exceeds ``max_assignments`` -- raise the bound
    deliberately, or pass an ILP ``solver`` (backend 2)."""


def variables(constraints: Iterable[LinearConstraint]) -> set[Port]:
    out: set[Port] = set()
    for c in constraints:
        out |= c.ports()
    return out


def _satisfied(c: LinearConstraint, assign: dict) -> bool:
    s = sum(coeff * assign[p] for p, coeff in c.terms)
    if c.rel == "<=":
        return s <= c.rhs
    if c.rel == ">=":
        return s >= c.rhs
    return s == c.rhs


def _unary_domain(var: Port, constraints: list[LinearConstraint], lo: int, bound: int) -> range:
    """Tighten ``[lo, bound]`` for ``var`` using the single-variable constraints
    ``{var:1} >= k`` / ``<= k`` / ``== k``. ``bound`` is a uniform global ceiling --
    a leg whose constraints force a value above it is simply infeasible at this
    bound (raise the bound), so ``==`` is clamped like the others."""
    low, high = lo, bound
    for c in constraints:
        if len(c.terms) == 1:
            (p, coeff), = tuple(c.terms)
            if p == var and coeff == 1:
                if c.rel in (">=", "=="):
                    low = max(low, c.rhs)
                if c.rel in ("<=", "=="):
                    high = min(high, c.rhs)
    return range(low, high + 1)  # empty iff low > high


def _enumerate(cons, *, bound, lo, pinned, max_assignments):
    """Yield each satisfying integer assignment ``{port: n}`` in the bounded box.

    The single home for the enumeration prologue that ``solve`` / ``all_solutions``
    / ``ranges`` each spelled out: the same unary-tightened domain build, the same
    ``FeasibilityTooLarge`` size guard, the same ``itertools.product`` order and the
    same ``_satisfied`` filter -- only the *accumulation* over the yielded
    assignments differs between the callers."""
    pinned = dict(pinned or {})
    free = sorted(variables(cons) - set(pinned), key=str)
    domains = [_unary_domain(v, cons, lo, bound) for v in free]
    size = 1
    for d in domains:
        size *= len(d)
        if size > max_assignments:
            raise FeasibilityTooLarge(
                f"{size}+ assignments over {len(free)} vars at bound {bound}; "
                "raise max_assignments, lower the bound, or pass an ILP solver (§34)"
            )
    for combo in itertools.product(*domains):
        assign = dict(pinned)
        assign.update(zip(free, combo))
        if all(_satisfied(c, assign) for c in cons):
            yield assign


def solve(
    constraints: Iterable[LinearConstraint],
    *,
    bound: int = 1,
    lo: int = 1,
    pinned: dict | None = None,
    max_assignments: int = 200_000,
    solver=None,
) -> dict | None:
    """A satisfying integer assignment ``{port: n}`` (over the constrained legs), or
    ``None`` if none exists in the box. ``pinned`` fixes some variables (e.g. an
    upstream-determined input total). ``solver`` (backend 2) overrides the default
    bounded enumeration."""
    cons = list(constraints)
    if solver is not None:
        return solver(cons, bound=bound, lo=lo, pinned=dict(pinned or {}))
    return next(
        _enumerate(cons, bound=bound, lo=lo, pinned=pinned, max_assignments=max_assignments),
        None,
    )


def feasible(constraints: Iterable[LinearConstraint], **kw) -> bool:
    """Whether the system has a satisfying assignment in the box (see :func:`solve`)."""
    return solve(constraints, **kw) is not None


def all_solutions(
    constraints: Iterable[LinearConstraint], *, bound: int = 1, lo: int = 0,
    pinned: dict | None = None, max_assignments: int = 200_000,
) -> list[dict]:
    """Every satisfying integer assignment ``{port: n}`` over the constrained legs
    (the **unroller** backend, §38). Unlike :func:`ranges` -- which reports each
    variable's marginal min/max and so loses the joint correlation a partition
    ``n_i + n_n == T`` imposes -- this returns each full joint assignment, one per
    grounded generator variant.

    ``lo`` defaults to **0** here (not 1): a shared-key partition leg may legitimately
    carry 0 objects (the "i alone, no n" object-XOR branch). Interval constraints
    still clamp the real lower bound back up (``[1,5]`` -> ``lo`` 1 for that leg).
    Legs mentioned by no constraint are not variables -- they stay at their default
    count 1 and are simply absent from the returned dicts. Bounded-enumeration only;
    raises :class:`FeasibilityTooLarge` past ``max_assignments``."""
    return list(_enumerate(
        list(constraints), bound=bound, lo=lo, pinned=pinned, max_assignments=max_assignments,
    ))


def ranges(
    constraints: Iterable[LinearConstraint], *, bound: int = 1, lo: int = 1, pinned: dict | None = None,
    max_assignments: int = 200_000,
) -> dict[Port, tuple[int, int]]:
    """Per-variable ``(min, max)`` over all feasible assignments in the box -- the
    valid integer range of each leg multiplicity. ``{}`` if the system is
    infeasible. Bounded-enumeration backend only."""
    out: dict[Port, tuple[int, int]] = {}
    for assign in _enumerate(
        list(constraints), bound=bound, lo=lo, pinned=pinned, max_assignments=max_assignments,
    ):
        for p, n in assign.items():
            lo_p, hi_p = out.get(p, (n, n))
            out[p] = (min(lo_p, n), max(hi_p, n))
    return out
