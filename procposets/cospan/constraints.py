"""Accessible builders for leg-multiplicity constraints (§32).

A generator's boundary is a **multiset** of typed wires: each leg-port ``p`` carries
an integer multiplicity variable ``n_p`` (the number of objects on that wire).
``Generator.left``/``right`` are the multiset *support*; the multiplicities are these
variables, **default 1**, governed by :class:`~procposets.cospan.signature.LinearConstraint`.
The frontier (``Counter[Port]``) is the grounded multiset.

Constraints are general linear type-inequalities ``Σ coeff·n_p <rel> rhs``. These
builders cover the common cases; the OCCN bindings (cardinality intervals +
shared-key partitions) are just particular instances -- see the OCCN leg-constraint
builder in :mod:`procposets.occn.to_signature`.

Authoring example::

    from procposets.cospan.constraints import interval, partition, cset
    g = Generator("s", left, right, cset(
        interval(order_in, 1, None),     # 1..* orders
        interval(cont_in, 1, 1),         # exactly one container
    ))
"""
from __future__ import annotations

from collections.abc import Iterable

from .signature import LinearConstraint, Port


def constraint(coeffs: dict, rel: str, rhs: int) -> LinearConstraint:
    """A raw linear constraint ``Σ coeff·n_p <rel> rhs`` (``rel`` in ``<=``/``>=``/``==``).
    Zero coefficients are dropped."""
    return LinearConstraint(
        frozenset((p, int(c)) for p, c in coeffs.items() if c != 0), rel, int(rhs)
    )


def at_least(port: Port, k: int) -> LinearConstraint:
    return constraint({port: 1}, ">=", k)


def at_most(port: Port, k: int) -> LinearConstraint:
    return constraint({port: 1}, "<=", k)


def exactly(port: Port, k: int) -> LinearConstraint:
    return constraint({port: 1}, "==", k)


def interval(port: Port, cmin: int = 1, cmax: int | None = None) -> list[LinearConstraint]:
    """``cmin <= n_port <= cmax`` (``cmax is None`` = unbounded ``*``). The OCCN
    cardinality of one marker."""
    out = [at_least(port, cmin)]
    if cmax is not None:
        out.append(at_most(port, cmax))
    return out


def partition(total: Port, parts: Iterable[Port]) -> LinearConstraint:
    """``Σ n_part == n_total`` -- the shared-key object distribution: the legs
    sharing a key partition the objects on ``total`` (each object to exactly one)."""
    coeffs: dict = {}
    for p in parts:
        coeffs[p] = coeffs.get(p, 0) + 1
    coeffs[total] = coeffs.get(total, 0) - 1
    return constraint(coeffs, "==", 0)


def remap(c: LinearConstraint, f) -> LinearConstraint:
    """Rewrite a constraint's leg ports via ``f: Port -> Port`` (e.g. the
    ``forget_provenance`` quotient), summing the coefficients of any ports that
    collapse onto the same image."""
    agg: dict = {}
    for p, coeff in c.terms:
        q = f(p)
        agg[q] = agg.get(q, 0) + coeff
    return constraint(agg, c.rel, c.rhs)


def union(generators: Iterable) -> frozenset:
    """The accumulated constraint system of a set of generators (§32): the union of
    each generator's leg constraints. Wires glued across firings share their
    ``Port`` identity, so the union over the shared variables *is* the run's
    system (loop-free; one firing per generator)."""
    out: set = set()
    for g in generators:
        out |= set(g.constraints)
    return frozenset(out)


def cset(*items) -> frozenset:
    """Flatten constraints / lists-of-constraints into one ``frozenset`` for a
    ``Generator``."""
    out: set = set()
    for it in items:
        if isinstance(it, LinearConstraint):
            out.add(it)
        else:
            out.update(it)
    return frozenset(out)
