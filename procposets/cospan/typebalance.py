"""Type-balance admissibility ``⋈`` for the object-centric instances.

The general decorated-cospan framework places no constraint on how a generator
relates its input types to its output types -- that freedom is the point.  The
OC *notations*, though, must forbid **arbitrary type conversion**: an activity
may not turn a ``lab`` into a ``bed`` as if transmuting one object into another.

The naive balance "predecessor and successor type-sets must coincide" is too
strong -- it rejects every object-*creating* or *consuming* activity (an
activity that mints a ``bed`` has no ``bed`` predecessor).  ``admissible ≡ True``
is too weak -- it permits the conversion above.  The principled middle is
**type-balance modulo a declared per-activity create/consume profile** ``κ``:

  * a type neither created nor consumed by ``a`` is *conserved* -- present on
    the left iff present on the right (it flows through);
  * a created type may appear right-only; a consumed type, left-only.

``κ`` is declared for an authored model (``pathway_master``) and derived from an
OCEL for a discovered one (:func:`kappa_from_ocel`: a type is *created* at the
activity of an object's first event, *consumed* at its last).  Untyped
(``None``) wires are exempt, matching ``LMGraph.validate``'s treatment of
untyped edges.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .signature import Generator, Signature

Bundle = frozenset  # frozenset[tuple[str, str | None]]


@dataclass(frozen=True)
class Profile:
    """The object-type license ``κ(a)`` of an activity: which types it may mint
    (``creates``) and which it may absorb (``consumes``).  Anything else must be
    conserved."""

    creates: frozenset[str] = field(default_factory=frozenset)
    consumes: frozenset[str] = field(default_factory=frozenset)


Kappa = dict[str, Profile]


def _delta(left_types: set, right_types: set) -> tuple[set, set]:
    """(created, consumed) for a context: created = right-only, consumed =
    left-only.  Through-types (on both sides) are conserved and drop out."""
    return right_types - left_types, left_types - right_types


def admissible(label: str, P: Bundle, S: Bundle, kappa: Kappa) -> bool:
    """``⋈``: is the context ``(P, S)`` for activity ``label`` type-balanced
    under ``κ``?  Activities absent from ``κ`` are unconstrained (permissive),
    so a partial ``κ`` never produces false rejections."""
    prof = kappa.get(label)
    if prof is None:
        return True
    left = {t for (_, t) in P if t is not None}
    right = {t for (_, t) in S if t is not None}
    created, consumed = _delta(left, right)
    return created <= prof.creates and consumed <= prof.consumes


@dataclass(frozen=True)
class Violation:
    generator: Generator
    bad_creates: frozenset[str]   # types emitted without a creation licence
    bad_consumes: frozenset[str]  # types absorbed without a consumption licence

    def __str__(self) -> str:
        bits = []
        if self.bad_creates:
            bits.append(f"creates {set(self.bad_creates)} unlicensed")
        if self.bad_consumes:
            bits.append(f"consumes {set(self.bad_consumes)} unlicensed")
        return f"{self.generator.label}: " + "; ".join(bits)


def generator_violation(g: Generator, kappa: Kappa) -> Violation | None:
    """The type-balance violation of a single generator under ``κ``, or ``None``
    if it is balanced (or its label is unconstrained)."""
    prof = kappa.get(g.label)
    if prof is None:
        return None
    left = {p.typ for p in g.left if p.typ is not None}
    right = {p.typ for p in g.right if p.typ is not None}
    created, consumed = _delta(left, right)
    bad_c = created - prof.creates
    bad_k = consumed - prof.consumes
    if bad_c or bad_k:
        return Violation(g, frozenset(bad_c), frozenset(bad_k))
    return None


def type_balance(sig: Signature, kappa: Kappa) -> list[Violation]:
    """Every generator in ``sig`` that violates ``κ``.  Empty list == balanced."""
    out = [generator_violation(g, kappa) for g in sig]
    return [v for v in out if v is not None]


def kappa_from_ocel(ocel) -> Kappa:
    """Derive ``κ`` from an OCEL: a type is *created* at the activity of an
    object's first event and *consumed* at its last (by timestamp)."""
    rel = ocel.relations
    creates: dict[str, set] = defaultdict(set)
    consumes: dict[str, set] = defaultdict(set)
    ordered = rel.sort_values("ocel:timestamp")
    for _oid, grp in ordered.groupby("ocel:oid", sort=False):
        otype = grp["ocel:type"].iloc[0]
        creates[grp["ocel:activity"].iloc[0]].add(otype)
        consumes[grp["ocel:activity"].iloc[-1]].add(otype)
    labels = set(rel["ocel:activity"])
    return {
        a: Profile(frozenset(creates.get(a, ())), frozenset(consumes.get(a, ())))
        for a in labels
    }
