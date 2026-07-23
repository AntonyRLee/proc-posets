"""The **factored skeleton**: per-activity context families kept per-arc, the
cross-arc product never materialised.

:func:`engine.extract_signature` (slow) materialises, per activity ``a``, every
context ``(P, S) in B_a x F_a`` as one atomic :class:`Generator` -- intractable on
wide object-centric nets, where ``|B_a|`` alone is the product over ``a``'s in-arcs
of their XOR-alternative counts (exponential in the number of AND-combined arcs).
Yet each side family *is* definitionally that product: ``backward_bundles`` /
``forward_bundles`` are ``engine._and`` over the per-arc ``engine._traverse``
results.  This module stops one step earlier and keeps the per-arc families
themselves -- a *sum* over arcs, O(model + per-arc alternatives) -- as one
:class:`FactoredGenerator` per activity.

Consumers re-derive what they need lazily:

* :meth:`FactoredSignature.materialise` -- the full slow-equivalent
  :class:`Signature`, reusing the very same ``engine`` helpers (``_and``,
  ``_collapse_pure_terminus``, ``_gen_from_bundles``, ``admissible``), so it is
  equal **by construction** to ``extract_signature`` (the migration-discipline
  seam ``tests/regression/test_cospan_skeleton.py`` pins exactly).
* :meth:`FactoredGenerator.ready_lefts` -- the output-sensitive fire-time query
  :func:`procposets.cospan.compose.compose_signature` uses: only the alternatives the
  current port pool covers enter the cross-arc product, so composition never
  pays for the astronomically many contexts that are not currently fireable.
  The admissibility join ``><`` (``typebalance.admissible``) is applied by the
  consumer at pairing time -- lazy-``><`` over the two marginals yields exactly
  the materialised ``><``-filtered product.

Terminus handling mirrors the slow engine's two modes: the default strip
(``gamma2`` legs absorbed) commutes with the cross-arc union, so it is baked
into the stored per-arc families; the ``surface_termini`` pure-terminus
collapse reads the *whole* side (is every leg ``gamma2``?), so it cannot be
applied per-arc -- the raw families are stored and ``collapse_terminus`` defers
the collapse to materialisation of each concrete side.
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine import (
    _and,
    _collapse_pure_terminus,
    _gen_from_bundles,
    _prepare_extraction,
    _strip_termini,
    _traverse,
)
from .lmgraph import LMGraph
from .signature import Port, Signature
from .typebalance import Kappa, admissible

Bundle = frozenset  # frozenset[tuple[str, str | None]] -- as engine.Bundle


@dataclass(frozen=True)
class FactoredGenerator:
    """One activity's factored context families.

    ``left`` / ``right`` hold, per in-/out-arc of the activity, that arc's
    XOR-alternative bundle family (``engine._traverse``'s output, frozen).  A
    concrete boundary side is the union of ONE alternative per arc, i.e. the
    side family equals ``engine._and`` of the per-arc families -- but the
    product is only ever formed on demand (:meth:`left_bundles` /
    :meth:`right_bundles`), or pool-restricted (:meth:`ready_lefts`)."""

    label: str
    left: tuple[frozenset, ...]   # per in-arc: frozenset[Bundle] alternatives
    right: tuple[frozenset, ...]  # per out-arc: frozenset[Bundle] alternatives
    collapse_terminus: bool = False  # surface_termini: all-gamma2 union -> zero-right

    def left_bundles(self) -> set:
        """``B_a`` materialised -- the full cross-arc product (use sparingly)."""
        return _and([set(f) for f in self.left])

    def right_bundles(self) -> set:
        """``F_a`` materialised, with the mode's terminus handling applied."""
        S = _and([set(f) for f in self.right])
        return _collapse_pure_terminus(S) if self.collapse_terminus else S

    def left_can_be_empty(self) -> bool:
        """Is ``frozenset()`` in the left family -- i.e. is this a source
        (zero-left) alternative?  The empty union arises exactly when every
        in-arc admits the empty alternative (trivially true with no in-arcs)."""
        return all(any(not b for b in f) for f in self.left)

    def ready_lefts(self, available: dict[Port, int]) -> set:
        """The nonempty concrete left bundles fireable on ``available`` now.

        Exact and output-sensitive: a union is covered iff every factor is, so
        restricting each arc to its covered alternatives *before* the product
        yields precisely the covered members of the full family, at the cost of
        the covered product only (bounded by the pool, not the model).  Legs
        carry weight 1 -- extracted generators are ungrounded (weights are the
        separate §38 grounding, absent from any LM-graph extraction)."""
        per_arc = []
        for f in self.left:
            cov = {
                b for b in f
                if all(available.get(Port(p, t, self.label), 0) >= 1 for (p, t) in b)
            }
            if not cov:
                return set()  # this arc has no fireable alternative
            per_arc.append(cov)
        out = _and(per_arc)
        out.discard(frozenset())  # the zero-left alternative seeds, never fires
        return out


@dataclass(frozen=True)
class FactoredSignature:
    """The factored twin of :class:`Signature`: one :class:`FactoredGenerator`
    per activity node (same-label activities stay separate entries, exactly as
    they contribute separately to the slow signature's frozenset union), plus
    the admissibility profile ``kappa`` the consumer applies at pairing time
    (the slow engine applies it at extraction; deferring it is what makes the
    lazy join land on the identical context set)."""

    generators: tuple[FactoredGenerator, ...]
    kappa: Kappa | None = None

    def __len__(self) -> int:
        return len(self.generators)

    def __iter__(self):
        return iter(self.generators)

    def labels(self) -> set[str]:
        return {fg.label for fg in self.generators}

    def materialise(self) -> Signature:
        """The full slow-equivalent :class:`Signature` -- every ``(P, S)``
        context of every activity, ``kappa``-filtered.  Intractable exactly
        where :func:`engine.extract_signature` is; exists as the equality seam
        (and for below-cap models that want the atomic form)."""
        gens = set()
        for fg in self.generators:
            F = fg.right_bundles()
            for P in fg.left_bundles():
                for S in F:
                    if self.kappa is not None and not admissible(fg.label, P, S, self.kappa):
                        continue
                    gens.add(_gen_from_bundles(fg.label, P, S))
        return Signature(frozenset(gens))


def extract_skeleton(
    g: LMGraph, kappa: Kappa | None = None, remove_silent: bool = True,
    *, surface_termini: bool = False,
) -> FactoredSignature:
    """LM-graph -> factored skeleton (the per-arc stop of the slow pipeline).

    Mirrors :func:`engine.extract_signature`'s parameters exactly; ``kappa`` is
    recorded on the result rather than applied here (see
    :class:`FactoredSignature`).  The default terminus strip commutes with the
    cross-arc union (``strip(b1 | b2) == strip(b1) | strip(b2)``), so it is
    applied per-arc here; the ``surface_termini`` pure-terminus collapse does
    not (it reads the whole side), so it is deferred via ``collapse_terminus``."""
    g, surface_termini = _prepare_extraction(g, remove_silent, surface_termini)
    fgens = []
    for a in sorted(g.activities):
        left = tuple(
            frozenset(_traverse(g, e.src, (e.typ,), frozenset(), forward=False))
            for e in g.in_edges(a)
        )
        raw = (
            _traverse(g, e.tgt, (e.typ,), frozenset(), forward=True)
            for e in g.out_edges(a)
        )
        if surface_termini:
            right = tuple(frozenset(f) for f in raw)
        else:
            right = tuple(frozenset(_strip_termini(f)) for f in raw)
        fgens.append(
            FactoredGenerator(g.lab(a), left, right, collapse_terminus=surface_termini)
        )
    return FactoredSignature(tuple(fgens), kappa=kappa)
