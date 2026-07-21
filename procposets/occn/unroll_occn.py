"""OCCN faithful per-firing grounding (§38-39) -- the OCCN half of unrolling.

Grounds a mined OCCN into a weighted, object-conserving cospan ``Signature``.
Reads cardinalities off the OCCN markers on *both* sides plus a within-firing
**conservation** ``Σ out_ot == Σ in_ot`` per object type, so each grounded
generator conserves objects through one firing.

This lives in ``occn`` (not ``cospan``) and imports *down* into
:mod:`procposets.cospan.unroll_core`, so the cospan algebra never depends on
the OCCN miner -- the dependency runs miner -> algebra, never the reverse.
"""
from __future__ import annotations

from ..cospan.constraints import constraint, cset, interval
from ..cospan.signature import Generator, Port, Signature
from ..cospan.unroll_core import unroll_generator
from ._lift import _boundary_generators, _in_port, _out_port, _type_balanced, _types
from .markers import OCCN, MarkerGroup


def _firing_system(t: str, ig: MarkerGroup, og: MarkerGroup) -> frozenset:
    """Per-firing constraint system for context ``(ig, og)`` (§38): every marker's
    cardinality as an interval on its leg, plus ``Σ out_ot == Σ in_ot`` for each type
    present on both sides (objects of a type are conserved through one firing -- this
    subsumes the shared-key partition, since same-key output legs sum to the typed
    input total)."""
    cons: list = []
    for m in ig:
        cons += interval(_in_port(t, m), m.cmin, m.cmax)
    for m in og:
        cons += interval(_out_port(t, m), m.cmin, m.cmax)
    for ot in _types(ig) & _types(og):
        coeffs: dict = {}
        for m in og:
            if m.otype == ot:
                p = _out_port(t, m)
                coeffs[p] = coeffs.get(p, 0) + 1
        for m in ig:
            if m.otype == ot:
                p = _in_port(t, m)
                coeffs[p] = coeffs.get(p, 0) - 1
        cons.append(constraint(coeffs, "==", 0))
    return cset(*cons)


def _interior_generators(occn: OCCN, *, order: int, max_assignments: int) -> set[Generator]:
    """The grounded interior activity generators (no boundary). Each context
    ``(P, S) ∈ I(t) × O(t)`` builds its faithful per-firing system
    (:func:`_firing_system`) and unrolls it up to ``order``."""
    gens: set[Generator] = set()
    activities = set(occn.input_groups) | set(occn.output_groups)
    for t in activities:
        in_groups = [grp for grp, _ in occn.input_groups.get(t, [])] or [frozenset()]
        out_groups = [grp for grp, _ in occn.output_groups.get(t, [])] or [frozenset()]
        for ig in in_groups:
            for og in out_groups:
                if not _type_balanced(ig, og):
                    continue
                left = frozenset(_in_port(t, m) for m in ig)
                right = frozenset(_out_port(t, m) for m in og)
                system = _firing_system(t, ig, og)
                base = Generator(t, left, right, system)
                gens |= unroll_generator(base, order=order, max_assignments=max_assignments)
    return gens


def ground_occn(occn: OCCN, *, order: int = 8, max_assignments: int = 200_000) -> Signature:
    """Mined OCCN -> grounded (weighted) signature (§38) with per-type
    ``START_<ot>``/``END_<ot>`` boundary generators. The result is already grounded and
    object-conserving. (For the γ1/γ2 boundary-interface reading, see :func:`ground_run`.)"""
    gens = _interior_generators(occn, order=order, max_assignments=max_assignments)
    gens |= _boundary_generators(occn)
    return Signature(frozenset(gens))


def gamma_boundary(occn: OCCN, counts: dict[str, int], *, order: int) -> set[Generator]:
    """The γ1/γ2 boundary of one *run* (§39): γ1 and γ2 are the boundary *objects* of
    the whole composite cospan, not activities -- the OCCN's per-type
    ``START_<ot>``/``END_<ot>`` collapse into this single interface, with per-type
    object multiplicity carried as **leg counts**.

    * **γ1** (a zero-left source, fired once by the existing zero-left rule) seeds the
      initial frontier: for each type it produces ``counts[ot]`` objects, partitioned
      across that type's start activities (e.g. ``order`` may begin at ``a`` *or* ``b``)
      -- a sum constraint unrolled into one γ1 variant per distribution.
    * **γ2** = per-end-arc **weight-1** sinks (re-fireable): they drain whatever
      *variable* terminal mix conservation produces (e.g. 3 orders ending as 1 via ``i``
      + 2 via ``n``), which a single fixed-weight γ2 could not.
    """
    ocdg = occn.ocdg
    right: set[Port] = set()
    cons: list = []
    cap = max(order, max(counts.values(), default=1))
    for ot, start_node in ocdg.starts.items():
        ports = [Port(s, o, t) for (s, o, t) in ocdg.arcs if s == start_node and o == ot]
        right |= set(ports)
        cons.append(constraint({p: 1 for p in ports}, "==", counts.get(ot, 0)))
    g1 = Generator("gamma1", frozenset(), frozenset(right), cset(*cons))
    gens = unroll_generator(g1, order=cap)
    for ot, end_node in ocdg.ends.items():
        for src, o, tgt in ocdg.arcs:
            if tgt == end_node and o == ot:
                gens.add(Generator(end_node, frozenset({Port(src, ot, end_node)}), frozenset()))
    return gens


def ground_run(occn: OCCN, counts: dict[str, int], *, order: int = 8, max_assignments: int = 200_000) -> Signature:
    """Grounded signature for one run with per-type object content ``counts`` (§39):
    the unrolled interior plus the γ1/γ2 boundary interface (:func:`gamma_boundary`).
    Feed to :func:`procposets.cospan.extract_dp.extract_classes` (``one_origin=True``):
    γ1 fires once to seed the frontier, weight-aware reachability runs to γ2-closure,
    and object contents with no closing path are pruned by reachability."""
    gens = _interior_generators(occn, order=order, max_assignments=max_assignments)
    gens |= gamma_boundary(occn, counts, order=order)
    return Signature(frozenset(gens))
