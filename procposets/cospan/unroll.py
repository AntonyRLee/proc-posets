"""Bounded grounding of a constrained signature (§38, the *practice* half).

The symbolic constrained ``Signature`` (``occn_to_signature`` + §33 constraints) is
the bound-independent comparison invariant. To *execute* it -- check feasibility,
exhibit witness runs, kill the §35 box+``n`` leak -- we **ground** it: per generator,
enumerate the concrete leg-multiplicity assignments its own constraints admit up to an
order ``k`` (capping ``cmax = *`` at ``k``), one **weighted** generator per assignment
(:attr:`Generator.weights`). The expanded signature then runs through the ordinary
:func:`cpm.cospan.compose.compose_signature`: node multiplicity emerges from port
accumulation (a 3-order bundle at ``s`` is fed by three ``b``-firings), and an
over-count consumer variant is simply **unreachable** when its tokens never accumulate
(the leak dies; see CLASS_EXTRACTION.md §38).

**Faithful per-firing system (``ground_occn``).** Grounding reads cardinalities off the
OCCN markers on *both* sides plus a within-firing **conservation** ``Σ out_ot == Σ in_ot``
per object type -- so each grounded generator conserves objects through one firing. This
differs from :func:`cpm.occn.to_signature.occn_to_signature`, which keeps only the
*consumer*-side interval (the §35 choice that avoided the convergent-wire contradiction
in the symbolic-global feasibility path). Grounding can carry both sides safely because
a producer's per-firing output (``b -> s`` = 1) and the consumer's bundle (``s`` input =
``[2,5]``) live on **different** generators and are reconciled by token accumulation, not
by equating on one wire.
"""
from __future__ import annotations

from ..occn.markers import OCCN, MarkerGroup
from .constraints import constraint, cset, interval
from .feasibility import all_solutions
from .signature import Generator, Port, Signature


def _types(group: MarkerGroup) -> frozenset[str]:
    return frozenset(m.otype for m in group)


def _type_balanced(ig: MarkerGroup, og: MarkerGroup) -> bool:
    if not ig or not og:
        return True
    return _types(ig) == _types(og)


def _in_port(t: str, m) -> Port:
    return Port(m.activity, m.otype, t)


def _out_port(t: str, m) -> Port:
    return Port(t, m.otype, m.activity)


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


def _boundary_generators(occn: OCCN) -> set[Generator]:
    """``START_<ot>``/``END_<ot>`` sources/sinks, weight 1 (one object per firing; a
    bundle of ``k`` objects enters as ``k`` firings, which token accumulation supplies).
    Mirrors :func:`cpm.occn.to_signature._boundary_generators`."""
    gens: set[Generator] = set()
    ocdg = occn.ocdg
    for otype, start_node in ocdg.starts.items():
        for src, ot, tgt in ocdg.arcs:
            if src == start_node and ot == otype:
                gens.add(Generator(start_node, frozenset(), frozenset({Port(start_node, otype, tgt)})))
    for otype, end_node in ocdg.ends.items():
        for src, ot, tgt in ocdg.arcs:
            if tgt == end_node and ot == otype:
                gens.add(Generator(end_node, frozenset({Port(src, otype, end_node)}), frozenset()))
    return gens


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
    Feed to :func:`cpm.cospan.extract_dp.extract_classes` (``one_origin=True``): γ1
    fires once to seed the frontier, weight-aware reachability runs to γ2-closure, and
    object contents with no closing path are pruned by reachability."""
    gens = _interior_generators(occn, order=order, max_assignments=max_assignments)
    gens |= gamma_boundary(occn, counts, order=order)
    return Signature(frozenset(gens))
