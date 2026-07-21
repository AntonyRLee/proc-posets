"""OCCN -> cospan generator Signature (the deferred D1 adapter).

Maps a mined :class:`markers.OCCN` into the repo's
``procposets.cospan.signature.Signature`` (the same datatype the PN/PT/BPMN/CN
pipeline produces), so OCCN joins the cross-notation comparison.

Construction (matches the paper's §4 OCCN -> cospan and ``RUNNING_EXAMPLE.md``):
a generator cospan ``g_{t,(P,S)}`` exists for activity ``t`` and each valid
context ``(P, S)`` drawn from ``I(t) x O(t)``. We take left ports from an input
marker group and right ports from an output marker group, keeping only
**type-balanced** pairs (same set of object types on both sides — the interior
binding preserves object types; an event's input and output carry the same
objects, paper Def. 6). A marker ``(other, otype, ...)`` becomes:

    input  marker of t  ->  Port(other, otype, t)     (left boundary)
    output marker of t  ->  Port(t, otype, other)      (right boundary)

**Bindings surfaced as constraints (§32).** A ``Port`` is a bare typed triple (the
wire identity); the boundary is a **multiset** whose leg multiplicities ``n_p`` are
governed by linear constraints on ``Generator.constraints``. The marker's
**cardinality** ``(cmin,cmax)`` becomes an interval ``cmin <= n_p <= cmax``; a
**shared-key distribution** becomes a partition equality (the distributed legs sum
to the input total of that type). A 1-1 unkeyed marker adds no constraint (the leg
defaults to count 1). So the object-XOR a shared key encodes is recorded explicitly
as a ``Σ = T`` partition, not flattened into anonymous parallel ports.
"""
from __future__ import annotations

from collections import defaultdict

from ..cospan.constraints import constraint, cset, interval
from ..cospan.signature import Generator, Port, Signature
from .markers import OCCN, MarkerGroup


def _types(group: MarkerGroup) -> frozenset[str]:
    return frozenset(m.otype for m in group)


def _type_balanced(ig: MarkerGroup, og: MarkerGroup) -> bool:
    """Interior activities preserve object types -> same type set both sides.
    A missing input or output side (a source/sink context) is unconstrained."""
    if not ig or not og:
        return True
    return _types(ig) == _types(og)


def _boundary_generators(occn: OCCN) -> set[Generator]:
    """Generators for the synthetic ``START_<ot>``/``END_<ot>`` nodes.

    Activities' own marker groups already reference these nodes (e.g.
    ``gamma1``'s input markers include ``(START_img, img, ...)`` whenever
    ``gamma1`` is the first ``img``-typed event in a trace, via the
    ``fhm.py`` START/END fallback) -- but ``START_<ot>``/``END_<ot>`` are
    never themselves keys of ``input_groups``/``output_groups`` (they're
    not real OCEL events), so the main loop below never emits a generator
    for them, leaving every activity downstream of a start unreachable from
    the empty frontier. One generator per outgoing/incoming arc, not one
    combined generator per type: ``START_<ot>`` may have several possible
    first activities across different traces, and those are alternatives
    (an object starts at exactly one of them), not a simultaneous bundle --
    mirrors how real activities get one generator per alternative marker
    group, never one generator unioning every alternative's ports.
    """
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


def _in_port(t: str, m) -> Port:
    return Port(m.activity, m.otype, t)


def _out_port(t: str, m) -> Port:
    return Port(t, m.otype, m.activity)


def _leg_constraints(t: str, ig: MarkerGroup, og: MarkerGroup) -> frozenset:
    """Surface the markers' bindings (§32) as linear constraints on the legs --
    the generator's full **per-leg N-linear blueprint** (§42):

    * each **input** marker's cardinality ``[cmin,cmax]`` -> an interval on its
      input leg (the objects ``t`` consumes on that arc per firing).
    * each **output** marker's cardinality ``[cmin,cmax]`` -> an interval on its
      output leg (the objects ``t`` emits on that arc per firing).
    * a **shared-key output partition** (>=2 output markers of an otype sharing a
      key) -> the distributed successor legs sum to the input total of that otype
      (object conservation): ``Σ out_legs == Σ in_legs``. This is the ``r -> {i,n}``
      split, ``n_i + n_n == n_(orders in)``.

    Both sides are kept because a generator cospan is an **independent morphism**
    parameterised by its own legs (the "blueprint"): the input interval and the
    mirror output interval describe *different* per-firing quantities and live on
    *different* generators. They are only reconciled when the cospans are composed --
    :func:`procposets.cospan.constraints.union` intersects the shared ``Port`` identity,
    which **is** the pushout. A convergent wire whose producer per-firing fan-out and
    consumer per-firing intake disagree is therefore pruned (or grows node
    multiplicity, §36) *at composition/grounding*, not pre-resolved here. Earlier this
    adapter dropped the output side ("consumer-only"); that was a premature
    optimisation that hid the discovered model's real output cardinalities from
    ``signature.json``/``cospans.svg``/the comparison (§42). The behavioural splice
    is unaffected (it is built from the plain signature) and grounding reads the
    model markers directly, so this enrichment is representation + comparison only.
    """
    cons: list = []
    for m in ig:
        cons += interval(_in_port(t, m), m.cmin, m.cmax)
    for m in og:
        cons += interval(_out_port(t, m), m.cmin, m.cmax)

    by_key: dict = defaultdict(list)
    for m in og:
        by_key[(m.otype, m.key)].append(m)
    for (otype, _key), ms in by_key.items():
        if len(ms) < 2:
            continue  # unique key -> independent leg, no partition
        in_legs = [_in_port(t, im) for im in ig if im.otype == otype]
        if not in_legs:
            continue  # nothing of that type comes in -> no total to partition
        coeffs: dict = {_out_port(t, m): 1 for m in ms}
        for q in in_legs:
            coeffs[q] = coeffs.get(q, 0) - 1
        cons.append(constraint(coeffs, "==", 0))
    return cset(*cons)


def occn_to_signature(occn: OCCN, *, bindings: bool = True) -> Signature:
    """Mined OCCN -> generator-cospan signature.

    ``bindings`` (default ``True``) surfaces the OCCN's multi-object machinery as §32
    constraints on the generator legs: each marker's cardinality ``[cmin,cmax]`` as an
    interval, each shared-key distribution as a partition equality. Pass
    ``bindings=False`` for the **plain** signature -- every leg ``1-1`` and no key
    constraints -- i.e. the forgetful typed-causal-net reading (the CLI default; see
    the consumer's ``cpm.signature_cli``)."""
    gens: set[Generator] = set()
    activities = set(occn.input_groups) | set(occn.output_groups)
    for t in activities:
        in_groups = [g for g, _ in occn.input_groups.get(t, [])] or [frozenset()]
        out_groups = [g for g, _ in occn.output_groups.get(t, [])] or [frozenset()]
        for ig in in_groups:
            for og in out_groups:
                if not _type_balanced(ig, og):
                    continue
                left = frozenset(_in_port(t, m) for m in ig)
                right = frozenset(_out_port(t, m) for m in og)
                cons = _leg_constraints(t, ig, og) if bindings else frozenset()
                gens.add(Generator(t, left, right, cons))
    gens |= _boundary_generators(occn)
    return Signature(frozenset(gens))
