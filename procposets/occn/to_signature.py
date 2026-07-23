"""OCCN -> cospan generator Signature (the deferred D1 adapter).

Maps a mined :class:`markers.OCCN` into the repo's
``procposets.cospan.signature.Signature`` (the same datatype the PN/PT/BPMN/CN
pipeline produces), so OCCN joins the cross-notation comparison.

Construction (matches the paper's §4 OCCN -> cospan):
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
from ..cospan.signature import Generator, Signature
from ._lift import _boundary_generators, _in_port, _out_port, _type_balanced
from .markers import OCCN, MarkerGroup


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


def _generators_with_counts(
    occn: OCCN, bindings: bool
) -> dict[Generator, tuple[int | None, int | None]]:
    """The interior generator set with each generator's observed binding counts
    ``(in_count, out_count)`` -- the observation counts of the input/output
    marker groups it was built from (``None`` for a missing side)."""
    out: dict[Generator, tuple[int | None, int | None]] = {}
    activities = set(occn.input_groups) | set(occn.output_groups)
    for t in activities:
        in_groups = occn.input_groups.get(t, []) or [(frozenset(), None)]
        out_groups = occn.output_groups.get(t, []) or [(frozenset(), None)]
        for ig, ic in in_groups:
            for og, oc in out_groups:
                if not _type_balanced(ig, og):
                    continue
                left = frozenset(_in_port(t, m) for m in ig)
                right = frozenset(_out_port(t, m) for m in og)
                cons = _leg_constraints(t, ig, og) if bindings else frozenset()
                out[Generator(t, left, right, cons)] = (ic, oc)
    return out


def occn_to_signature(occn: OCCN, *, bindings: bool = True) -> Signature:
    """Mined OCCN -> generator-cospan signature.

    ``bindings`` (default ``True``) surfaces the OCCN's multi-object machinery as §32
    constraints on the generator legs: each marker's cardinality ``[cmin,cmax]`` as an
    interval, each shared-key distribution as a partition equality. Pass
    ``bindings=False`` for the **plain** signature -- every leg ``1-1`` and no key
    constraints -- i.e. the forgetful typed-causal-net reading (the CLI default).

    To thin rare co-firing structures first, filter the model:
    ``occn_to_signature(occn.filtered(min_rel=...))`` (see :meth:`OCCN.filtered`);
    per-generator observation counts come from :func:`occn_generator_counts`."""
    gens = set(_generators_with_counts(occn, bindings))
    gens |= _boundary_generators(occn)
    return Signature(frozenset(gens))


def occn_generator_counts(
    occn: OCCN, *, bindings: bool = True
) -> dict[Generator, tuple[int | None, int | None]]:
    """Map each interior generator of ``occn_to_signature(occn, bindings=...)``
    to its observed binding counts ``(in_count, out_count)``. Boundary
    ``START_``/``END_`` generators are not included (they are synthetic, not
    observed marker groups). Built from the SAME construction as
    :func:`occn_to_signature`, so the keys align exactly."""
    return _generators_with_counts(occn, bindings)
