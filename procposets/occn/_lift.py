"""Shared OCCN -> cospan lift primitives.

The type/port helpers and the ``START_<ot>``/``END_<ot>`` boundary-generator
scaffold used by **both** :mod:`procposets.occn.to_signature` (the plain D1
adapter) and :mod:`procposets.occn.unroll_occn` (the faithful per-firing
grounding). Only the type-neutral primitives live here; the two consumers'
constraint *tails* stay in their own modules -- ``to_signature`` surfaces
per-key partition equalities (:func:`~procposets.occn.to_signature._leg_constraints`),
``unroll_occn`` per-type conservation (:func:`~procposets.occn.unroll_occn._firing_system`).

This module imports only *down* into :mod:`procposets.cospan.signature`, keeping
the miner -> algebra dependency direction intact.
"""
from __future__ import annotations

from ..cospan.signature import Generator, Port
from .markers import OCCN, MarkerGroup


def _types(group: MarkerGroup) -> frozenset[str]:
    return frozenset(m.otype for m in group)


def _type_balanced(ig: MarkerGroup, og: MarkerGroup) -> bool:
    """Interior activities preserve object types -> same type set both sides.
    A missing input or output side (a source/sink context) is unconstrained."""
    if not ig or not og:
        return True
    return _types(ig) == _types(og)


def _in_port(t: str, m) -> Port:
    return Port(m.activity, m.otype, t)


def _out_port(t: str, m) -> Port:
    return Port(t, m.otype, m.activity)


def _boundary_generators(occn: OCCN) -> set[Generator]:
    """Generators for the synthetic ``START_<ot>``/``END_<ot>`` nodes.

    Activities' own marker groups already reference these nodes (e.g.
    ``gamma1``'s input markers include ``(START_img, img, ...)`` whenever
    ``gamma1`` is the first ``img``-typed event in a trace, via the
    ``fhm.py`` START/END fallback) -- but ``START_<ot>``/``END_<ot>`` are
    never themselves keys of ``input_groups``/``output_groups`` (they're
    not real OCEL events), so the main lift loops never emit a generator
    for them, leaving every activity downstream of a start unreachable from
    the empty frontier. One generator per outgoing/incoming arc, not one
    combined generator per type: ``START_<ot>`` may have several possible
    first activities across different traces, and those are alternatives
    (an object starts at exactly one of them), not a simultaneous bundle --
    mirrors how real activities get one generator per alternative marker
    group, never one generator unioning every alternative's ports. Each is
    weight 1 (one object per firing; a bundle of ``k`` objects enters as
    ``k`` firings, which token accumulation supplies).
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
