"""Faithful cospan -> OCEL generation.

:mod:`.simulate` threads exactly **one object per type per run** -- it ignores
every N-linear leg constraint, so a master with a ``[1,*]`` bundle or a ``[1,5]``
batch simulates degenerately to 1-1 (which is *why* a multi-object log has to be
hand-built). This module closes that gap: it plays the master signature as an
**object-centric token game** whose multiplicities, batching and key-splits are
read straight off the leg constraints. The generated OCEL is therefore a
*function of the master cospan* -- one source of truth -- with the only freedom a
seeded RNG sampling concrete counts inside the admitted ranges.

What it reads off each generator (label = activity):
  * per input leg ``[lo,hi]`` -> objects consumed **per firing**; ``hi`` finite and < the
    objects waiting => the activity **batches** (fires ``ceil(n/hi)`` times, e.g. an
    ``[1,5]`` batch); ``hi = *`` => it consumes the whole **bundle** in one firing.
  * a **partition** ``Σ out_legs == in_leg`` (the shared-key object split) -> the consumed
    objects are *routed* (conserved identities) over the out-legs: a leg pinned by an
    explicit single-leg constraint takes that many (``exactly(i)==1``), the unpinned leg
    is the **remainder** (``n``).
  * ``gamma1`` (no inputs) is the **source** -- it seeds each run's objects; ``gamma2``
    (no outputs) is the **sink** -- it drains them. Neither becomes an OCEL event (they are
    the boundary; discovery re-introduces START/END).

Falls back to the plain 1-1 path when the signature carries no real multiplicity (so
pure-1-1 masters are unaffected) -- see :func:`needs_faithful` and the dispatch in
:func:`.simulate.ocel_from_signature`.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from datetime import datetime, timedelta

from pm4py.objects.ocel.obj import OCEL

from ._ocel import _ocel_from_rows
from .signature import Generator, Port, Signature

_BASE = datetime(2026, 1, 1)
_STEP = timedelta(minutes=5)

BOUNDARY = ("gamma1", "gamma2")


def _wire(p: Port) -> tuple:
    """The wire a port names: ``(producer, type, consumer)`` -- shared identity across the
    producing generator's right leg and the consuming generator's left leg."""
    return (p.src, p.typ, p.tgt)


def _leg_card(g: Generator) -> tuple[dict, set]:
    """``(card, explicit)``: ``card[port] = (lo, hi)`` (``hi`` may be ``inf``) folding the
    single-leg unit-coefficient constraints; ``explicit`` = the ports that carry such a
    constraint (so a partition can tell a *pinned* out-leg from the *remainder* leg whose
    only constraint is the partition itself)."""
    raw: dict = {}
    for c in g.constraints:
        terms = list(c.terms)
        if len(terms) != 1 or terms[0][1] != 1:
            continue
        p = terms[0][0]
        lo, hi = raw.get(p, [None, None])
        if c.rel in (">=", "=="):
            lo = c.rhs if lo is None else max(lo, c.rhs)
        if c.rel in ("<=", "=="):
            hi = c.rhs if hi is None else min(hi, c.rhs)
        raw[p] = [lo, hi]
    card: dict = {}
    for p in g.left | g.right:
        if p in raw:
            lo, hi = raw[p]
            card[p] = (1 if lo is None else lo, math.inf if hi is None else hi)
        else:
            card[p] = (1, 1)
    return card, set(raw)


def _partitions(g: Generator) -> list[tuple]:
    """The generator's object-conservation partitions, as ``(typ, in_ports, out_ports)``:
    a ``Σ(+out) − Σ(in) == 0`` constraint where in/out are this generator's own legs."""
    parts: list[tuple] = []
    for c in g.constraints:
        if c.rel != "==" or c.rhs != 0 or len(c.terms) < 2:
            continue
        outs = [p for p, co in c.terms if co > 0 and p in g.right]
        ins = [p for p, co in c.terms if co < 0 and p in g.left]
        if not outs or not ins:
            continue
        typ = outs[0].typ
        parts.append((typ, ins, outs))
    return parts


class _Model:
    """The master signature parsed into the facts the token game needs."""

    def __init__(self, sig: Signature):
        self.gens = sorted(sig.generators, key=str)
        self.by_label: dict[str, list[Generator]] = defaultdict(list)
        for g in self.gens:
            self.by_label[g.label].append(g)
        self.card: dict = {}
        self.explicit: dict = {}
        self.parts: dict = {}
        for g in self.gens:
            self.card[g], self.explicit[g] = _leg_card(g)
            self.parts[g] = _partitions(g)

        # gamma1 source legs by type (orders fan over route alternatives; carriers are fixed)
        self.src_out: dict[str | None, list[Port]] = defaultdict(list)
        for g in self.by_label.get("gamma1", []):
            for p in g.right:
                self.src_out[p.typ].append(p)

        # convergence = a generator consuming >=2 object types (s); the primary (bulk) type
        # is the one with >=2 source routes (orders split a/skip), the rest are carriers.
        self.primary: str | None = max(self.src_out, key=lambda t: len(self.src_out[t])) if self.src_out else None
        self.carriers = [t for t in self.src_out if t != self.primary]

        self.topo = self._topo()

    def _topo(self) -> list[str]:
        import networkx as nx

        g = nx.DiGraph()
        for gen in self.gens:
            g.add_node(gen.label)
            for p in gen.left | gen.right:
                g.add_edge(p.src, p.tgt)
        order = [n for n in nx.topological_sort(g) if n not in BOUNDARY]
        return order

    def k_range(self, carrier: str) -> tuple[int, float]:
        """The primary-type count a run with this carrier admits = the convergence
        generator's primary input interval (``s`` container order ``[1,*]`` / box ``[1,1]``)."""
        for g in self.gens:
            if any(p.typ == carrier for p in g.left) and any(p.typ == self.primary for p in g.left):
                for p in g.left:
                    if p.typ == self.primary:
                        return self.card[g][p]
        return (1, 1)


def needs_faithful(sig: Signature) -> bool:
    """True iff some leg admits a count != 1 (interval ``hi>1``/``*`` or a partition split) --
    i.e. a 1-1 simulation would be lossy. Pure-1-1 signatures report False."""
    for g in sig.generators:
        card, _ = _leg_card(g)
        if any(hi > 1 for _lo, hi in card.values()):
            return True
        if any(len(outs) >= 2 for _t, _ins, outs in _partitions(g)):
            return True
    return False


def _route(g: Generator, model: _Model, consumed: dict, pool: dict) -> None:
    """Forward the consumed objects (identities conserved) onto ``g``'s output wires: a
    partition splits them (pinned legs first, remainder to the unpinned leg); otherwise a
    single same-type out-leg carries all of them."""
    part_by_type = {t: outs for t, _ins, outs in model.parts[g]}
    out_by_type: dict[str | None, list[Port]] = defaultdict(list)
    for p in g.right:
        out_by_type[p.typ].append(p)
    for typ, objs in consumed.items():
        outs = part_by_type.get(typ) or out_by_type.get(typ, [])
        objs = list(objs)
        if not outs:
            continue
        if len(outs) == 1:
            pool[_wire(outs[0])].extend(objs)
            continue
        remainder: list = []
        pinned = []
        open_legs = []
        for p in sorted(outs, key=str):
            lo, hi = model.card[g][p]
            (pinned if (p in model.explicit[g] and lo == hi) else open_legs).append((p, lo))
        for p, lo in pinned:
            take = min(lo, len(objs))
            pool[_wire(p)].extend(objs[:take])
            objs = objs[take:]
        remainder = objs
        if open_legs:
            tgt = open_legs[0][0]
            pool[_wire(tgt)].extend(remainder)
        elif pinned:  # no open leg: pile remainder on the last pinned (degenerate)
            pool[_wire(pinned[-1][0])].extend(remainder)


def _fire(g: Generator, model: _Model, pool: dict, events: list) -> None:
    """Fire ``g`` repeatedly while its inputs are available: each firing consumes up to
    ``hi`` per input leg (``*`` => the whole bundle), emits one event for the consumed
    objects, and routes them forward. Finite ``hi`` < waiting => batching."""
    while True:
        if not all(len(pool[_wire(p)]) >= model.card[g][p][0] for p in g.left):
            break
        if g.left and all(len(pool[_wire(p)]) == 0 for p in g.left):
            break
        consumed: dict = defaultdict(list)
        ok = True
        take_by_port: dict = {}
        for p in g.left:
            lo, hi = model.card[g][p]
            avail = len(pool[_wire(p)])
            if avail < lo:
                ok = False
                break
            take_by_port[p] = avail if hi == math.inf else min(hi, avail)
        if not ok:
            break
        for p, take in take_by_port.items():
            wk = _wire(p)
            for _ in range(take):
                consumed[p.typ].append(pool[wk].pop(0))
        events.append((g.label, {t: list(v) for t, v in consumed.items()}))
        _route(g, model, consumed, pool)


def _gen_run(model: _Model, rng: random.Random, rid: int, bound: int) -> list[tuple]:
    """One process execution (object group): seed a carrier + sampled primary objects via
    ``gamma1``, then fire the interior activities in topological order."""
    pool: dict = defaultdict(list)
    events: list[tuple] = []
    carrier = rng.choice(model.carriers) if model.carriers else None
    if carrier is not None:
        lo, hi = model.k_range(carrier)
        hi_eff = bound if hi == math.inf else int(hi)
        k = rng.randint(int(lo), max(int(lo), hi_eff))
        cport = model.src_out[carrier][0]
        pool[_wire(cport)].append(f"{carrier}_{rid}")
    else:
        k = 1
    routes = sorted(model.src_out.get(model.primary, []), key=str)
    for j in range(k):
        oid = f"{model.primary}_{rid}_{j}"
        port = rng.choice(routes) if routes else None
        if port is not None:
            pool[_wire(port)].append(oid)
    for label in model.topo:
        for g in model.by_label[label]:
            _fire(g, model, pool, events)
    return events


def faithful_ocel_from_signature(
    sig: Signature, *, n_runs: int = 100, seed: int = 7, bound: int | None = None,
) -> OCEL:
    """Generate an OCEL by playing ``sig`` as an object-centric token game (see module
    docstring). ``n_runs`` process executions, each a seeded random carrier + primary-count
    sample inside the leg ranges. ``bound`` caps an unbounded ``*`` primary count (default:
    the largest finite batch cap downstream + 2, so the cap-exceeding cases that exercise
    batching are observed)."""
    model = _Model(sig)
    if bound is None:
        caps = [hi for g in model.gens for _lo, hi in model.card[g].values()
                if hi != math.inf and hi > 1]
        bound = (max(caps) + 2) if caps else 5
    rng = random.Random(seed)

    rows: list[tuple] = []
    eid = 0
    for rid in range(n_runs):
        base = _BASE + timedelta(hours=rid * 6)
        for i, (activity, objs) in enumerate(_gen_run(model, rng, rid, bound)):
            ts = base + _STEP * (i + 1)
            e = f"e{eid}"
            eid += 1
            for ot in sorted(objs):
                for oid in objs[ot]:
                    rows.append((e, activity, ts, oid, ot))

    return _ocel_from_rows(rows)
