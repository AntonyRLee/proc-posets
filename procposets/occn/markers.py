"""M2 — Phase 2: marker-group discovery following the paper's Listing 1.1.

Decision D6: implement the *published* pseudocode (per-activity markers from the
event-ordering causal heuristic + same-type key unification + min/max
cardinalities), NOT the reference's set-partition key enumeration. We
therefore expect the marker *key labels* to
differ from the reference even when the marker *structure* agrees.

Pipeline (per the paper):
  1. event-ordering causal predecessors/successors, per object trace: for object
     o of type t in event e, its closest predecessor is the nearest earlier event
     e' in o's trace with arc (act(e'), t, act(e)) in D (else START_t if that arc
     exists). Successors are the mirror (nearest later event; else END_t).
  2. per event, per direction: bucket the shared objects by (other_activity,
     otype) -> object set; one marker per bucket with cardinality = |objects|.
  3. keys (Listing 1.1): markers of the SAME type whose object sets are disjoint
     are an object-distribution split -> share one key. A type with a single
     marker keeps a unique key (objects free). [On single-object-per-type logs
     no same-type split ever arises, so all keys are unique; see Finding F1.]
  4. aggregate over an activity's events: group marker groups by structure
     (the {(other_activity, otype)} set + the same-type key partition), widen
     (cmin, cmax) to the observed min/max, and record the occurrence count.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .fhm import OCDG, mine_ocdg


@dataclass(frozen=True)
class Marker:
    """One marker of a group: an arc to the NEIGHBOUR ``activity`` carrying
    ``otype`` objects, with cardinality ``[cmin, cmax]`` and object-distribution
    ``key``.  ``activity`` names the endpoint at the *other* end of the arc -- the
    marker's owner is the activity that keys it in the I/O map, not this field."""

    activity: str
    otype: str
    cmin: int
    cmax: int
    key: int


# a marker group is an immutable set of markers; I/O map activity -> [(group, count)]
MarkerGroup = frozenset[Marker]


@dataclass
class OCCN:
    ocdg: OCDG
    input_groups: dict[str, list[tuple[MarkerGroup, int]]]
    output_groups: dict[str, list[tuple[MarkerGroup, int]]]


# ---------------------------------------------------------------------------
# OCEL indexing
# ---------------------------------------------------------------------------

def _index(ocel):
    rel = ocel.relations.sort_values("ocel:timestamp", kind="stable")
    act_of = dict(zip(ocel.events["ocel:eid"], ocel.events["ocel:activity"]))
    otype_of: dict[str, str] = {}
    traces: dict[str, list[str]] = defaultdict(list)  # object -> [eid,...] in time order
    objs_of_event: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for eid, oid, ot in zip(rel["ocel:eid"], rel["ocel:oid"], rel["ocel:type"]):
        otype_of[oid] = ot
        traces[oid].append(eid)
        objs_of_event[eid][ot].add(oid)
    return act_of, otype_of, traces, objs_of_event


# ---------------------------------------------------------------------------
# event-ordering causal heuristic
# ---------------------------------------------------------------------------

def _closest(trace: list[str], pos: int, step: int, ok) -> str | None:
    """Walk ``trace`` from ``pos`` in direction ``step`` (+1/-1); return the first
    eid whose activity satisfies predicate ``ok``; else None."""
    i = pos + step
    while 0 <= i < len(trace):
        if ok(trace[i]):
            return trace[i]
        i += step
    return None


def _neighbour_buckets(ocel, ocdg: OCDG, direction: str):
    """For every event, bucket shared objects by (other_activity, otype).

    ``direction='in'``  -> predecessors (arc other->e); START_t fallback.
    ``direction='out'`` -> successors  (arc e->other);  END_t fallback.
    Returns ``eid -> {(other_activity, otype): set(objects)}``.
    """
    act_of, otype_of, traces, _ = _index(ocel)
    arcs = ocdg.arcs
    step = -1 if direction == "in" else +1
    buckets: dict[str, dict[tuple[str, str], set[str]]] = defaultdict(lambda: defaultdict(set))

    for oid, trace in traces.items():
        t = otype_of[oid]
        for pos, eid in enumerate(trace):
            a = act_of[eid]
            if direction == "in":
                ok = lambda e: (act_of[e], t, a) in arcs
            else:
                ok = lambda e: (a, t, act_of[e]) in arcs
            nb = _closest(trace, pos, step, ok)
            if nb is not None:
                other = act_of[nb]
            else:  # START/END fallback
                if direction == "in" and (ocdg.starts.get(t), t, a) in arcs:
                    other = ocdg.starts[t]
                elif direction == "out" and (a, t, ocdg.ends.get(t)) in arcs:
                    other = ocdg.ends[t]
                else:
                    continue
            buckets[eid][(other, t)].add(oid)
    return buckets, act_of


# ---------------------------------------------------------------------------
# per-event marker group + Listing 1.1 key unification
# ---------------------------------------------------------------------------

def _event_group(bucket: dict[tuple[str, str], set[str]]) -> MarkerGroup:
    # key unification: same type, disjoint object sets -> one shared key.
    # bucket keys partition objects by (activity,type), so same-type markers are
    # always disjoint -> a type with >1 marker is a distribution split (shared key).
    by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (act, ot) in bucket:
        by_type[ot].append((act, ot))

    key_of: dict[tuple[str, str], int] = {}
    next_key = 0
    for ot, members in by_type.items():
        if len(members) > 1:  # object-distribution split: shared key
            for m in members:
                key_of[m] = next_key
            next_key += 1
        else:  # unique key
            key_of[members[0]] = next_key
            next_key += 1

    markers = set()
    for (act, ot), objs in bucket.items():
        c = len(objs)
        markers.add(Marker(act, ot, c, c, key_of[(act, ot)]))
    return frozenset(markers)


def _aggregate(per_event: dict[str, MarkerGroup], act_of) -> dict[str, list[tuple[MarkerGroup, int]]]:
    """Group an activity's per-event marker groups by structure (arcs + same-type
    key partition), widening (cmin,cmax) and counting occurrences."""
    # collect raw groups per activity
    raw: dict[str, list[MarkerGroup]] = defaultdict(list)
    for eid, group in per_event.items():
        raw[act_of[eid]].append(group)

    out: dict[str, list[tuple[MarkerGroup, int]]] = {}
    for act, groups in raw.items():
        merged: dict[tuple, dict] = {}
        for group in groups:
            sig = _structure_sig(group)
            slot = merged.setdefault(sig, {"count": 0, "card": {}})
            slot["count"] += 1
            for m in group:
                lo, hi = slot["card"].get((m.activity, m.otype, m.key), (m.cmin, m.cmax))
                slot["card"][(m.activity, m.otype, m.key)] = (min(lo, m.cmin), max(hi, m.cmax))
        result = []
        for sig, slot in merged.items():
            grp = frozenset(
                Marker(a, ot, lo, hi, key) for (a, ot, key), (lo, hi) in slot["card"].items()
            )
            result.append((grp, slot["count"]))
        out[act] = sorted(result, key=lambda gc: (-gc[1], _structure_sig(gc[0])))
    return out


def _structure_sig(group: MarkerGroup):
    """Key-label-agnostic signature: the (activity,otype) set + which same-type
    markers share a key (the distribution partition). Two groups with the same
    signature are 'the same marker group' for aggregation/comparison."""
    arcs = tuple(sorted((m.activity, m.otype) for m in group))
    # partition of (activity,otype) by key, canonicalised (sorted blocks)
    blocks: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for m in group:
        blocks[m.key].append((m.activity, m.otype))
    partition = tuple(sorted(tuple(sorted(b)) for b in blocks.values()))
    return (arcs, partition)


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------

def mine_occn(ocel, **fhm_kwargs) -> OCCN:
    ocdg = mine_ocdg(ocel, **fhm_kwargs)
    in_buckets, act_of = _neighbour_buckets(ocel, ocdg, "in")
    out_buckets, _ = _neighbour_buckets(ocel, ocdg, "out")
    in_per_event = {eid: _event_group(b) for eid, b in in_buckets.items()}
    out_per_event = {eid: _event_group(b) for eid, b in out_buckets.items()}
    return OCCN(ocdg, _aggregate(in_per_event, act_of), _aggregate(out_per_event, act_of))


