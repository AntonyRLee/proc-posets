"""M1 — Phase 1: hand-rolled Flexible Heuristics Miner -> typed OCDG.

Follows the *paper's* stated method (Weijters & Ribeiro, "Flexible Heuristics
Miner", 2011) rather than pm4py's ``discover_heuristics_net`` — which is what the
reference actually uses. We mine only the
dependency relation D (typed arcs ``(src, otype, tgt)``) plus per-type START/END
nodes; AND/XOR/OR splits are recovered downstream by the Listing-1.1 marker miner
(M2), so FHM AND-measures are intentionally not computed (decision D8).

Measures (per object type, on that type's flattened traces):
  direct succession  |a>b|         = # adjacent (a,b)
  length-2 loop      |a>>b|        = # of pattern  a b a   (b between two a's)
  dependency  a==b:  |a>a|/(|a>a|+1)                                  [L1 loop]
              a!=b:  (|a>b|-|b>a|)/(|a>b|+|b>a|+1)
  l2 dep             (|a>>b|+|b>>a|)/(|a>>b|+|b>>a|+1)                 [L2 loop]

Acceptance (decision D8 = threshold + best-edge connectedness):
  - self-loop  if dep(a,a) >= L1T
  - L2 loop pair (a,b) if l2dep >= L2T and |a>b|>0
  - any (a,b) with dep(a,b) >= DT and |a>b|>0
  - best-edge: every activity keeps its best successor & best predecessor (and
    any within ``rel_to_best`` of it) provided the measure is positive, so no
    reachable task is orphaned.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class OCDG:
    """Object-centric dependency graph: typed arcs + per-type start/end nodes."""

    activities: frozenset[str]
    otypes: tuple[str, ...]
    arcs: frozenset[tuple[str, str, str]]  # (src, otype, tgt)
    starts: dict[str, str]  # otype -> "START_<otype>"
    ends: dict[str, str]  # otype -> "END_<otype>"


def flatten(ocel) -> dict[str, dict[str, list[str]]]:
    """``otype -> {object_id -> [activity,...] ordered by timestamp}``."""
    rel = ocel.relations.sort_values("ocel:timestamp", kind="stable")
    out: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for oid, grp in rel.groupby("ocel:oid", sort=False):
        otype = grp["ocel:type"].iloc[0]
        out[otype][oid] = list(grp["ocel:activity"])
    return out


def succession_counts(traces: list[list[str]]):
    direct: Counter = Counter()
    l2: Counter = Counter()
    starts: Counter = Counter()
    ends: Counter = Counter()
    for seq in traces:
        if not seq:
            continue
        starts[seq[0]] += 1
        ends[seq[-1]] += 1
        for i in range(len(seq) - 1):
            direct[(seq[i], seq[i + 1])] += 1
        for i in range(len(seq) - 2):
            if seq[i] == seq[i + 2] and seq[i] != seq[i + 1]:
                l2[(seq[i], seq[i + 1])] += 1  # pattern a b a  -> a>>b
    return direct, l2, starts, ends


def dependency(direct: Counter, a: str, b: str) -> float:
    ab, ba = direct[(a, b)], direct[(b, a)]
    if a == b:
        return ab / (ab + 1)
    return (ab - ba) / (ab + ba + 1)


def l2_dependency(l2: Counter, a: str, b: str) -> float:
    ab, ba = l2[(a, b)], l2[(b, a)]
    return (ab + ba) / (ab + ba + 1) if (ab + ba) else 0.0


def accept_arcs(
    activities: set[str],
    direct: Counter,
    l2: Counter,
    dt: float,
    l1t: float,
    l2t: float,
    rel_to_best: float = 0.05,
) -> set[tuple[str, str]]:
    arcs: set[tuple[str, str]] = set()

    for a in activities:  # L1 self-loops
        if dependency(direct, a, a) >= l1t:
            arcs.add((a, a))

    for a in activities:  # L2 loops
        for b in activities:
            if a != b and direct[(a, b)] > 0 and l2_dependency(l2, a, b) >= l2t:
                arcs.add((a, b))

    for a in activities:  # threshold deps
        for b in activities:
            if a != b and direct[(a, b)] > 0 and dependency(direct, a, b) >= dt:
                arcs.add((a, b))

    # best-edge connectedness (no orphaned reachable task)
    for a in activities:
        succ = [(b, dependency(direct, a, b)) for b in activities if b != a and direct[(a, b)] > 0]
        if succ:
            best = max(s for _, s in succ)
            if best > 0:
                arcs.update((a, b) for b, s in succ if s >= best - rel_to_best)
    for b in activities:
        pred = [(a, dependency(direct, a, b)) for a in activities if a != b and direct[(a, b)] > 0]
        if pred:
            best = max(s for _, s in pred)
            if best > 0:
                arcs.update((a, b) for a, s in pred if s >= best - rel_to_best)

    return arcs


def mine_ocdg(
    ocel,
    dependency_threshold: float = 0.5,
    loop1_threshold: float = 0.5,
    loop2_threshold: float = 0.5,
    rel_to_best: float = 0.05,
) -> OCDG:
    flat = flatten(ocel)
    otypes = tuple(flat.keys())
    arcs: set[tuple[str, str, str]] = set()
    activities: set[str] = set()
    starts: dict[str, str] = {}
    ends: dict[str, str] = {}

    for otype, objs in flat.items():
        traces = list(objs.values())
        acts = {a for seq in traces for a in seq}
        activities |= acts
        direct, l2, st, en = succession_counts(traces)
        for a, b in accept_arcs(acts, direct, l2, dependency_threshold, loop1_threshold, loop2_threshold, rel_to_best):
            arcs.add((a, otype, b))
        start_node, end_node = f"START_{otype}", f"END_{otype}"
        starts[otype], ends[otype] = start_node, end_node
        for a in st:
            arcs.add((start_node, otype, a))
        for a in en:
            arcs.add((a, otype, end_node))

    return OCDG(frozenset(activities), otypes, frozenset(arcs), starts, ends)
