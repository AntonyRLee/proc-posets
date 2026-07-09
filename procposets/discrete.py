"""Discrete / weight-free comparison of models, and its relation to the block-SMD.

A *model* here is a set of labelled posets over a common activity alphabet (its variants), optionally
weighted. This module gives two families of discrete comparison and the three candidate weightings
that turn the (continuous) block-SMD into a weight-free number, so we can measure how they relate.

Kemeny family (compare the ORDER RELATIONS)
-------------------------------------------
- ``kemeny(P, Q)``: the classical partial-order metric = size of the symmetric difference of the two
  strict order relations (over a common vertex/label set). Hard, combinatorial.
- ``order_angle(m1, m2)``: an information-geometry lift of Kemeny. Each unordered label pair {x,y}
  gets a 3-outcome categorical over {x<y, y<x, x||y}, weighted across the model's variants; the
  distance is the Bhattacharyya angle summed over pairs, in the SAME form and units as the block-SMD
  (Eq. smd). On single total orders it equals pi*sqrt(K/2) with K the Kemeny distance, mirroring the
  block-SMD's pi*sqrt(H) on deterministic block models.

Block family (compare the TILING / block transitions) -- the three weightings
-----------------------------------------------------------------------------
- ``"uniform_variant"``  rho_i = 1/r  (option 1): uniform over the variants.
- ``"support"``          (option 2): keep only WHICH block transitions are possible, spread mass
                         uniformly over each row's support -- the max-entropy, structure-only reading.
- ``"lincount"``         rho_i ∝ e(P_i) (option 3): the weighting under which the pooled event log is
                         uniform over traces (each variant weighted by its number of linear extensions).

All three feed the block-SMD; comparing them and the Kemeny family is the point of demo/08_discrete_comparison.
"""
from __future__ import annotations

import math
from itertools import combinations

from .distance import _augment
from .matrix import END, START, build
from .moddecomp import Prime, Series, decompose
from .poset import Poset
from .poset import count_extensions
from .traces import linear_extensions

# --------------------------------------------------------------------------- Kemeny family


def _label_relation(P: Poset):
    """Map the (transitively closed) strict order to ordered LABEL pairs. Requires distinct labels."""
    labs = list(P.labels.values())
    if len(set(labs)) != len(labs):
        raise ValueError("Kemeny/precedence need distinct labels within a poset (common vertex set).")
    return {(P.labels[u], P.labels[v]) for (u, v) in P.less}


def kemeny(P: Poset, Q: Poset) -> int:
    """Classical partial-order (Kemeny) distance = |R_P symmetric-difference R_Q| over label pairs."""
    rP, rQ = _label_relation(P), _label_relation(Q)
    return len(rP ^ rQ)


def _alphabet(model):
    V = set()
    for P, _ in model:
        V |= set(P.labels.values())
    return sorted(V)


def _pair_relation(P: Poset, x: str, y: str) -> str:
    """'lt' if x<y, 'gt' if y<x, else 'par' (incomparable), by label in a distinct-label poset."""
    inv = {lab: e for e, lab in P.labels.items()}
    ex, ey = inv[x], inv[y]
    if (ex, ey) in P.less:
        return "lt"
    if (ey, ex) in P.less:
        return "gt"
    return "par"


def precedence(model):
    """For each unordered label pair, the rho-weighted categorical over {lt, gt, par}."""
    V = _alphabet(model)
    tot = sum(w for _, w in model) or 1.0
    out = {}
    for x, y in combinations(V, 2):
        cat = {"lt": 0.0, "gt": 0.0, "par": 0.0}
        for P, w in model:
            cat[_pair_relation(P, x, y)] += w / tot
        out[(x, y)] = cat
    return out


def order_angle(m1, m2):
    """Bhattacharyya angle between the two models' pairwise-precedence footprints (the 'Kemeny angle').
    Same form as the block-SMD, over the common label pairs. Requires a common alphabet."""
    if _alphabet(m1) != _alphabet(m2):
        raise ValueError("order_angle needs a common alphabet (common vertex set).")
    p1, p2 = precedence(m1), precedence(m2)
    total = 0.0
    per = {}
    for pair in p1:
        bc = sum(math.sqrt(p1[pair][o] * p2[pair][o]) for o in ("lt", "gt", "par"))
        ang = math.acos(min(1.0, max(0.0, bc)))
        per[pair] = ang
        total += ang * ang
    return 2.0 * math.sqrt(total), per


# --------------------------------------------------------------------------- block family (3 weightings)


def _reweight(model, weighting: str):
    if weighting == "uniform_variant":
        return [(P, 1.0) for P, _ in model]
    if weighting == "lincount":
        return [(P, float(count_extensions(P))) for P, _ in model]
    if weighting in ("support", "given"):
        return list(model)
    raise ValueError(f"unknown weighting {weighting!r}")


def _support_uniformise(row):
    keys = [k for k, v in row.items() if v > 0]
    if not keys:
        return dict(row)
    p = 1.0 / len(keys)
    return {k: p for k in keys}


def _matrix_angle(a1, a2, states):
    total = 0.0
    per = {}
    for s in states:
        bc = sum(math.sqrt(a1[s].get(t, 0.0) * a2[s].get(t, 0.0)) for t in states)
        ang = math.acos(min(1.0, max(0.0, bc)))
        per[s] = ang
        total += ang * ang
    return 2.0 * math.sqrt(total), per


def block_angle(m1, m2, weighting="uniform_variant", context_depth=1):
    """Block-SMD between two models under one of the three weightings (option 1/2/3).

    For 'support' the row frequencies are discarded: each row is made uniform over its support on the
    common state space (the max-entropy, structure-only reading, option 2)."""
    b1, s1 = build(_reweight(m1, weighting), context_depth)
    b2, s2 = build(_reweight(m2, weighting), context_depth)
    states = sorted(s1 | s2)
    a1, a2 = _augment(b1, states), _augment(b2, states)
    if weighting == "support":
        a1 = {s: _support_uniformise(r) for s, r in a1.items()}
        a2 = {s: _support_uniformise(r) for s, r in a2.items()}
    return _matrix_angle(a1, a2, states)


# ------------------------------------------------------ prime-gradation hybrid (fan-out)
# Prime blocks are atomic in the block alphabet, so two distinct primes are all-or-nothing (P3, the
# 'coarser inside a prime' limitation). The fan-out refinement makes the transition INTO a prime spread
# uniformly over the prime's Hasse-edge relation-atoms (as intermediate states) before reconverging on
# the block that follows. Two primes then SHARE the relation-atoms they have in common, so the ordinary
# row-Bhattacharyya grades them by relation overlap -- relation-aware inside primes, tiling-aware on the
# series-parallel part (which is untouched). Distinct labels within a prime are assumed (v1 caveat).


def _covers(P: Poset):
    return {(u, v) for (u, v) in P.less
            if not any((u, w) in P.less and (w, v) in P.less for w in P.elements)}


def _prime_atoms(prime: Prime):
    """The Hasse-edge relation-atoms 'x<y' of a prime block (its shared/graded pieces)."""
    P = prime.poset
    return sorted(f"{P.labels[u]}<{P.labels[v]}" for (u, v) in _covers(P))


def _block_items(tree):
    """A normal form as a list of items: ('block', symbol) or ('prime', [relation-atoms])."""
    def item(node):
        if isinstance(node, Prime):
            return ("prime", _prime_atoms(node))
        return ("block", node.canonical())
    return [item(c) for c in tree.parts] if isinstance(tree, Series) else [item(tree)]


def _build_refined(model, refine: bool):
    """Depth-1 block matrix in which primes are fanned out over their relation-atoms (if refine)."""
    raw: dict = {}
    states = {START, END}

    def add(s, t, w):
        raw.setdefault(s, {}).setdefault(t, 0.0)
        raw[s][t] += w

    for P, w in model:
        frontier = {START: float(w)}                       # active source states -> mass
        for kind, payload in _block_items(decompose(P)):
            total = sum(frontier.values())
            if kind == "block" or not refine:
                sym = payload if kind == "block" else "(" + "*".join(payload) + ")"  # atomic prime
                for s, m in frontier.items():
                    add(s, sym, m)
                states.add(sym)
                frontier = {sym: total}
            else:                                          # prime fan-out over relation-atoms
                atoms = payload or ["<empty>"]
                for s, m in frontier.items():
                    for a in atoms:
                        add(s, a, m / len(atoms))
                states.update(atoms)
                frontier = {a: total / len(atoms) for a in atoms}
        for s, m in frontier.items():
            add(s, END, m)
    matrix = {s: {t: v / sum(row.values()) for t, v in row.items()} for s, row in raw.items()}
    return matrix, states


def disc_angle(m1, m2, refine=True):
    """Discrete block-SMD with the prime fan-out on (refine=True) or off (refine=False = atomic primes)."""
    b1, s1 = _build_refined(m1, refine)
    b2, s2 = _build_refined(m2, refine)
    states = sorted(s1 | s2)
    return _matrix_angle(_augment(b1, states), _augment(b2, states), states)
