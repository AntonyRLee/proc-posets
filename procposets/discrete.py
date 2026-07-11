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

Refined family (fan-out) -- ``disc_angle``
------------------------------------------
The paper's Remark V.1: prime blocks fan out over covering-relation atoms, parallel blocks over
typed element atoms, uniform splits, the SMD unchanged. Everything else in this package stays
atomic (the paper's default). See the comment block above ``disc_angle`` for the conventions,
the closed form, and the defaults; pins in ``tests/test_refinement.py``.
"""
from __future__ import annotations

import math
from itertools import combinations

from .distance import _augment
from .matrix import END, START, build
from .moddecomp import Parallel, Prime, Series, decompose
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


# ------------------------------------------------------ the refined family (fan-out)
# Atomic blocks are all-or-nothing: two distinct primes, or two distinct parallel blocks, differ
# maximally however much structure they share. The fan-out refinement (paper, Remark V.1) replaces
# the transition INTO such a block by a uniform spread over the block's atoms as TYPED intermediate
# states, reconverging on the block that follows. Two blocks then share the atoms they have in
# common, and the ordinary row-Bhattacharyya grades them by overlap; the SMD formula, the sink
# convention, and the series-parallel part of the comparison are untouched.
#
# Conventions (each the paper's declared choice, stated where it acts):
#   * atoms per block type -- PRIME: labelled covering (Hasse) relations "x<y", the poset's unique
#     irredundant generating set; PARALLEL: the child modules' canonical symbols, typed "sym||"
#     (an antichain has no covering relations, so its gradable content is its element multiset);
#   * mass split -- uniform over the atoms: the same maximum-entropy convention as the
#     uniform-linear-extension law and the uniform-prior weighting;
#   * typed states -- a relation atom "x<y" or an element atom "x||" can never collide with a
#     block symbol, so refining cannot merge a concurrency with its interleavings;
#   * the distance is unchanged -- the refined object feeds the same row-Bhattacharyya SMD, with
#     the same sink-and-reset closure on the union state space.
# Closed form for an isolated pair of same-kind blocks with atom multisets A, A':
#     d = 2*arccos(|A & A'| / sqrt(|A| * |A'|)),
# the angular (Ochiai) set similarity. Between totally parallel models this coincides with the
# Bhattacharyya angle on the uniform activity distributions (the activity-marginal comparison).
#
# DEFAULTS: everything else in this package is atomic -- `matrix.build`, `distance.smd`, and
# `block_angle` never refine (the paper's default comparison object). `disc_angle` is the refined
# family's entry point: refine=True enables both instantiations, refine={"prime"} / {"parallel"}
# one of them, refine=False none (atomic). Depth-1 only; distinct labels within a prime assumed
# (v1 caveat).
#
# RECURSIVE FAN-OUT (E1, paper App C "Outlook"; recursive=True, default False -- adoption pending
# the demo-08 pre-adoption checks in the paper repo's docs/TODO.md): a refined block's atom
# multiset is augmented with the DISCLOSED CONTENT of every composite descendant module, so shared
# nested content is credited at every depth. The disclosure of a module is again the canonical
# content of the paper's statement (iii), one level down:
#   * series module   -> its consecutive covering relations "sym_i<sym_{i+1}" over its parts'
#     canonical symbols (the transitive reduction of a chain of modules);
#   * parallel module -> its parts' canonical symbols as typed element atoms "sym||";
#   * prime module    -> its labelled covering relations (as at the top level).
# Level-1 atoms are unchanged, so any block whose children are all leaves (every flat prime and
# flat parallel block) has EXACTLY the same atoms as recursive=False: the existing validated
# numbers are invariant, and only nested blocks gain gradation. Membership credit for the leaves
# of a nested series (the E2 k=1 rung) is deliberately NOT included: a chain's irredundant content
# is its covers. Relation atoms share one type across nesting (a disclosed series cover "c<d" may
# meet a prime cover "c<d": both assert the same immediate precedence); element atoms keep the
# "||" type. The uniform split, the closed form over the (enlarged) multisets, the sink
# convention, and the SMD formula are untouched -- each disclosure level is again a dilation.


def _covers(P: Poset):
    return {(u, v) for (u, v) in P.less
            if not any((u, w) in P.less and (w, v) in P.less for w in P.elements)}


def _prime_atoms(prime: Prime):
    """The labelled covering-relation atoms 'x<y' of a prime block (its shared/graded pieces)."""
    P = prime.poset
    return sorted(f"{P.labels[u]}<{P.labels[v]}" for (u, v) in _covers(P))


def _parallel_atoms(block: Parallel):
    """The typed element atoms 'sym||' of a parallel block: its child modules' canonical symbols."""
    return sorted(f"{c.canonical()}||" for c in block.parts)


def _module_content(node):
    """One level of a composite module's canonical content (the E1 disclosure of `node` itself):
    consecutive covers for a series module, typed element symbols for a parallel module, labelled
    covers for a prime. A leaf has no content."""
    if isinstance(node, Parallel):
        return [f"{c.canonical()}||" for c in node.parts]
    if isinstance(node, Series):
        syms = [c.canonical() for c in node.parts]
        return [f"{syms[i]}<{syms[i + 1]}" for i in range(len(syms) - 1)]
    if isinstance(node, Prime):
        return _prime_atoms(node)
    return []


def _disclosure(node):
    """Flattened E1 disclosure below `node`: the content of every composite descendant module,
    multiset-preserving (duplicated children disclose duplicated atoms)."""
    out: list = []
    for c in node.children:
        sub = _module_content(c)
        if sub:
            out += sub + _disclosure(c)
    return out


def _block_items(tree, recursive: bool = False):
    """A normal form as (kind, atoms, atomic_symbol) items, kind in {'block','prime','parallel'}.
    With recursive=True the atoms of a refined block are augmented by its E1 disclosure; blocks
    whose children are all leaves are unchanged (flat invariance)."""
    def item(node):
        if isinstance(node, Prime):
            atoms = _prime_atoms(node) + (sorted(_disclosure(node)) if recursive else [])
            return ("prime", atoms, node.canonical())
        if isinstance(node, Parallel):
            atoms = _parallel_atoms(node) + (sorted(_disclosure(node)) if recursive else [])
            return ("parallel", atoms, node.canonical())
        return ("block", None, node.canonical())
    return [item(c) for c in tree.parts] if isinstance(tree, Series) else [item(tree)]


_REFINE_KINDS = frozenset({"prime", "parallel"})


def _normalise_refine(refine):
    if refine is True:
        return set(_REFINE_KINDS)
    if not refine:
        return set()
    kinds = {refine} if isinstance(refine, str) else set(refine)
    unknown = kinds - _REFINE_KINDS
    if unknown:
        raise ValueError(f"unknown refine kinds {sorted(unknown)}; allowed: 'prime', 'parallel'")
    return kinds


def _build_refined(model, kinds: set, context_depth: int = 1, strict: bool = True,
                   recursive: bool = False):
    """Block matrix in which blocks of the selected `kinds` fan out over their atoms.

    `recursive=True` enables the E1 recursive fan-out: refined blocks spread over their atoms
    PLUS the disclosed content of every composite descendant module (see the comment block above);
    flat blocks are unchanged.

    `context_depth` types every state by its preceding block context, mirroring matrix.build's
    windows: an unrefined block at depth k occupies the '|'-joined window of the last <=k block
    symbols; a fan-out atom is typed by the last <=k-1 PRECEDING block symbols plus the atom name
    (at k=1 the context is empty, so atoms are global -- shared across blocks and models, the
    maximally graded reading). `strict=True` (the default) enforces EXACTNESS: if any state recurs
    within a single variant at the chosen depth -- the occurrence collision that merges rows and
    admits spurious trajectories -- a ValueError names the state; raise context_depth for the
    faithful chain, or pass strict=False to accept the merge as the declared robustness relaxation
    (the refined analogue of lowering the memory depth in the atomic chain)."""
    k = max(1, context_depth)
    raw: dict = {}
    states = {START, END}

    def add(s, t, w):
        raw.setdefault(s, {}).setdefault(t, 0.0)
        raw[s][t] += w

    for P, w in model:
        frontier = {START: float(w)}                       # active source states -> mass
        visited: set = set()                               # states this variant has occupied
        ctx: list = []                                     # block symbols emitted so far
        for kind, atoms, sym in _block_items(decompose(P), recursive):
            total = sum(frontier.values())
            prefix = ctx[-(k - 1):] if k > 1 else []
            if kind == "block" or kind not in kinds:       # atomic step: standard window state
                win = "|".join((ctx + [sym])[-k:])
                targets = {win: 1.0}                       # each source sends all its mass
                new_frontier = {win: total}
            else:                                          # fan-out over context-typed atoms
                atoms = atoms or ["<empty>"]
                targets = {}
                for a in atoms:                            # multiset-safe: duplicates accumulate
                    st = "|".join(prefix + [a])
                    targets[st] = targets.get(st, 0.0) + 1.0 / len(atoms)
                new_frontier = {st: f * total for st, f in targets.items()}
            hit = set(new_frontier) & visited
            if hit and strict:
                raise ValueError(
                    f"state {sorted(hit)[0]!r} recurs within one variant at "
                    f"context_depth={k}: rows would merge and the chain would admit spurious "
                    "trajectories. Increase context_depth (the faithful default), or pass "
                    "strict=False to accept the merge as a declared robustness relaxation.")
            visited |= set(new_frontier)
            for s, m in frontier.items():
                for st, f in targets.items():
                    add(s, st, m * f)
            states.update(new_frontier)
            frontier = new_frontier
            ctx.append(sym)
        for s, m in frontier.items():
            add(s, END, m)
    matrix = {s: {t: v / sum(row.values()) for t, v in row.items()} for s, row in raw.items()}
    return matrix, states


def disc_angle(m1, m2, refine=True, context_depth=1, strict=True, recursive=False):
    """Block-SMD with the fan-out refinement of the paper's Remark V.1.

    refine=True        -- the full refined family: primes fan out over covering-relation atoms,
                          parallel blocks over typed element atoms.
    refine={"prime"} or {"parallel"} -- one instantiation only ({"prime"} reproduces the earlier
                          prime-only behaviour exactly).
    refine=False       -- atomic blocks throughout (the paper's default comparison).
    context_depth      -- memory depth for the refined states (atoms typed by their preceding
                          block context); pick the faithful depth for the models compared.
    strict=True        -- exact by default: refuse to merge recurring states within a variant
                          (ValueError); strict=False accepts the merge (declared relaxation).
    recursive=False    -- E1 recursive fan-out (paper App C "Outlook"): refined blocks also
                          disclose the content of composite descendant modules, so shared nested
                          content is graded; flat blocks are bit-identical to recursive=False.
                          Off by default pending the pre-adoption checks (docs/TODO.md).
    The SMD row formula, the sink-and-reset closure, and the union state space are identical in
    every mode; only the state space changes. Isolated same-kind block pairs obey the closed form
    2*arccos(sum_x sqrt(m(x) m'(x)) / sqrt(|A||A'|)) over their atom multisets with multiplicities
    m; for equal shared multiplicities this is 2*arccos(|A & A'|/sqrt(|A||A'|)); under
    recursive=True the same closed form holds over the disclosure-enlarged multisets."""
    kinds = _normalise_refine(refine)
    b1, s1 = _build_refined(m1, kinds, context_depth, strict, recursive)
    b2, s2 = _build_refined(m2, kinds, context_depth, strict, recursive)
    states = sorted(s1 | s2)
    return _matrix_angle(_augment(b1, states), _augment(b2, states), states)
