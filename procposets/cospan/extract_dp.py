"""Pomset-DP morphism extraction: the iso-class-quotiented replacement for the
path-enumerating ``class_extraction.extract_classes``.

Design: ``CLASS_EXTRACTION.md`` §21f. The old DFS enumerated morphisms of
``F(Sigma)`` as *linear firing sequences* (words over generators) and deduped
by raw tuple -- so the exponentially-many interleavings of independent
concurrent generators were all minted separately (the OCPN blowup). But a
morphism of a free hypergraph category *is* an iso-class of decorated cospans
(Fong--Spivak) -- exactly the boundary-rooted occurrence-net DAG and
``occurrence.canonical_key`` built in §19/§20. So the right unit of
computation is the **iso-class**, deduped *during* generation, not the word.

Structure:

* **State** = ``(frontier, used_labels)`` -- the multiset of pending typed
  ports, plus which zero-left origin *labels* (``gamma1``/``START_*``) have
  already fired. **One origin per label** (§40a): same-label zero-left
  generators are routing variants of one start transition (XOR, fire once);
  distinct-label origins (OCCN's per-type ``START_<ot>``) still co-fire -- one
  process instance per closing. The Markov property (§2) holds on the state.
* **Frontier-state graph** explored once, then condensed into SCCs (Tarjan). A
  non-trivial SCC is a loop region. Keying origins on the *label*, not the
  generator, is what keeps the state count tractable: otherwise a discovered
  net's many same-label routing-variant origins all co-fire and the frontier
  explodes (an OCPN's 6 ``gamma1`` variants -> 141k states before this rule).
* **Closing DP**: ``closing_pomsets(s)`` = iso-classes of *loop-free* pomsets
  from ``s`` to a closed (empty-frontier) state, deduped by ``canonical_key``.
  Memoised on DAG states (Markov); SCC-internal states are recomputed with a
  path-visited set so loop back-edges are not traversed.
* **Loops**: a back-edge into a state on the current path is a loop; its body
  pomset is recorded at its anchor frontier, deduped by iso.

Output is the same ``ExtractionResult`` shape, but its fragments are now
genuine iso-classes (the "unique string diagrams" used as comparison building
blocks), not raw-word representatives.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass

from .class_extraction import (
    ExtractionResult,
    FrontierKey,
    NamedMorphism,
    TooManyFrontiers,
    _enabled,
    _fire,
    _flatten_generators,
    _merge_parallel,
    _sort_key,
    _to_counter,
    _to_key,
)
from .discovery_cleanup import close_gamma2_termini
from .occurrence import canonical_key, is_isomorphic, to_event_dag
from .signature import Generator, Signature

# A search state: the pending-port frontier plus the set of zero-left origin
# *labels* already fired on the way here (each label may fire only once -- the
# per-label single-origin rule; see _enabled_gens).
State = tuple  # (FrontierKey, frozenset[str])  -- str = origin label


def _touched_boundary(frontier_key: FrontierKey, body_gens: tuple) -> FrontierKey:
    """Project a loop's cycle frontier to the ports its body actually **touches**.

    A frontier token that no loop generator consumes or produces is a *passenger*: the loop
    is ``f ⊗ id_passenger``, so the passenger belongs to the ambient monoidal context, not to
    the loop's own anchor. Keeping it in the boundary would (i) split one loop into a distinct
    copy per passenger multiplicity (an accumulator riding through a clean round → one "loop"
    per budget level), and (ii) leave the passenger-free anchor a *subset* of every richer
    frontier, so it ⊇-bleeds into splice sites it never really anchored. Restricting to the
    touched sub-frontier recovers the categorical ``f`` (passenger-independence)."""
    touched = {p for gen in body_gens for p in (gen.left | gen.right)}
    return frozenset((p, c) for (p, c) in frontier_key if p in touched)


def _enabled_gens(
    sig_gens: list[Generator],
    frontier: Counter,
    used_labels: frozenset,
    *,
    one_origin: bool = False,
) -> list[Generator]:
    out = []
    for g in sig_gens:
        if not g.left:
            if g.label in used_labels:
                continue  # one origin PER LABEL: a zero-left generator of this
                # label already fired. Same-label zero-left generators are routing
                # variants of one start transition (XOR -- fire one); distinct-label
                # origins (e.g. OCCN's per-type START_<ot>) still co-fire. One
                # process instance per closing, without forbidding multi-type starts.
            if one_origin and used_labels:
                continue  # stricter opt-in: collapse ALL origins to a single firing
                # (global single-origin, across labels too). Rarely needed now that
                # per-label is the default; kept for the force-one-instance case.
        if _enabled(frontier, g):
            out.append(g)
    return out


def _step(state: State, g: Generator) -> State:
    frontier_key, used = state
    nf = _fire(_to_counter(frontier_key), g)
    new_used = used | {g.label} if not g.left else used
    return (_to_key(nf), new_used)


def _is_closed(state: State) -> bool:
    """A state is a closing target iff its frontier is empty and at least one
    generator has fired (so the bare root state is not itself closed)."""
    frontier_key, used = state
    return len(frontier_key) == 0 and len(used) > 0


def _compress_parallel(body: tuple, frontier_states: list) -> tuple:
    """`@`-merge a representative linear body for display (purely cosmetic --
    iso-dedup already happened on ``canonical_key``)."""
    fr = [_to_counter(s[0]) for s in frontier_states]
    return tuple(_merge_parallel(fr, list(body)))


def _replay_frontiers(root: State, body: tuple) -> list:
    """The frontier-state-before-each-step list ``_merge_parallel`` needs."""
    states = []
    s = root
    for g in body:
        states.append(s)
        s = _step(s, g)
    return states


@dataclass
class _Graph:
    states: list[State]
    index_of: dict[State, int]
    succ: dict[State, list[tuple]]  # state -> [(g, next_state)]


def _explore(sig_gens: list[Generator], root: State, max_states: int, *, one_origin: bool = False) -> _Graph:
    """Build the reachable frontier-state graph from ``root`` (worklist)."""
    index_of: dict[State, int] = {root: 0}
    states: list[State] = [root]
    succ: dict[State, list[tuple]] = {}
    stack = [root]
    while stack:
        s = stack.pop()
        if s in succ:
            continue
        frontier = _to_counter(s[0])
        edges = []
        if not _is_closed(s):
            for g in _enabled_gens(sig_gens, frontier, s[1], one_origin=one_origin):
                ns = _step(s, g)
                edges.append((g, ns))
                if ns not in index_of:
                    if len(states) >= max_states:
                        raise TooManyFrontiers(f"exceeded {max_states} states")
                    index_of[ns] = len(states)
                    states.append(ns)
                    stack.append(ns)
        succ[s] = edges
    return _Graph(states=states, index_of=index_of, succ=succ)


def _tarjan_scc(graph: _Graph) -> dict[State, int]:
    """Tarjan's SCC; returns state -> scc-id. Iterative (state graphs can be
    ~1000s deep)."""
    idx: dict[State, int] = {}
    low: dict[State, int] = {}
    on_stk: dict[State, bool] = {}
    scc_of: dict[State, int] = {}
    stk: list[State] = []
    counter = [0]
    scc_id = [0]

    for root in graph.states:
        if root in idx:
            continue
        work = [(root, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                idx[v] = low[v] = counter[0]
                counter[0] += 1
                stk.append(v)
                on_stk[v] = True
            recurse = False
            edges = graph.succ.get(v, [])
            i = pi
            while i < len(edges):
                w = edges[i][1]
                if w not in idx:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    recurse = True
                    break
                elif on_stk.get(w):
                    low[v] = min(low[v], idx[w])
                i += 1
            if recurse:
                continue
            if low[v] == idx[v]:
                while True:
                    w = stk.pop()
                    on_stk[w] = False
                    scc_of[w] = scc_id[0]
                    if w == v:
                        break
                scc_id[0] += 1
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    return scc_of


def extract_classes(
    sig: Signature, *, max_frontiers: int = 200_000, max_pomsets_per_state: int = 512,
    one_origin: bool = False,
) -> ExtractionResult:
    """Pomset-DP extraction (§21f): the iso-class-quotiented catalogue.

    Returns the same ``ExtractionResult`` shape as the legacy extractor, but
    every fragment is a genuine occurrence-net iso-class (deduped by
    ``occurrence.canonical_key`` + VF2), so concurrent interleavings collapse
    to one morphism and scenario-dead generators never appear.

    ``max_pomsets_per_state`` caps the loop-free closing iso-classes kept *per
    frontier state*. Well-formed models stay far under it (full exact
    catalogue); a signature whose execution space is a genuine free product
    (e.g. an OCPN of independent per-type subnets, §22) hits the cap, which
    bounds the work and sets ``ExtractionResult.truncated`` -- the "this net
    over-generates" signal. Loops are extracted in full regardless (they are
    the factored generating structure used for comparison)."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    # a terminating object (the discovered OCPN's ``s`` carrier -> bare final place, surfaced
    # as a ``(…, gamma2)`` right leg by ``engine._traverse``) drains only through a consuming
    # generator; a bare sink has none, so without this the frontier never empties (0
    # closings). No-op when the terminus is already consumed (master ``gamma2`` gens, OCCN
    # ``END_``). gamma2 drains are zero-right -> not single-firing-constrained (§40a is gamma1
    # only), so several may co-drain in one closing.
    sig = close_gamma2_termini(sig)
    sig_gens = sorted(sig.generators, key=_sort_key)
    root: State = (frozenset(), frozenset())

    graph = _explore(sig_gens, root, max_frontiers, one_origin=one_origin)
    scc_of = _tarjan_scc(graph)

    fragments: dict[str, NamedMorphism] = {}
    closings_by_key: dict[tuple, NamedMorphism] = {}  # (wl, canonical) bucket -> rep
    loops_by_key: dict[tuple, NamedMorphism] = {}
    name_counts = {"M": 0, "L": 0}

    def _mint(prefix: str, boundary: FrontierKey, body: tuple, registry: dict) -> NamedMorphism:
        dag = to_event_dag(NamedMorphism("", boundary, body), {})
        wl = canonical_key(dag)
        for rep in registry.get(wl, []):
            if rep[0].boundary == boundary and is_isomorphic(
                to_event_dag(rep[0], {}), dag
            ):
                return rep[0]
        name_counts[prefix] += 1
        nm = NamedMorphism(f"{prefix}{name_counts[prefix]}", boundary, body)
        registry.setdefault(wl, []).append((nm, dag))
        fragments[nm.name] = nm
        return nm

    # Memo for the closing DP. Keyed on (state, visited-within-its-own-SCC):
    # back-edges only target states in the same SCC, so that intersection is
    # the *only* path context a state's loop-free closings depend on. DAG
    # (trivial-SCC) states therefore key on (state, {}) -- fully Markov-memoised
    # -- while SCC-internal states get one memo entry per distinct blocked set,
    # bounded by the (small) SCC, instead of being recomputed once per path.
    memo: dict[tuple, list[tuple]] = {}
    path: list[State] = []  # states on the current recursion (for loop bodies)
    path_gens: list = []
    path_index: dict[State, int] = {}  # state -> position on path (O(1) back-edge)
    scc_path: dict[int, frozenset] = {}  # scc-id -> frozenset of path states in it
    truncated = [False]  # mutable flag: set True iff any state hits max_pomsets_per_state

    def closing_pomsets(state: State) -> list[tuple]:
        """Iso-classes of loop-free pomsets from ``state`` to a closed state.
        Each returned element: (canonical_key, EventDag-of-body-from-state, body)."""
        if _is_closed(state):
            return [("", None, ())]
        sid = scc_of[state]
        blocked = scc_path.get(sid, frozenset())
        memo_key = (state, blocked)
        cached = memo.get(memo_key)
        if cached is not None:
            return cached

        path_index[state] = len(path)
        path.append(state)
        path_gens.append(None)
        prev_blocked = scc_path.get(sid, frozenset())
        scc_path[sid] = prev_blocked | {state}
        reps: list[tuple] = []  # (wl, dag, body)
        bucket: dict[str, list] = {}
        for g, ns in graph.succ.get(state, []):
            # back-edge into the current path => a loop; record, do not traverse
            k = path_index.get(ns)
            if k is not None:
                loop_body = tuple(path_gens[k:-1]) + (g,)
                # anchor on the touched sub-frontier, not the whole cycle frontier: a
                # passenger token (e.g. an accumulator budget riding a clean round) is
                # `id` alongside the loop and must not split or ⊇-bleed it.
                boundary = _touched_boundary(ns[0], loop_body)
                _mint("L", boundary, _compress_parallel(loop_body, path[k:]), loops_by_key)
                continue
            path_gens[-1] = g
            for (_, _, sub_body) in closing_pomsets(ns):
                body = (g,) + sub_body
                dag = to_event_dag(NamedMorphism("", state[0], body), {})
                wl = canonical_key(dag)
                if any(is_isomorphic(d, dag) for (_, d, _) in bucket.get(wl, [])):
                    continue
                entry = (wl, dag, body)
                bucket.setdefault(wl, []).append(entry)
                reps.append(entry)
                if len(reps) >= max_pomsets_per_state:  # over-generation guard (§22): cap the
                    truncated[0] = True                  # loop-free closings kept per frontier
                    break                                # state; well-formed models stay far under
            if truncated[0]:
                break
        path.pop()
        path_gens.pop()
        del path_index[state]
        scc_path[sid] = prev_blocked
        memo[memo_key] = reps
        return reps

    closing_reps = closing_pomsets(root)
    for (_, _, body) in closing_reps:
        merged = _compress_parallel(body, _replay_frontiers(root, body))
        _mint("M", frozenset(), merged, closings_by_key)

    valid: set[Generator] = set()
    for nm in fragments.values():
        if nm.is_closing():
            valid |= _flatten_generators(nm.body, fragments)

    return ExtractionResult(
        fragments=fragments, valid_generators=valid, frontiers_visited=len(graph.states),
        truncated=truncated[0],
    )
