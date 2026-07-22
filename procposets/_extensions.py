"""The single guarded ideal-lattice engine for linear-extension counting and
uniform sampling.

Moved here verbatim from the NPMLE `Rel` toolkit so there is exactly one
implementation, shared by both poset views:

- ``rel.py`` (the distinct-label ``Rel = frozenset[(str, str)]`` view) and
- ``poset.py`` (the canonical id+label ``Poset``, whose ``less`` is a set of
  ``(int, int)`` pairs).

Both call these functions -- the element type is irrelevant, only the order
*pairs* matter.  The guard (``MAX_IDEAL_STATES`` + the polynomial chain-cover
precheck) is the only thing standing between the meet-closure oracle and an
unbounded hang on a wide poset (DESIGN_REVIEW W12.1), so it travels *with* the
counter, not around it.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Tuple

# Hard ceiling on the ideal-lattice DP's state count.  The DP is exponential
# only in poset width; the meet-closure oracle runs at *any* m, where meets of
# a few long chains are typically wide -- without a guard this is the one
# place the library can hang or exhaust memory with no printed verdict
# (DESIGN_REVIEW W12.1).  The bound below is certified before recursing.
MAX_IDEAL_STATES = 1_000_000


class IdealBudgetExceeded(ValueError):
    """The ideal-lattice DP would exceed MAX_IDEAL_STATES states.

    Raised *before* any exponential work happens.  The oracle catches this,
    skips the offending candidate, and downgrades the certificate loudly."""


def preds(elements: Iterable, pairs) -> Dict:
    """predecessors[e] = {a : a < e}, for the order given by ``pairs``."""
    p: Dict = {e: set() for e in elements}
    for a, b in pairs:
        p[b].add(a)
    return {e: frozenset(s) for e, s in p.items()}


def ideal_state_bound(elements, pairs, max_states: int = MAX_IDEAL_STATES) -> int:
    """Upper bound on the number of order ideals, via a greedy chain cover.

    Every ideal meets each chain of a chain cover in a prefix, so
    #ideals <= prod_i (|c_i| + 1) for ANY chain cover -- a sound (if not
    minimal) budget check.  Chains are peeled greedily longest-first by DP
    on the DAG; the whole check is polynomial.  ``max_states`` is the
    threshold at which the running product short-circuits (the caller's
    knob -- ``rel.MAX_IDEAL_STATES`` / the module default).
    """
    remaining = set(elements)
    # successor adjacency built ONCE (was rebuilt from `pairs` on every peel);
    # per-node order follows the `pairs` iteration order exactly as before, so the
    # greedy chain choice -- hence the integer bound -- is byte-identical.
    adj: Dict = {e: [] for e in remaining}
    for a, b in pairs:
        if a in adj and b in adj:
            adj[a].append(b)
    bound = 1
    while remaining:
        succs = {e: [b for b in adj[e] if b in remaining] for e in remaining}
        # longest-path DP, iterative post-order (was a recursion that hit
        # RecursionError on a chain longer than the interpreter's limit). Same
        # first-argmax tie-break (cand > ln), so best[] and the peeled chain match.
        best: Dict = {}
        for root in remaining:
            if root in best:
                continue
            stack = [(root, False)]
            while stack:
                x, done = stack.pop()
                if done:
                    ln, nxt = 1, None
                    for y in succs[x]:
                        cand = 1 + best[y][0]
                        if cand > ln:
                            ln, nxt = cand, y
                    best[x] = (ln, nxt)
                elif x not in best:
                    stack.append((x, True))
                    for y in succs[x]:
                        if y not in best:
                            stack.append((y, False))

        top = max(remaining, key=lambda x: best[x][0])
        chain = [top]
        while best[chain[-1]][1] is not None:
            chain.append(best[chain[-1]][1])
        bound *= len(chain) + 1
        remaining -= set(chain)
        if bound > max_states:
            return bound
    return bound


def check_ideal_budget(elements, pairs, max_states: int = MAX_IDEAL_STATES) -> None:
    bound = ideal_state_bound(elements, pairs, max_states)
    if bound > max_states:
        raise IdealBudgetExceeded(
            f"ideal-lattice DP refused: chain-cover bound of {bound:.2e} "
            f"states exceeds the budget = {max_states:.0e} "
            f"(poset too wide for exact e(P) on this budget)"
        )


def count_extensions(elements, pairs, max_states: int = MAX_IDEAL_STATES) -> int:
    """e(P) for an arbitrary partial order, by DP over the ideal lattice.

    e(remaining) = sum over minimal x of remaining of e(remaining - x).
    The memo has one entry per order ideal -- at most 2^m, and typically far
    fewer; this is exact for every poset, with cost exponential only in the
    width.  A chain-cover bound on the ideal count is certified against
    ``max_states`` before recursing; a too-wide poset raises
    IdealBudgetExceeded instead of hanging (DESIGN_REVIEW W12.1).
    """
    check_ideal_budget(elements, pairs, max_states)
    pr = preds(elements, pairs)
    memo: Dict[FrozenSet, int] = {frozenset(): 1}

    def rec(rem: FrozenSet) -> int:
        got = memo.get(rem)
        if got is not None:
            return got
        out = sum(rec(rem - {x}) for x in rem if not (pr[x] & rem))
        memo[rem] = out
        return out

    return rec(frozenset(elements))


def sample_extension_poset(elements, pairs, rng, max_states: int = MAX_IDEAL_STATES) -> Tuple:
    """Uniform linear extension of an arbitrary partial order (the ideal-DP engine).

    Sequential sampling with the ideal-lattice DP as the exact proposal:
    the first element is x (minimal) with probability e(P - x) / e(P), which
    telescopes to 1/e(P) for every completed extension.  Guarded by the same
    ideal-state budget as count_extensions.
    """
    check_ideal_budget(elements, pairs, max_states)
    pr = preds(elements, pairs)
    memo: Dict[FrozenSet, int] = {frozenset(): 1}

    def count(rem: FrozenSet) -> int:
        got = memo.get(rem)
        if got is not None:
            return got
        out = sum(count(rem - {x}) for x in rem if not (pr[x] & rem))
        memo[rem] = out
        return out

    out: List = []
    rem = frozenset(elements)
    while rem:
        mins = sorted(x for x in rem if not (pr[x] & rem))
        weights = [count(rem - {x}) for x in mins]
        total = sum(weights)
        r = rng.random() * total
        acc = 0
        for x, w in zip(mins, weights):
            acc += w
            if r < acc:
                break
        out.append(x)
        rem = rem - {x}
    return tuple(out)
