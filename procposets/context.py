"""Canonical context tree — the lossless VLMC state space of a model or a fleet.

``matrix.build`` uses a UNIFORM window: the state after emitting blocks b_0..b_i is always the
last ``context_depth`` blocks, whether or not the histories support that merge. That is an
order-k chain, not a variable-length one, and below the faithful depth it manufactures
transitions the model never makes.

This module builds the real thing. A *context* is a trailing run of blocks; the state of a
history is the SHORTEST context that is **safe**, where safe means every history ending in it
has the same future (its whole continuation law, stop included). Safety is decided by exact
backward signatures, so the merge is lossless: the chain started at START and absorbed at END
regenerates the model's block-word distribution exactly.

Three things that are easy to get wrong, and are handled here:

* **The safety test is over the whole future, not the next block.** Two contexts whose next-block
  distributions agree can still have different continuations. Comparing one step ahead prunes
  a;b and x;b in the model {a;b;c;d, x;b;c;e} (both go to c with probability 1) and leaks mass
  onto a;b;c;e — words the model gives probability zero.
* **Stop is a symbol.** The signature carries the END mass, so "this is where words end" is part
  of the future. Rows here are compared unnormalised in that sense: the stop mass is a component
  of the signature, never divided out.
* **Anchored contexts.** "The word so far is exactly u" and "a long history ending in u" can need
  different treatment: in {b;c, a;b;d} the history (b) goes to c and (a,b) goes to d, so a
  floating context b merges two different futures. A history with no safe floating context falls
  back to its anchored whole prefix, whose fibre is a singleton and therefore always safe.

Weights are converted to ``Fraction`` once and all comparisons are exact. A tolerance would make
the tree — hence k*, hence |X|, hence every normalised distance — a function of an epsilon and of
summation order, and a false merge is silent information loss. Exact arithmetic fails safe: two
weights *intended* equal but computed differently compare unequal, giving a tree deeper than
necessary but still lossless.

``ceiling`` caps how far back a context may reach. It is a CEILING, not an order: contexts stay
mixed-length below it, and merges below the ceiling are only ever the safe ones. The single unsafe
merge is the fallback when no safe context fits within the ceiling — which is exactly the crude
summary the classical footprint charts commit. ``ceiling=None`` (the default) is the lossless
chart and takes no parameter at all.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

from .matrix import END, START, block_sequence
from .moddecomp import decompose
from .poset import Model

SEP = "|"        # joins blocks within a context key
ANCHOR = "^"     # marks an anchored (whole-prefix) context key


def _check_labels(blocks) -> None:
    """Block labels come straight from activity labels and are not escaped, so a label
    containing the separator, or one shadowing a sentinel, would corrupt the state keys."""
    for b in blocks:
        if SEP in b or b.startswith(ANCHOR) or b in (START, END):
            raise ValueError(
                f"block label {b!r} collides with the state-key encoding "
                f"(separator {SEP!r}, anchor {ANCHOR!r}, sentinels {START!r}/{END!r})"
            )


def _words(model: Model) -> dict[tuple[str, ...], Fraction]:
    """Block words of a model with exact weights, duplicates accumulated."""
    out: dict[tuple[str, ...], Fraction] = defaultdict(Fraction)
    for P, w in model:
        blocks = tuple(block_sequence(decompose(P)))
        _check_labels(blocks)
        out[blocks] += Fraction(w)
    return dict(out)


def _flows(words):
    """``flow[h]`` = mass of words having h as a prefix; ``stop[h]`` = mass of words equal to h."""
    flow: dict[tuple, Fraction] = defaultdict(Fraction)
    stop: dict[tuple, Fraction] = defaultdict(Fraction)
    for w, m in words.items():
        for i in range(len(w) + 1):
            flow[w[:i]] += m
        stop[w] += m
    return dict(flow), dict(stop)


def _signatures(flow, stop):
    """``sig[h]`` encodes the ENTIRE continuation law after history h, exactly.

    Built longest-prefix-first so each child's signature is available. Two histories share a
    signature iff no sequence of future blocks (nor stopping) ever distinguishes them — the
    equivalence a safe merge must respect."""
    children = defaultdict(list)
    for h in flow:
        if h:
            children[h[:-1]].append(h)
    sig: dict[tuple, tuple] = {}
    for h in sorted(flow, key=len, reverse=True):
        f = flow[h]
        row = tuple(
            sorted(((c[-1], flow[c] / f, sig[c]) for c in children.get(h, ())), key=lambda t: t[0])
        )
        sig[h] = (row, stop.get(h, Fraction(0)) / f)
    return sig


def _safe_contexts(flow, sig, ceiling):
    """The floating contexts every one of whose histories shares a signature."""
    seen = defaultdict(set)
    for h in flow:
        if not h:
            continue
        longest = len(h) if ceiling is None else min(ceiling, len(h))
        for length in range(1, longest + 1):
            seen[h[-length:]].add(sig[h])
    return {ctx for ctx, sigs in seen.items() if len(sigs) == 1}


class ContextTree:
    """A frozen context assignment, shared by every model in a comparison.

    ``safe`` is the set of floating contexts; ``ceiling`` caps their length (None = unbounded).
    ``anchored`` enables the whole-prefix fallback; with it disabled a history that has no safe
    context is truncated instead, which is lossy but reproduces the classical fixed-order chart."""

    __slots__ = ("safe", "ceiling", "anchored")

    def __init__(self, safe, ceiling=None, anchored=True):
        self.safe = frozenset(safe)
        self.ceiling = ceiling
        self.anchored = anchored

    def context(self, history: tuple[str, ...]):
        """The state of ``history``: the shortest safe trailing run, else the anchored whole
        prefix, else (ceiling reached, anchoring off) the truncation — the one unsafe merge."""
        if not history:
            return None                                     # the empty history is START
        longest = len(history) if self.ceiling is None else min(self.ceiling, len(history))
        for length in range(1, longest + 1):
            ctx = history[-length:]
            if ctx in self.safe:
                return (ctx, False)
        if self.anchored and (self.ceiling is None or len(history) <= self.ceiling):
            return (history, True)                          # singleton fibre, always safe
        return (history[-longest:], False)

    def key(self, context) -> str:
        """State key in the string form ``matrix.build`` and ``distance`` expect."""
        if context is None:
            return START
        ctx, anchored = context
        return (ANCHOR + SEP if anchored else "") + SEP.join(ctx)

    def state_of(self, history) -> str:
        return self.key(self.context(history))

    def step(self, context, block: str):
        """The successor state of ``context`` on emitting ``block``.

        Well defined — this is what makes the chain Markov. A safe context stays safe when a
        block is appended, so the successor depends only on (context, block) and never on the
        history behind it."""
        if context is None:
            return self.context((block,))
        ctx, anchored = context
        return self.context(ctx + (block,))

    @property
    def depth(self) -> int:
        """Longest context in use — k* when the tree is the lossless one."""
        return max((len(c) for c in self.safe), default=0)

    def __repr__(self) -> str:
        return (f"ContextTree(|safe|={len(self.safe)}, depth={self.depth}, "
                f"ceiling={self.ceiling}, anchored={self.anchored})")


def context_tree(models, *, ceiling: int | None = None, anchored: bool = True) -> ContextTree:
    """ONE tree for the whole comparison: a context is merged only when it is safe in EVERY model.

    Pass the entire fleet. A per-pair tree would give each pair its own state space, and with the
    1/sqrt(|X|) normalisation a pair-dependent |X| breaks the triangle inequality. Models that
    never reach a context impose no constraint on it."""
    if models and isinstance(models[0], tuple):
        models = [models]                                   # a bare model, not a list of them
    per_model = []
    for m in models:
        flow, stop = _flows(_words(m))
        per_model.append(_safe_contexts(flow, _signatures(flow, stop), ceiling))
    if not per_model:
        return ContextTree(set(), ceiling, anchored)
    safe = set.intersection(*(set(s) for s in per_model))
    return ContextTree(safe, ceiling, anchored)


def build(model: Model, tree: ContextTree):
    """Return ``(matrix, states)`` on ``tree`` — the same contract as ``matrix.build``.

    Histories mapping to one state have their flows added, so a row is a visit-weighted average.
    That is exactly right when the merge is safe (the averaged rows are equal) and is where a
    ceiling-forced merge loses information."""
    flow, stop = _flows(_words(model))
    children = defaultdict(list)
    for h in flow:
        if h:
            children[h[:-1]].append(h)
    raw: dict[str, dict[str, Fraction]] = defaultdict(lambda: defaultdict(Fraction))
    states: set[str] = {START, END}
    for h in flow:
        src = tree.state_of(h)
        states.add(src)
        for child in children.get(h, ()):
            raw[src][tree.state_of(child)] += flow[child]
        if stop.get(h):
            raw[src][END] += stop[h]
    matrix: dict[str, dict[str, float]] = {}
    for s in states:
        row = raw.get(s, {})
        total = sum(row.values())
        matrix[s] = {d: float(v / total) for d, v in row.items()} if total else {}
    return matrix, states


def _as_fleet(models):
    """Accept either one model or a list of them."""
    if models and isinstance(models[0], tuple):
        return [models]
    return list(models)


def _longest_word(fleet) -> int:
    """L, the longest block word in the comparison — the ceiling above which no context can grow."""
    return max((len(w) for m in fleet for w in _words(m)), default=0)


def faithful_depth(models, *, anchored: bool = True) -> int:
    """k*: the smallest ceiling at which the chart is lossless for every model.

    Computed, not bounded. Sweeps upward to L (no context can exceed the longest block word) and
    stops at the first ceiling whose tree regenerates every model exactly."""
    fleet = _as_fleet(models)
    for k in range(1, max(_longest_word(fleet), 1) + 1):
        capped = context_tree(fleet, ceiling=k, anchored=anchored)
        if all(is_lossless(m, capped) for m in fleet):
            return k
    return max(_longest_word(fleet), 1)


def stable_depth(models, *, anchored: bool = True) -> int:
    """The ceiling above which the chart stops changing — NOT the same number as k*.

    Losslessness and stability are two different thresholds, and k* <= stable_depth. Raising the
    ceiling past k* can still move the chart: a history that fell back to its anchored prefix may
    find a safe floating context higher up, giving a different (equally lossless) chart. So a
    plateau on a depth axis begins here, not at k*."""
    fleet = _as_fleet(models)
    tree = context_tree(fleet, ceiling=None, anchored=anchored)
    longest = 0
    for model in fleet:
        flow, _ = _flows(_words(model))
        for h in flow:
            ctx = tree.context(h)
            if ctx is not None:
                longest = max(longest, len(ctx[0]))
    return max(longest, 1)


def generated_words(matrix, states=None, *, max_len: int | None = None, limit: int = 100000):
    """The block-word distribution the chain actually emits, walking START to END.

    The check that a representation is lossless rather than merely plausible: read off the matrix
    alone, with no reference to the model it came from.

    Returns ``(words, overflow)``. A lossy merge can put the chain in a cycle — the order model
    {a;b;c, a;c;b} at ceiling one lets b follow c and c follow b forever — so the walk is bounded
    at ``max_len`` blocks and whatever mass is still running is returned as ``overflow`` rather
    than enumerated. Overflow above zero is itself proof the chart is lossy."""
    out: dict[tuple[str, ...], float] = defaultdict(float)
    overflow = 0.0
    frontier = [((), START, 1.0)]
    seen = 0
    while frontier:
        word, state, mass = frontier.pop()
        seen += 1
        if seen > limit:
            raise RuntimeError("word enumeration exceeded its budget")
        if max_len is not None and len(word) >= max_len:
            overflow += mass * sum(p for d, p in matrix.get(state, {}).items() if d != END)
            out[word] += mass * matrix.get(state, {}).get(END, 0.0)
            continue
        for dest, p in matrix.get(state, {}).items():
            if dest == END:
                out[word] += mass * p
            elif dest == START:
                continue                                    # the END -> START reset, not a word
            else:
                frontier.append((word + (dest.split(SEP)[-1],), dest, mass * p))
    return dict(out), overflow


def is_lossless(model: Model, tree: ContextTree, *, tol: float = 1e-12) -> bool:
    """Does the chain on ``tree`` regenerate ``model``'s block-word distribution exactly?"""
    words = _words(model)
    total = sum(words.values())
    want = {w: float(m / total) for w, m in words.items()}
    longest = max((len(w) for w in words), default=0)
    got, overflow = generated_words(*build(model, tree), max_len=longest + 1)
    if overflow > tol:
        return False                                        # mass escaping past any model word
    if set(want) != {w for w, p in got.items() if p > tol}:
        return False
    return all(abs(want[w] - got.get(w, 0.0)) <= tol for w in want)
