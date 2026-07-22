"""General morphism-class extraction: given any generator-cospan ``Signature``,
discover the named composite morphisms -- atomic generators plus
hierarchically-discovered loop structures -- that participate in closing a
boundary (e.g. gamma1 -> gamma2).

This module implements the S7 algorithm, plus one design finding made
during implementation -- see "zero-left generators" below.

Frontiers are multisets of ``Port`` (``collections.Counter``), not sets --
future signatures may have multiplicity >1 boundaries even though today's
don't. The live search only ever fires one atomic generator at a time, in a
fixed canonical order; any apparent parallelism (``@``) and any loop
hierarchy is recovered purely by compressing *completed* cycle-paths against
a growing fragment dictionary -- the live transition rule never gains new
moves, which preserves frontier-only memoization (a Markov property: what
happens next depends only on the current frontier, never on history).

Zero-left generators (``gamma1``-shaped: empty left boundary) are trivially
"enabled" at *every* frontier, since ``frozenset() <= anything``. Left
unrestricted, this lets a start generator refire at every visited frontier,
producing an unboundedly-growing, never-cycling family of frontiers (a second,
concurrent process instance stacked on top of the first, rather than a cycle)
-- found empirically while tracing the algorithm by hand before writing tests,
not anticipated in the design doc. Restricting each zero-left generator to
fire at most once per DFS path (tracked alongside the stack) is the fix: it
is never the closing edge of a genuine cycle anyway (firing it only ever
grows the frontier, so it can never land back on a smaller ancestor unless
its right boundary is *also* empty, a degenerate no-op case not handled
specially here), so restricting it only prunes wasteful exploration, never
hides a real loop.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .signature import Generator, Port, Signature

# A frontier's canonical, hashable form: a frozenset of (port, count) pairs
# with all-positive counts (zero counts dropped). Using a multiset rather
# than a plain set is what makes two interleavings of independent generators
# land on the literal same key with no extra canonicalisation step.
FrontierKey = frozenset


def _to_key(counts: Counter[Port]) -> FrontierKey:
    return frozenset((p, n) for p, n in counts.items() if n > 0)


def _to_counter(key: FrontierKey) -> Counter[Port]:
    return Counter(dict(key))


def _left_weighted(g: Generator) -> Counter[Port]:
    """Left boundary as a weighted multiset (§38): leg ``p`` carries ``g.weight(p)``
    objects, not the unit count a bare frozenset would give. Ungrounded generators
    have empty ``weights`` so every leg is weight 1 -- identical to ``Counter(g.left)``."""
    return Counter({p: g.weight(p) for p in g.left})


def _right_weighted(g: Generator) -> Counter[Port]:
    return Counter({p: g.weight(p) for p in g.right})


def _enabled(frontier: Counter[Port], g: Generator) -> bool:
    return all(frontier[p] >= w for p, w in _left_weighted(g).items())


def _fire(frontier: Counter[Port], g: Generator) -> Counter[Port]:
    out = Counter(frontier)
    for p, w in _left_weighted(g).items():
        out[p] -= w
    for p, w in _right_weighted(g).items():
        out[p] += w
    return Counter({p: n for p, n in out.items() if n > 0})


def _sort_key(g: Generator):
    return (g.label, tuple(sorted(g.left)), tuple(sorted(g.right)))


@dataclass(frozen=True)
class Ref:
    """A reference to a previously-named fragment, used as one recorded step."""

    name: str

    def __str__(self) -> str:
        return self.name


# A recorded step: a single generator firing, an `@`-merged simultaneous
# group, or a reference to a previously-named fragment.
Step = Generator | frozenset[Generator] | Ref


def _step_str(step) -> str:
    if isinstance(step, Ref):
        return step.name
    if isinstance(step, frozenset):
        return "(" + " @ ".join(sorted(g.label for g in step)) + ")"
    return step.label


@dataclass(frozen=True)
class NamedMorphism:
    """A named composite morphism: an atomic-generator chain (with `@`-merged
    parallel groups and references to smaller named fragments folded in) that
    closes a boundary back to where it started. ``boundary == frozenset()``
    (the empty frontier) marks a gamma1->gamma2-closing instance; any other
    boundary marks an interior loop discovered at that frontier."""

    name: str
    boundary: FrontierKey
    body: tuple  # tuple[Generator | frozenset[Generator] | Ref, ...]

    def is_closing(self) -> bool:
        return len(self.boundary) == 0

    def __str__(self) -> str:
        return f"{self.name} = " + " >> ".join(_step_str(s) for s in self.body)


def _merge_parallel(frontiers_before: list[Counter[Port]], moves: list[Generator]) -> list:
    """Fold maximal runs of consecutive, jointly-independent firings into one
    `@`-group. ``moves[k]`` fires from ``frontiers_before[k]``. A run
    extends from position ``i`` to ``j`` iff the *combined* left boundary of
    moves[i..j] fits inside ``frontiers_before[i]`` -- i.e. they could all have
    fired simultaneously from that one frontier, in any order, with the same
    net result (sound because multiset add/subtract commute).

    A group is a ``frozenset`` of *distinct* generators, so two **identical**
    parallel firings (a tensor power ``a^{⊗k}`` -- e.g. two orders each launched by
    their own copy of activity ``a``, §38/§39) must NOT be merged into one group, or
    the set would silently drop the duplicate and lose its tokens. They stay as
    separate steps (cosmetically ``a >> a`` rather than ``a @ a``); the reconstructed
    occurrence DAG is identical either way -- independent firings carry no edge
    between them regardless of framing."""
    merged = []
    i, n = 0, len(moves)
    while i < n:
        group = [moves[i]]
        group_left = _left_weighted(moves[i])
        start = frontiers_before[i]
        j = i + 1
        while j < n:
            if moves[j] in group:
                break  # identical parallel firing: keep separate (set would drop it)
            candidate = group_left + _left_weighted(moves[j])
            if all(start[p] >= c for p, c in candidate.items()):
                group.append(moves[j])
                group_left = candidate
                j += 1
            else:
                break
        merged.append(group[0] if len(group) == 1 else frozenset(group))
        i = j
    return merged


def _compress(body: tuple, fragments_by_body: dict) -> tuple:
    """Greedy left-to-right longest-match substitution of ``body``'s steps
    against previously-named fragments, longest fragment first at each
    position."""
    if not fragments_by_body:
        return body
    known = sorted(fragments_by_body.items(), key=lambda kv: -len(kv[0]))
    out = []
    i, n = 0, len(body)
    while i < n:
        for frag_body, nm in known:
            flen = len(frag_body)
            if flen and flen <= n - i and tuple(body[i : i + flen]) == frag_body:
                out.append(Ref(nm.name))
                i += flen
                break
        else:
            out.append(body[i])
            i += 1
    return tuple(out)


def _flatten_generators(body: tuple, by_name: dict[str, NamedMorphism]) -> set[Generator]:
    out: set[Generator] = set()
    for step in body:
        if isinstance(step, Ref):
            out |= _flatten_generators(by_name[step.name].body, by_name)
        elif isinstance(step, frozenset):
            out |= step
        else:
            out.add(step)
    return out


class TooManyFrontiers(RuntimeError):
    """Raised when the search visits more distinct frontiers than the
    configured safety cap -- a deliberate, controlled failure rather than
    an open-ended hang."""


@dataclass
class ExtractionResult:
    fragments: dict[str, NamedMorphism]
    valid_generators: set[Generator]
    frontiers_visited: int
    truncated: bool = False  # True iff a per-state iso-class cap was hit (the
    # signature's loop-free closing catalogue is larger than enumerated -- the
    # "this net over-generates" signal; e.g. an OCPN
    # with independent per-type subnets whose execution space is a free product)

    def closing(self) -> list[NamedMorphism]:
        """The gamma1->gamma2-closing instances (boundary == empty)."""
        return [nm for nm in self.fragments.values() if nm.is_closing()]

    def loops(self) -> list[NamedMorphism]:
        """The interior loop structures (boundary != empty)."""
        return [nm for nm in self.fragments.values() if not nm.is_closing()]

    def pretty(self) -> str:
        lines = [str(nm) for nm in sorted(self.fragments.values(), key=lambda nm: nm.name)]
        return "\n".join(lines)


# ``extract_classes`` is the pomset-DP, iso-class-quotiented extractor
# defined in ``extract_dp.py``. The legacy
# path-enumerating DFS that lived here -- whose ``exits_to_root`` replay both
# dropped ``M(1,trop)`` and, once corrected, blew up combinatorially on
# concurrent generators (§21) -- has been removed in favour of the categorical
# iso-class DP. The helper functions above (``_enabled``/``_fire``/``_to_key``/
# ``_merge_parallel``/``_compress``/``NamedMorphism``/...) remain the shared
# substrate used by ``extract_dp``, ``occurrence``, ``morphism_schema`` and
# ``signature_diff``.
def __getattr__(name):  # PEP 562: lazy re-export, breaks the import cycle
    # (extract_dp imports this module's helpers at its top; resolving the
    # re-export eagerly would need extract_dp fully initialised before this
    # module is, which fails when extract_dp is imported first).
    if name == "extract_classes":
        from .extract_dp import extract_classes

        return extract_classes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
