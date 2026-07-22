"""Compose a generator signature ``Sigma`` back into connected composite
string diagrams -- the finite set of "ground truth" diagrams the paper's
``F(Sigma)`` lets us reconstruct (each one a distinct XOR-resolved branch;
no gateway nodes are reintroduced -- AND legs are just a generator's multiple
boundary ports).

Composition is pushout-by-port-identity: a generator's ``right`` ports are
typed triples ``(this, type, succ)`` that get added to a pool of unmet
("available") ports, and a generator is *ready* exactly when its entire
``left`` is a subset of that pool. At each step every ready generator (across
every label, not just one chosen port) is a candidate -- restricting to a
single port's consumers first is unsound: an AND-join (e.g. two concurrent
branches both feeding one activity) can look "ready" from one branch's port
before its sibling branch has fired, and greedily taking it strands the
sibling. Zero ready generators is a dead end (a malformed signature, or a
branch that can never complete); exactly one is forced sequencing; more than
one is a genuine XOR fork (a `(P,S)` context choice) -- we branch the search.
Different firing orders of independent (AND-concurrent) generators reach the
same final composite by different paths; we dedupe on the resulting
multiset of generators used, keeping the first ordering found.

A composite is done when the pool empties immediately after firing a
generator labelled ``end_label``. Loops are detected by a generator
recurring along one path; we unroll a bounded number of repeats (so the
repeated body is visible) and then truncate that branch as an open
``LoopBox`` rather than unrolling forever -- ``f^(n)`` is a box-label
shorthand for "repeat sequentially n times", not a traced/feedback wire.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product

from .signature import Generator, Port, Signature


@dataclass(frozen=True)
class LoopBox:
    """A collapsed repeated unit: ``body`` fired more than ``unroll`` times
    in a row along one composite; the diagram shows ``body`` once, labelled
    ``f^(n)`` with ``n`` left symbolic."""

    body: tuple[Generator, ...]

    @property
    def label(self) -> str:
        names = "·".join(g.label for g in self.body)
        return f"({names})^(n)"


@dataclass(frozen=True)
class CompositeDiagram:
    """One connected, gateway-free composite string diagram: a sequence of
    generator placements (and any collapsed loop boxes) from the start
    boundary to the end boundary."""

    placements: tuple[Generator | LoopBox, ...]

    def labels(self) -> tuple[str, ...]:
        return tuple(p.label for p in self.placements)

    def label_multiset_key(self) -> tuple[str, ...]:
        """Order-independent identity for this composite: its sorted label multiset.

        Concurrent (AND-branch) generators have no fixed order in a string diagram,
        only sequential ones do -- ``compose_signature`` already dedupes interleavings
        of the same generator set down to one representative ordering, so two
        composites differing only in *which* interleaving survived that dedup are the
        same diagram and must compare equal. Renamed from ``canonical_key`` to
        disambiguate from the two unrelated key functions
        :func:`procposets.cospan.occurrence.canonical_key` (the EventDag WL hash) and
        :func:`procposets.cospan.signature_compare.canon_key` (the Generator key)."""
        return tuple(sorted(self.labels()))


def compose_signature(
    sigma: Signature,
    start_label: str | None = None,
    end_label: str | None = None,
    unroll: int = 2,
) -> list[CompositeDiagram]:
    """Enumerate every connected composite diagram of ``sigma``.

    *Sources* are generators with empty ``left`` (the per-type lifecycle starts
    ``▷_ot``, or a single process start such as the running example's ``G1``);
    *sinks* are those with empty ``right``.  Sources sharing a label are XOR
    alternatives (choose one); sources with *different* labels are independent
    concurrent channels (each chosen one is fired) -- so one composite is seeded
    per (non-empty subset of source labels) x (one generator per chosen label).
    A run is complete when the pool of unmet ports empties (every obligation
    discharged through a sink).  The pool is a *multiset* of ports, not a set:
    a port produced twice (e.g. by a generator looping before any of its
    consumers fire) banks two outstanding tokens, so a downstream consumer can
    fire that many times independently of the producer's own loop count --
    this is what makes ``v <= u`` style coupling between two loops
    representable (``rem:loop-adequacy``), not just same-generator recurrence.

    ``start_label`` / ``end_label`` are optional filters: ``start_label``
    restricts the seeded sources to that label; ``end_label``, if given, also
    requires the last placed generator to carry it (the single-source/sink
    convention the running-example golden test uses).  Both ``None`` (the
    boundary-explicit OCCN default) uses all sources/sinks.
    """
    # `Signature.generators` is a frozenset; its iteration order depends on
    # Python's per-process hash-seed randomization. Sort once, by the same
    # stable key as `Signature.pretty()`/`vis.pick()`, so which composite is
    # enumerated first -- and the placement order within one, for an
    # AND-concurrent pair -- is reproducible across runs.
    gens = sorted(sigma, key=str)
    by_lab: dict[str, list[Generator]] = defaultdict(list)
    for g in gens:
        if not g.left and (start_label is None or g.label == start_label):
            by_lab[g.label].append(g)
    labels = sorted(by_lab)
    results: list[CompositeDiagram] = []
    seen: set[tuple[str, ...]] = set()
    for r in range(1, len(labels) + 1):
        for chosen_labels in combinations(labels, r):
            for choice in product(*(by_lab[lab] for lab in chosen_labels)):
                pool: dict[Port, int] = {}
                for g in choice:
                    for p in g.right:
                        pool[p] = pool.get(p, 0) + g.weight(p)
                _extend(gens, pool, list(choice), end_label, unroll, results, seen)
    return results


def _extend(
    gens: list[Generator],
    available: dict[Port, int],
    placed: list[Generator],
    end_label: str | None,
    unroll: int,
    results: list[CompositeDiagram],
    seen: set[tuple[str, ...]],
) -> None:
    if not available:
        if placed and (end_label is None or placed[-1].label == end_label):
            key = tuple(sorted(str(g) for g in placed))
            if key not in seen:
                seen.add(key)
                results.append(CompositeDiagram(tuple(placed)))
        return

    # Readiness/consumption respect leg weights (§38 grounding): a grounded
    # generator consuming a bundle of w objects on a wire needs w tokens of that
    # port banked and removes all w in one firing. Ungrounded generators (no
    # weights) carry weight 1 everywhere, recovering the prior behaviour.
    candidates = [
        g for g in gens
        if g.left and all(available.get(p, 0) >= g.weight(p) for p in g.left)
    ]
    if not candidates:
        return  # dead end: no generator is ready on the current pool

    for g in candidates:
        if placed.count(g) >= unroll:
            box = LoopBox(_loop_body(placed, g))
            # LoopBox composites must honour the same interleaving-dedup as
            # completed ones (module docstring / label_multiset_key): AND-concurrent
            # interleavings of the same pre-loop generators that truncate on the same
            # loop body are the same diagram. Key on the placed-generator multiset
            # (str(g) basis, matching the completion path above) plus the *ordered*
            # loop body, namespaced with a sentinel so it can never collide with a
            # completion key. Direction was previously one-way over-production: this
            # branch never touched ``seen``.
            key = (
                tuple(sorted(str(x) for x in placed))
                + ("<loop>",)
                + tuple(str(b) for b in box.body)
            )
            if key not in seen:
                seen.add(key)
                results.append(CompositeDiagram(tuple(placed) + (box,)))
            continue
        new_available = dict(available)
        for p in g.left:
            new_available[p] -= g.weight(p)
            if new_available[p] == 0:
                del new_available[p]
        for p in g.right:
            new_available[p] = new_available.get(p, 0) + g.weight(p)
        _extend(gens, new_available, placed + [g], end_label, unroll, results, seen)


def _loop_body(placed: list[Generator], repeated: Generator) -> tuple[Generator, ...]:
    """The sequence of placements since ``repeated`` last occurred, i.e. the
    one repeating unit to show inside the ``f^(n)`` box."""
    last = len(placed) - 1 - placed[::-1].index(repeated)
    return tuple(placed[last:]) + (repeated,)
