"""Regression (compose LoopBox dedup): the loop-truncation branch of
``compose_signature`` must honour the SAME interleaving-dedup as completed
composites.

A signature pairing a self-looping generator (``L``: x->x) with an independent
concurrent generator (``sq``->``N``) yields several firing interleavings that
truncate to the same loop composite -- identical pre-loop generator multiset and
identical loop body, differing only in the order the concurrent ``N`` was
interleaved against the two ``L`` firings. Those are the same diagram and must
collapse to one, exactly as AND-concurrent interleavings of a *completed*
composite already do (the ``seen`` set).

Before the fix the LoopBox branch appended straight to ``results`` without ever
consulting/updating ``seen``, so it returned the duplicates (one-way
over-production). Loop-free enumeration -- e.g. the running-example golden -- is
unaffected (it never enters the LoopBox branch).

The key below reconstructs the exact key the enumerator dedupes on
(``placements == placed + (box,)``, so the non-box placements ARE ``placed``);
asserting every returned composite has a distinct such key is precisely the
``seen`` invariant the fix restores.
"""
from procposets.cospan.compose import CompositeDiagram, LoopBox, compose_signature
from procposets.cospan.signature import Generator, Port, Signature

_X = Port("L", "t", "L")    # the wire L self-loops on
_Q = Port("sq", "u", "N")   # the independent sq -> N channel

_SP = Generator("sp", frozenset(), frozenset({_X}))       # source: emits x
_L = Generator("L", frozenset({_X}), frozenset({_X}))     # self-loop on x
_SQ = Generator("sq", frozenset(), frozenset({_Q}))       # source: emits q
_N = Generator("N", frozenset({_Q}), frozenset())         # drains q

_SIG = Signature(frozenset({_SP, _L, _SQ, _N}))


def _seen_key(c: CompositeDiagram):
    placed = [p for p in c.placements if not isinstance(p, LoopBox)]
    box = next((p for p in c.placements if isinstance(p, LoopBox)), None)
    base = tuple(sorted(str(g) for g in placed))
    if box is None:
        return base
    return base + ("<loop>",) + tuple(str(b) for b in box.body)


def test_loopbox_composites_are_interleaving_deduped():
    results = compose_signature(_SIG, unroll=2)
    loop = [c for c in results if any(isinstance(p, LoopBox) for p in c.placements)]
    assert loop, "expected at least one LoopBox composite (the truncation branch must be exercised)"
    keys = [_seen_key(c) for c in results]
    assert len(set(keys)) == len(results), (
        "compose_signature returned duplicate composites under the placed-multiset + "
        "loop-body key: the LoopBox branch is bypassing `seen` (interleaving-dedup)."
    )
