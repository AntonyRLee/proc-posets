# Plan: output-sensitive `extract_signature` (kill the |B|×|F| product blow-up)

**Status:** ✅ IMPLEMENTED 2026-07-20 as `procposets/cospan/engine_fast.py`
(`extract_signature_fast`), exported from `cospan/__init__.py`, with the opt-in
`from_ocpn.signature_from_ocpn(ocpn, *, canonical=…)`. Cross-check
`tests/regression/test_cpm_extract_fast.py` is green (8 tests; full suite 335
passed, 20 skipped). **The "ready-to-drop-in" code near the bottom of this doc was
NOT byte-exact and is SUPERSEDED — see the correction note directly below.**
**Owner repo:** procposets (`procposets/cospan/engine.py`; the `-DIAGRAM-/sim/cpm`
copy is byte-identical and vendored).

## Correction (2026-07-20) — the verbatim drop-in was not exact

The mandated golden cross-check (`extract_signature_fast` CanonKeys == the slow
`extract_signature` CanonKeys) FAILED on the required running-example / mixed-graph
fixtures: slow=16 CanonKeys, fast=15, plus wrong terminus legs. Two independent
bugs, both invisible to the Bundestag-only prototype validation:

- **Bug A — no `surface_termini=False` strip.** The slow `forward_bundles` calls
  `_strip_termini` (drop every `gamma2` endpoint) when surfacing is off; the drop-in
  only handled the `surface=True` collapse. So every bare-sink activity grew a
  spurious terminus leg (`G2: … → None` vs `→ ·`; `X: · → box order` vs `→ order`).
  Also fires when an activity is literally labelled `"gamma2"` (== the `GAMMA2`
  sentinel).
- **Bug B — the "types are independent" premise is false under an *untyped* choice
  node.** `LMGraph.validate` only forces each mediator's *typed* edges to agree; a
  fully-untyped XOR (running-example `p2`) routes to differently-typed subtrees
  (`con` **xor** `box`). Grouping arcs by `e.typ` and collapsing by bare arity lost
  that split (`G1` got 1 CanonKey instead of 2).

Why the prototype looked exact: the Bundestag net is fully typed (each arc single-
typed → Bug B can't fire) and was validated with `surface_termini=True` (→ Bug A
can't fire). The regime it was validated in is exactly the regime where the drop-in
happens to be correct.

**The corrected algorithm (shipped in `engine_fast.py`):**
1. *Coupled-component decomposition* — union-find arcs by *shared reachable object
   type*. Two arcs reaching a common type are enumerated together (fixes Bug B:
   the untyped fan couples `con`/`box`); type-disjoint components factor. Over-
   coupling only costs enumeration, never correctness.
2. *Collapse to the CanonKey-relevant key* — each component keeps one representative
   bundle per distinct `(is_gamma2, type)`-multiset (a handful, not the `10^k` raw
   bundles). This is what kills the cross-type blow-up while staying exact.
3. *Reuse the slow engine's own* `_strip_termini` / `_collapse_pure_terminus` on the
   representatives — post-processing is identical by construction (fixes Bug A).

Below the per-component enum cap the fast side-bundle set is *equal set-for-set* to
the slow engine's `_and`. Above the cap (same-type high fan-out the slow engine also
can't finish): single-type → exact `[min-hitting, max-matching]` interval; multi-type
→ raises (only reachable from an untyped fan into a huge coupled type-web, never from
a discovered OCPN). The output-sensitivity (Bundestag 35,009 CanonKeys) is preserved
by moves 1–2; the `typed_hub` test pins the 16→1 collapse.
**Origin:** diagnosed from the `ocel-data-audit` project while running the SoSyM
Bundestag OCPN exhibit. Full evidence + figures:
`~/Research/ocel-data-audit/research/audits/2026-07-20-ocpn-extract-rootcause.md`
and `scripts/audits/{sosym_ocpn_extract_diagnose.py,fast_extract_prototype.py}`.

## Problem

`extract_signature` (engine.py:182) emits **one generator per firing-choice context
`(P, S)`** — `|B(a)| × |F(a)|` per activity (the `for P in B: for S in F:` loop,
engine.py:224–230). `forward_bundles`/`backward_bundles` compute `F`/`B` as
`engine._and(...)` = the **Cartesian product** over an activity's out-/in-arcs
(engine.py:171,177). After `without_silent`, the OCPN LM-graph has **only `XOR`
mediators** (from_petri: place→XOR, silent→SEQ-contracted), so `_and` is the *only*
product and it runs over an activity's arcs, each of which reaches transitively many
endpoints through the silent-contracted place→place closure.

Hence `|F(a)| = ∏_{p∈out-arcs} |reachable endpoints(p)|`, and likewise `|B|`. A hub
activity shared across `k` object types has ~`k` in-arcs and ~`k` out-arcs, so this
is **exponential in the object-type count**. On the Bundestag full log (44 types),
hub `Beratung` has in/out degree 26/26 → ≈**1.4×10³⁰ generators for that one
activity**. Stage B never returns (killed at timeout); pm4py's discovery is 6.2 s.
**It is not a graph scan — 1,228 arcs — it is a materialised DNF.**

The waste: `CanonKey = (label, per-type in-arity, per-type out-arity)`
(signature_compare.py:60–81) depends only on how many ports of each type a side has,
**not which specific predecessor/successor is chosen per arc**. So the ~10³⁰
generators collapse onto the true distinct-CanonKey count.

## The fix — compute the per-type arities directly (output-sensitive)

Object types are independent (mediators are type-preserving, `LMGraph.validate`), so
endpoints never mix across types. Therefore a CanonKey's per-type arity depends only
on **that type's** arcs. Algorithm, per activity, per side:

1. For each arc, get its reachable endpoint bundles via the existing `engine._traverse`
   (identical semantics: gamma2 sink termini forward, empty origins backward, cycle
   cuts) — cheap, no product.
2. Group arcs by object type. For each type, enumerate the (tiny) product **over that
   type's arcs only** to get the achievable set of arities (and whether an arity is
   realisable purely by gamma2 endpoints, for the pure-terminus collapse). Fall back
   to the exact `[min-hitting, max-matching]` interval if a single type's arc product
   exceeds a cap (rare high-fan-out same-type case).
3. The side's achievable type-multisets = product **over types** of the per-type
   arity choices (small — this is the genuinely distinct result). A forward bundle
   that is pure gamma2 across every type collapses to empty (mirror
   `_collapse_pure_terminus`).
4. CanonKeys of the activity = in-multisets × out-multisets; emit one representative
   `Generator` per CanonKey.

Cost ≈ `O(activities × arcs + |Σ_canonical|)` — **output-sensitive**, no cross-type
or cross-side product.

### Validation (already done, prototype)

`ocel-data-audit/scripts/audits/fast_extract_prototype.py` on the Bundestag OCPN,
restricted to the `k` largest object types, comparing the **CanonKey set** to
`canonical_generators(extract_signature(g, surface_termini=True))`:

| k | fast CanonKeys | fast time | slow engine |
|---:|---:|---:|---|
| 1 | 18 | 0.000 s | 18 (86 gens) ✓ EXACT |
| 2 | 28 | 0.000 s | 28 (102 gens) ✓ EXACT |
| 4 | 93 | 0.002 s | 93 (2,693 gens) ✓ EXACT |
| 6 | 193 | 0.006 s | 193 (55,975 gens, 1.6 s) ✓ EXACT |
| 8 | 623 | 0.038 s | never finishes |
| 12 | 1,179 | 0.053 s | never finishes |
| **44** | **35,009** | **4.25 s** | never finishes |

Exact on every case the slow engine can reach. (Note: the capped diagnostic earlier
reported "357" CanonKeys at k=44 — that was an undercount because 42 hub activities
were skipped at the product cap; the true count is **35,009**, and the fast extractor
gets it in 4.25 s.)

## Implementation steps (for the jump-over session)

1. **Add `procposets/cospan/engine_fast.py`** — the drop-in below (ready; mirrors the
   validated prototype, adds representative generators + the matching/hitting fallback).
   Export `extract_signature_fast` from `cospan/__init__.py` alongside `extract_signature`.
2. **`tests/test_extract_fast.py`** — assert `canonical_generators(extract_signature(g,
   **kw)).keys() == canonical_generators(extract_signature_fast(g, **kw)).keys()` on:
   the running-example fixtures (reuse `test_engine_running_example` / `test_surface_termini`
   builders), the object-centric fixtures (`test_object_centric_signatures`), and a
   gamma-boundary case (`test_gamma_boundary`). Cover both `surface_termini` values.
3. **Run `uv run pytest`** under the resource wrapper — expect the prior `334 passed,
   13 skipped` plus the new tests green. `extract_signature` is untouched, so nothing
   should regress.
4. **Wire an opt-in** in `discover.py`: `signature_from_ocpn(ocel, *, canonical=False)`
   → use `extract_signature_fast` when `True` (keep `extract_signature` the default so
   splice/behavioural users are unaffected). Decide the default in-session.
5. **Optional**: a guarded Bundestag-scale smoke (needs `.[pm4py]` + the OCEL; skip if
   absent) asserting the full 44-type net extracts (35,009 CanonKeys) in seconds.

### Caveats / decisions
- Returns **one representative generator per CanonKey** (OCPN generators carry no
  bindings). Correct for `compare` / `canonical_generators` and the type-level views.
  For the full per-context generator set (splice / behavioural semantics via
  `extract_dp`/`occurrence`) keep the original `extract_signature`.
- The 35,009 figure is inflated by activities whose *same-type* multi-arcs yield an
  arity **range** (variable/double arcs from the inductive miner). Whether that
  ambiguity is meaningful or discovery noise is a modelling question — see optimisation
  #4 (arity-range collapse) below.

## ⚠️ SUPERSEDED draft code (kept for provenance — do NOT copy)

The block below is the *original* prototype-mirroring draft. It is **not byte-exact**
(Bugs A & B above) and was replaced by the corrected `engine_fast.py` in the repo.
Read the shipped file, not this. Kept only to document what the cross-check caught.

```python
"""Output-sensitive signature extraction: same Sigma (one representative generator
per distinct CanonKey) without the |B|x|F| product blow-up of extract_signature.
See docs/2026-07-20-fast-signature-extraction.md for the derivation + validation."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product

from .engine import GAMMA2, _traverse
from .lmgraph import LMGraph
from .signature import Generator, Port, Signature
from .signature_compare import canon_key

_ENUM_CAP = 200_000  # per-TYPE arc product budget (tiny unless same-type high fan-out)


def _arc_options(g: LMGraph, a: str, forward: bool) -> dict:
    edges = g.out_edges(a) if forward else g.in_edges(a)
    by_type: dict = defaultdict(list)
    for e in edges:
        node = e.tgt if forward else e.src
        Se = _traverse(g, node, (e.typ,), frozenset(), forward=forward)
        by_type[e.typ].append([frozenset(b) for b in Se])
    return by_type


def _max_matching(reach: list) -> int:
    match: dict = {}

    def aug(i, seen):
        for ep in reach[i]:
            if ep in seen:
                continue
            seen.add(ep)
            if ep not in match or aug(match[ep], seen):
                match[ep] = i
                return True
        return False

    size = 0
    for i in range(len(reach)):
        if reach[i] and aug(i, set()):
            size += 1
    return size


def _min_hitting(must: list) -> int:
    if not must:
        return 0
    universe = sorted(set().union(*must), key=str)
    for s in range(1, len(must) + 1):
        for cand in combinations(universe, s):
            cs = set(cand)
            if all(cs & r for r in must):
                return s
    return len(must)


def _achievable(arc_opts: list) -> list:
    total = 1
    for o in arc_opts:
        total *= max(1, len(o))
    seen: dict = {}
    if total <= _ENUM_CAP:
        for combo in product(*arc_opts):
            eps = frozenset().union(*combo) if combo else frozenset()
            key = (len(eps), bool(eps) and all(e[0] == GAMMA2 for e in eps))
            seen.setdefault(key, eps)
        return [(ar, g2, wit) for (ar, g2), wit in seen.items()]
    reach = [{ep for b in opts for ep in b} for opts in arc_opts]
    must = [r for opts, r in zip(arc_opts, reach) if frozenset() not in opts and r]
    lo, hi = _min_hitting(must), _max_matching(reach)
    universe = sorted(set().union(*reach), key=str) if reach else []
    g2eps = [e for e in universe if e[0] == GAMMA2]
    out = []
    for ar in range(max(lo, 0), hi + 1):
        wit = frozenset(universe[:ar])
        out.append((ar, bool(wit) and all(e[0] == GAMMA2 for e in wit),
                    frozenset(g2eps[:ar]) if len(g2eps) >= ar else wit))
    return out or [(0, False, frozenset())]


def _side_representatives(g: LMGraph, a: str, forward: bool, surface: bool) -> set:
    by_type = _arc_options(g, a, forward)
    types = list(by_type)
    per_type = [_achievable(by_type[t]) for t in types]
    reps: set = set()
    for combo in (product(*per_type) if per_type else [()]):
        eps = frozenset()
        all_g2, any_port = True, False
        for ar, g2, wit in combo:
            if ar > 0:
                eps |= wit
                any_port = True
                if not g2:
                    all_g2 = False
        reps.add(frozenset() if (surface and any_port and all_g2) else eps)
    return reps


def extract_signature_fast(g: LMGraph, *, surface_termini: bool = False,
                           remove_silent: bool = True) -> Signature:
    """Output-sensitive twin of engine.extract_signature: one representative Generator
    per distinct CanonKey, without materialising |B|x|F|. For compare / type-level
    views; for the full per-context set (splice) use extract_signature."""
    if remove_silent and g.silent:
        g = g.without_silent()
    if surface_termini and any(
        g.lab(a) == GAMMA2 or g.lab(a).startswith("END_") for a in g.activities
    ):
        surface_termini = False
    best: dict = {}
    for a in g.activities:
        lab = g.lab(a)
        B = _side_representatives(g, a, forward=False, surface=False)
        F = _side_representatives(g, a, forward=True, surface=surface_termini)
        for P in B:
            for S in F:
                left = frozenset(Port(p, t, lab) for (p, t) in P)
                right = frozenset(Port(lab, t, s) for (s, t) in S)
                gen = Generator(lab, left, right)
                best.setdefault(canon_key(gen), gen)
    return Signature(frozenset(best.values()))
```

## Other optimisations to investigate (ranked)

1. **This: output-sensitive extract** — removes the exponential. Primary. Above.
2. **Adjacency index in `LMGraph`** — `out_edges`/`in_edges` (lmgraph.py:77–81) are
   O(|E|) linear scans, called throughout `_traverse`; build a `{node: [edges]}` index
   once at construction. Broad constant-to-linear win for *every* consumer (engine,
   occurrence, without_silent), independent of the product fix.
3. **Memoise `_traverse`** on `(node, forward)` (the reachable bundle-set from a node
   is direction-pure and reused across every arc/activity that hits it). Bounded by
   reachable endpoints; large constant-factor win on dense merged nets. Compose with #2.
4. **Arity-range collapse (semantic)** — most of the 35,009 come from activities whose
   same-type multi-arcs (inductive-miner variable/double arcs) yield an arity *range*.
   Optionally canonicalise to a single arity (max, or the modelled multiplicity),
   shrinking the signature by ~1–2 orders. Changes semantics → decide with the paper.
5. **Faster `without_silent`** — currently O(silent · in · out) splices with `spliced
   in g.edges` membership on a *list* (lmgraph.py:118) → up to O(silent·deg²·|E|). Use
   an edge `set`/index. Matters on inductive nets (silent-heavy).
6. **OCPN structural reduction before extraction** — free-choice / place-fusion
   reduction on the discovered net shrinks the place→place closure (hence per-arc
   reachable sets) before extraction; orthogonal to #1.
7. **Per-type-then-merge extraction** — build per-object-type signatures and merge only
   at shared activities, skipping cross-type products where types don't actually couple.
8. **CanonKey-native compare** — when only `compare` is needed, skip building the
   representative `Signature` and emit `canonical_generators` straight from the graph.
9. **Occurrence/unroll audit (`extract_dp`, `occurrence.py`)** — the splice/behavioural
   pipeline (`extract_classes`, `MAX_IDEAL_STATES` guard) has its own scaling wall;
   worth the same treatment if behavioural semantics are needed at full scale.
