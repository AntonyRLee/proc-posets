# Fast signature extraction — derivation and exactness contracts

How `procposets` extracts a generator-cospan signature from a wide
object-centric net without materialising the exponential context product, and
which committed test pins each exactness seam.  (The dated research trail —
measured Bundestag numbers, adjudication, migration plan — lives in the
repo-local `docs/2026-*.md` session notes, which are deliberately untracked;
this file is the stable, committed reference.)

## The three output layers

Extraction has three nested outputs; only the innermost is expensive:

- **L1 — skeleton**: per activity, the per-arc XOR-alternative bundle families
  (`skeleton.extract_skeleton`).  A *sum* over arcs: O(model + per-arc
  alternatives) — an arc's family is `_traverse`'s output, which AND/OR
  mediators can grow, so "O(model)" holds for discovered OCPNs (XOR places)
  but not unconditionally.
- **L2 — CanonKey set**: what `signature_compare.compare` consumes —
  `(label, per-type in-arity multiset, per-type out-arity multiset)`.
  Bounded by *achievable arity profiles*, not by the model: a hub with k
  independently-optional types has 2^k profiles, still enumerable.
- **L3 — full concrete contexts**: every `(P, S) ∈ B_a × F_a` as an atomic
  `Generator` (`engine.extract_signature`).  The product of the two marginals —
  on a wide hub, astronomically redundant (each side family is itself a product
  over the activity's arcs of their alternative counts).

The slow engine is kept as the **test oracle** and for the splice/behavioural
machinery (`extract_dp.extract_classes` needs concrete ports); everything else
consumes L1 or L2.

## L2: the output-sensitive CanonKey extractor (`engine_fast`)

A CanonKey reads only, per boundary side, the multiset of endpoint object
types (plus, under `surface_termini`, whether a leg is a `gamma2` terminus) —
never *which* concrete neighbour.  So `_side_profiles` combines an activity's
arcs one at a time carrying only the per-type `(real, gamma2)` leg counts,
deduping on that state after every fold:

- Two partial states with equal per-type counts have identical futures, so the
  state set is bounded by the number of distinct *profiles* — the output —
  never by the product-over-arcs of concrete endpoint choices.
- Arcs whose alternatives stay within one object type (the OCPN norm) factor
  into that type's independent contribution; only the rare cross-type-coupling
  arcs (untyped choice fan-ins) are folded jointly.
- Terminus handling mirrors the slow engine exactly: default strip absorbs
  `gamma2` legs; `surface_termini` keeps them and collapses an all-`gamma2`
  side to zero-right (`engine._collapse_pure_terminus`).
- The dedup space is the **output** space: per-type options are reduced to
  what the final mapping keeps — merged `real+gamma2` count plus one
  side-level saturating "has a real leg" bit under `surface_termini`, real
  count only under strip — *before* the cross-arc/cross-type folds.  The raw
  real-vs-gamma2 split space can be 16× larger than the profile space (2^19
  vs ~2^15 on the Bundestag hub side); folding reduced is exact because the
  reduction is additive per arc and OR-saturating on the bit.  Measured on the
  full 44-type Bundestag OCPN: 15.4 s → 5.5 s end-to-end (1.8 s profiles +
  1.1 s representative-generator build), i.e. parity with the CanonKey-only
  prototype once generator materialisation is subtracted.

**Contract:** byte-identical CanonKey set to `engine.extract_signature` on
every graph the slow engine can reach.  Pinned by
`tests/regression/test_cospan_extract_fast.py`, including the wide-optional-hub
scale guard (`(m+1)^k` concrete product vs `2^k` profiles).

## L1: the factored skeleton and lazy-⋈ composition

`skeleton.FactoredGenerator` stores, per in-/out-arc, that arc's alternative
bundle family — `engine._traverse`'s output stopped one step before
`engine._and`'s cross-arc product.  Two consumers:

- **`FactoredSignature.materialise()`** rebuilds the full L3 signature using
  the slow engine's own helpers, so it equals `extract_signature` *by
  construction* — for every `(kappa, remove_silent, surface_termini)`.  The
  default terminus strip commutes with the cross-arc union
  (`strip(b1 ∪ b2) = strip(b1) ∪ strip(b2)`), so it is baked into the stored
  families; the pure-terminus collapse reads the whole side, so it is deferred
  (`collapse_terminus`).  Pinned by `test_cospan_skeleton.py`.
- **`compose.compose_signature(FactoredSignature)`** joins contexts lazily at
  fire-time: `ready_lefts(pool)` restricts each arc to its pool-covered
  alternatives *before* the product (exact, because a union is covered iff
  every factor is — coverage distributes over union), and each covered `P` is
  paired with the memoised right family under `typebalance.admissible` (⋈).
  `Contexts(a)` is definitionally the ⋈-filtered product of the two marginals,
  so the lazy join realises exactly the concrete generators the slow path
  would — same `str`-sorted candidate order, hence *exact-list-equal*
  composites (stronger than the label-multiset bar).  Pinned by
  `test_cospan_compose_factored.py`, including a 2^20-left-context hub with a
  raising `left_bundles` spy (the lazy path must never materialise a side).

## Entry-point policy (the "fast default" flip)

- `cospan.from_ocpn.signature_from_ocpn`: **`canonical=True` by default** —
  the OCPN adapter's dominant use is the CanonKey level (compare, inventory,
  localisation), and the default must not hang on a wide discovered net.
  `canonical=False` opts back into the full L3 extraction for splice.
- `discover.signature_from_ocpn` / `discover_model`: **pinned
  `canonical=False`** — their established consumers feed
  `extract_classes`/splice, which need concrete ports; the migration
  discipline's byte-for-byte contract holds until those callers cut over
  (they pass `canonical=True` explicitly when they want the fast view).
- `engine.extract_signature` grows no dispatch flag: it is the oracle, and a
  `bindings=` toggle would collide with the OCCN Λ decoration vocabulary.

## L2 comparison: joint vs marginal keys

`compare(..., key="joint")` (default) aligns whole generators by CanonKey.
`key="marginal"` decomposes each key into per-`(activity, side, type)`
`ArityFact`s (0 = the type is optional there; an always-empty side is kept as
a `(label, side, None) {0}` presence fact so legless generators stay visible).
The joint set is a subset of the product of its marginals, with equality
exactly when types factor independently — always for discovered OCPNs, *not*
in general (cross-type XOR coupling is erased), which is why the joint key
stays the default.  Pinned by `test_cospan_compare_marginal.py`; the joint
path's byte-stability across the refactor was verified against the prior
implementation on dict and text renderings.
