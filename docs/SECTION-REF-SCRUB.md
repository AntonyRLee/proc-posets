# Scrub: internal design-document section references

Docstring references of the form `§NN` point at sections of an internal design
document that no reader of the package can see. Same defect as the manuscript
references cleaned in `beb4593`: the docstring should say the thing, not cite a
document the user does not have.

**Rule.** Delete the token, keep the content. `the §32 leg-multiplicity parameters`
becomes `the leg-multiplicity parameters`. Where the token is the *only* content
(`the §20 finding that ...`), name the fact instead. Never delete a clause just
because it carries a token.

**Total: 109 references across 24 files.**

Tick each line when its file is done and the suite passes.

## `procposets/cospan/_boundary.py` (1)

- [x] L10 `§40` — * ``GAMMA1``/``GAMMA2`` -- the §40 single master boundary source/sink labels.

## `procposets/cospan/class_extraction.py` (3)

- [x] L56 `§38` — """Left boundary as a weighted multiset (§38): leg ``p`` carries ``g.weight(p)``
- [x] L135 `§38/§39` — their own copy of activity ``a``, §38/§39) must NOT be merged into one group, or
- [x] L229 `§21` — # concurrent generators (§21) -- has been removed in favour of the categorical

## `procposets/cospan/compose.py` (2)

- [x] L94 `§38` — byte-for-byte -- including §38 grounding weights)."""
- [x] L100 `§38` — # Readiness/consumption respect leg weights (§38 grounding): a grounded

## `procposets/cospan/constraints.py` (2)

- [x] L1 `§32` — """Accessible builders for leg-multiplicity constraints (§32).
- [x] L80 `§32` — """The accumulated constraint system of a set of generators (§32): the union of

## `procposets/cospan/dag_diff.py` (7)

- [x] L5 `§19c` — false agreement on, e.g., an N-poset vs a complete bipartite order, §19c),
- [x] L7 `§19d` — §19d) up to isomorphism -- a genuine poset comparison.
- [x] L11 `§19e` — by the §19e certificate ``(anchor_types, body-DAG)`` so a loop's splice point
- [x] L14 `§19d` — share a language, §19d).
- [x] L28 `§19e` — §19e loop anchor-type signature (``()`` for closing instances, whose
- [x] L121 `§22h/§23b` — overlay/gallery figures (§22h/§23b).
- [x] L182 `§22h.` — # what must be forgotten here, §22h.)

## `procposets/cospan/discovery_cleanup.py` (6)

- [x] L45 `§38` — frozenset would drop one, so the collapsed leg records its count as a §38
- [x] L50 `§02b` — Applied as a comparison-time transform over the §02b-faithful fine
- [x] L62 `§38` — frozensets, so multiplicity above 1 can only be carried by §38
- [x] L107 `§32` — # carry the §32 leg constraints, remapping their ports through the
- [x] L122 `§13` — 1. **Explicit start/end generators** (OCCN's ``START_<ot>``/``END_<ot>``, §13) are
- [x] L205 `§40a` — that co-fire one-per-label (§40a), whereas a ``gamma2`` drain is zero-*right* and so is

## `procposets/cospan/engine.py` (2)

- [x] L37 `§40` — # explicit terminus activity (§40 single-boundary convention). There is deliberately no
- [x] L119 `§40a` — # distinct labels that co-fire one-per-label (§40a); minting a same-label ``gamma1``

## `procposets/cospan/extract_dp.py` (7)

- [x] L10 `§19/§20.` — ``occurrence.canonical_key`` built in §19/§20. So the right unit of
- [x] L17 `§40a` — already fired. **One origin per label** (§40a): same-label zero-left
- [x] L20 `§2` — process instance per closing. The Markov property (§2) holds on the state.
- [x] L226 `§21f` — """Pomset-DP extraction (§21f): the iso-class-quotiented catalogue.
- [x] L236 `§22` — (e.g. an OCPN of independent per-type subnets, §22) hits the cap, which
- [x] L257 `§40a` — # ``END_``). gamma2 drains are zero-right -> not single-firing-constrained (§40a is gamma1
- [x] L338 `§22` — if len(reps) >= max_pomsets_per_state:  # over-generation guard (§22): cap the

## `procposets/cospan/feasibility.py` (3)

- [x] L1 `§32` — """Integer feasibility for leg-multiplicity constraint systems (§32).
- [x] L82 `§34` — "raise max_assignments, lower the bound, or pass an ILP solver (§34)"
- [x] L123 `§38` — (the **unroller** backend, §38). Unlike :func:`ranges` -- which reports each

## `procposets/cospan/morphism_schema.py` (3)

- [x] L40 `§6` — to atomic generators, even through the §6 loop hierarchy."""
- [x] L94 `§38` — # §38 grounding: a leg carries g.weight(p) objects, so it consumes
- [x] L157 `§12` — members is explicitly out of scope here (§12 scope note) -- this only

## `procposets/cospan/occurrence.py` (15)

- [x] L11 `§14` — (§14), while still encoding causality -- a real poset comparison, not the
- [x] L12 `§19c/§19d` — lossy label-skeleton string comparison (§19c/§19d).
- [x] L21 `§19e` — (§19e), not represented as a back-edge here.
- [x] L23 `§23b` — Reading convention (§23b): every edge is **conjunctive (AND)**. At each event
- [x] L58 `§40` — # The single boundary nodes of a *closing* pomset (§40): all of a notation's origin
- [x] L88 `§40` — # connected γ1 source / γ2 sink (§40). This is a render/canonical-key concern only; it does
- [x] L112 `§19d` — the §19d label-pomset projection.
- [x] L117 `§19c` — §19c needs to keep (it is what distinguishes an N-poset from a complete
- [x] L121 `§13/§14` — ``START_<ot>``/``END_<ot>`` markers, §13/§14 B3). These are not merely
- [x] L128 `§20` — master/OCPN's by that one edge -- which is the §20 finding that kept the
- [x] L134 `§40` — # Origin/termination markers map onto the single γ1 source / γ2 sink (§40). A loop's
- [x] L164 `§38` — # weight-aware (§38): a leg of multiplicity w wires w tokens, so a
- [x] L307 `§19a` — the comparison *boundary-rooted* (§19a): admissible isos fix the
- [x] L319 `§19/§20d` — occurrence-net union / prefix-trie, §19/§20d) collapses shared prefixes
- [x] L346 `§19e` — """The §19e loop anchor projected to a type-multiset: the sorted

## `procposets/cospan/signature.py` (5)

- [x] L29 `§32` — the generator, keyed by port, not as ``Port`` fields (§32)."""
- [x] L44 `§32` — """A linear (in)equality over a generator's leg-multiplicity variables (§32).
- [x] L83 `§32` — ``constraints`` is the optional §32 decoration -- general linear inequalities
- [x] L87 `§38` — ``weights`` is the optional §38 **grounding**: a boundary is a *multiset* of typed
- [x] L110 `§38` — """Object-count on boundary leg ``port`` (§38 grounding); 1 if unset."""

## `procposets/cospan/signature_compare.py` (7)

- [x] L7 `§32` — this keeps the **generator inventory with its §32 leg-multiplicity parameters
- [x] L19 `§32` — and under that key we compare the per-``(side, type)`` binding intervals (the §32
- [x] L27 `§39` — differs by adapter convention (master's single γ1/γ2 interface §39 vs OCCN's
- [x] L28 `§40` — per-type START/END §40) -- a real, documented structural difference the matrix
- [x] L160 `§32` — """The §32 N-linear parameters of a generator, keyed by ``(side, type)`` so it is
- [x] L202 `§32` — """Extract :class:`BindingProfile` from one generator's §32 constraints.
- [x] L360 `§32` — §32 :class:`BindingProfile` payloads.  The 0-drift oracles pin these

## `procposets/cospan/signature_diff.py` (6)

- [x] L16 `§12` — 2. a within-signature dedup pre-pass reusing §12's ``schema_classes`` (one
- [x] L28 `§14` — degenerate miner noise, or genuine structural disagreement -- §14's B1-B5)
- [x] L31 `§13` — ``START_<ot>``/``END_<ot>`` boundary-generator convention added in §13) is
- [x] L69 `§12` — """§12's within-signature dedup, applied to one subset of a signature's
- [x] L216 `§9c/§11` — this purpose, but §9c/§11 found it never fires on a real signature at
- [x] L265 `§13` — §13's fix), but have no counterpart in an adapter whose own zero-left/

## `procposets/cospan/skeleton.py` (1)

- [x] L95 `§38` — separate §38 grounding, absent from any LM-graph extraction)."""

## `procposets/cospan/splice.py` (9)

- [x] L8 `§27c` — generating grammar of the model's trace language** (§27c): a run γ1→γ2 is a
- [x] L18 `§27` — causal pomset for series-parallel families (the OCCN ``72→2`` fix, §27).
- [x] L24 `§27b` — ``to_dict`` is byte-stable across runs and hash seeds (§27b determinism).
- [x] L91 `§40` — # node ids mix int (real events) and str (the gamma1/gamma2 boundary, §40), so
- [x] L137 `§27c` — must use ``pomset`` for exact linearization (§27c caveat)."""
- [x] L146 `§32` — constraints: tuple = ()  # the family's accumulated leg-multiplicity system (§32)
- [x] L166 `§32` — """``prune_bound`` (§32): if set, drop families whose accumulated leg
- [x] L190 `§32` — # §32: the family's leg-multiplicity system = the union of its baseline
- [x] L323 `§22h` — coordinates. Two mechanisms unioned (the §22h committed-vs-neutral split, at

## `procposets/cospan/trace_language.py` (1)

- [x] L8 `§27c` — Two honest scope limits, by design (§27c, user-accepted):

## `procposets/cospan/unroll_core.py` (3)

- [x] L1 `§38` — """Bounded grounding of a constrained signature (§38) -- the notation-agnostic
- [x] L25 `§38` — """Ground one generator into its weighted concrete instances (§38).
- [x] L46 `§36` — groundings never collapse while symmetric ones never duplicate (the §36

## `procposets/initialiser.py` (3)

- [x] L19 `§8` — **A refinement of §8, measured here.**  §8 motivated the ladder by claiming
- [x] L19 `§8` — **A refinement of §8, measured here.**  §8 motivated the ladder by claiming
- [x] L29 `§8.2` — §8.2) when the component count exceeds the low-order feature space, and the

## `procposets/occn/to_signature.py` (6)

- [x] L18 `§32` — **Bindings surfaced as constraints (§32).** A ``Port`` is a bare typed triple (the
- [x] L38 `§32` — """Surface the markers' bindings (§32) as linear constraints on the legs --
- [x] L39 `§42` — the generator's full **per-leg N-linear blueprint** (§42):
- [x] L57 `§36` — multiplicity, §36) *at composition/grounding*, not pre-resolved here. Earlier this
- [x] L60 `§42` — ``signature.json``/``cospans.svg``/the comparison (§42). The behavioural splice
- [x] L111 `§32` — ``bindings`` (default ``True``) surfaces the OCCN's multi-object machinery as §32

## `procposets/occn/unroll_occn.py` (5)

- [x] L1 `§38` — """OCCN faithful per-firing grounding (§38-39) -- the OCCN half of unrolling.
- [x] L22 `§38` — """Per-firing constraint system for context ``(ig, og)`` (§38): every marker's
- [x] L68 `§38` — """Mined OCCN -> grounded (weighted) signature (§38) with per-type
- [x] L77 `§39` — """The γ1/γ2 boundary of one *run* (§39): γ1 and γ2 are the boundary *objects* of
- [x] L108 `§39` — """Grounded signature for one run with per-type object content ``counts`` (§39):

## `procposets/viz/compare_vis.py` (2)

- [x] L53 `§32` — # chrome: the joint captions would mislabel arity SETS as §32 binding ranges
- [x] L144 `§32` — ("↓ input leg, ↑ output leg; ranges are objects-per-firing (§32); "

## `procposets/viz/dag_render.py` (6)

- [x] L184 `§20d` — """The true process-map overlay (§20d): the **causal-prefix-merged** union
- [x] L234 `§19e` — """Where each loop can be spliced into ``closing`` (§19e, Q1): a loop with
- [x] L258 `§22h` — (§22h) and return one ``(baseline, splices)`` pair per family.
- [x] L304 `§22d` — (§22d free-product + B4 self-bounces); without this, one splice site would
- [x] L328 `§27d` — :class:`~procposets.cospan.splice.SpliceRepresentation` (§27d): one cluster per
- [x] L409 `§27d` — it via :func:`render_splice_catalogue` (the single source of truth, §27d).

## `procposets/viz/string_diagram.py` (4)

- [x] L613 `§43` — # graphical N-linear bindings (§43): conservation drawn as a coupling arc over the box
- [x] L710 `§32` — """The generator's multi-leg §32 relations split for graphical rendering (§43):
- [x] L710 `§43` — """The generator's multi-leg §32 relations split for graphical rendering (§43):
- [x] L733 `§32` — """Per-leg §32 binding cardinality keyed by :class:`~procposets.cospan.signature.Port`,

