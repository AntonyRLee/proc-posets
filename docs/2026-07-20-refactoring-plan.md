# procposets — aggressive refactoring plan (2026-07-20)

Source: a 14-agent static review (10 subsystem readers + 4 cross-cutting
consolidations — naming, duplication, API/layering, efficiency) over all 54
non-test source modules (12 033 LoC). Findings verified by hand on the
load-bearing seams. This plan is **constraint-aware**: every item is tagged with
the invariants it must not break.

---

## 0. Guardrails (every change is gated on these)

These come from `CLAUDE.md` + the code and are non-negotiable while the package
is mid-extraction:

1. **Byte-for-byte value equivalence** with the three consumer repos
   (`poset-mixture-npmle`, `sim/cpm`, `stochastic_process_mining/spm`) until each
   is cut over. Every value-producing seam has a golden cross-check test. So each
   change carries a **risk tag**:
   - `none` — cosmetic/comment/docstring/type-hint only; cannot change a value.
   - `value-preserving` — refactor provably cannot change any output (byte-identical
     code moved, or reduction with **identical iteration/accumulation order**).
   - `changes-values` — could alter a computed number (e.g. float summation order,
     set-iteration nondeterminism). **Gate on the full golden suite; land only if
     unchanged.**
   - `api-breaking` — renames/moves a **public** symbol (`__init__.__all__`,
     `cospan/__init__`, public import paths). Needs a deprecation alias or a
     coordinated consumer bump.
2. **Do not unify the two renderers.** The SPTree `->`/`||` renderer (`rel.py`)
   and the moddecomp `;`/`*` renderer (`moddecomp.py`) are deliberately separate;
   their canonical strings are baked into downstream byte-exact regressions.
   Shared *primitives* under them (component-finding) are fine; the renderers are not.
3. **Keep the two poset views distinct.** id+label `Poset` (repeated-label-capable)
   vs `Rel = frozenset[(label,label)]` (certified distinct-label view via `bridge`).
4. **Preserve the lazy layering.** Dependency-free stdlib core; numpy via PEP-562
   `__getattr__`; networkx/pm4py/matplotlib behind extras. No heavy import may
   leak into the eager core.
5. **Resource rules for any *executed* verification.** One heavy process at a time,
   under `systemd-run … MemoryMax=3G MemorySwapMax=0 timeout <s>`, serial pytest,
   no parallel local compute. (This review was static; the stress-test phase in §7
   must obey this.)

**Working rule:** land in risk order — all `none` + `value-preserving` first
(mechanical, fully golden-gated), then `changes-values` one-at-a-time with the
suite as arbiter, then `api-breaking` last behind aliases at a consumer-coordinated
moment.

---

## Priority map at a glance

| Phase | Theme | Risk band | Effort |
|------|-------|-----------|--------|
| **0** | Hygiene + latent defects/safety | none → changes-values | S–M |
| **1** | Naming convention + safe renames | none → value-preserving | S–M |
| **2** | Deduplication | value-preserving (+ gated) | S–M |
| **3** | Maintainability, types, docs | none → value-preserving | M–L |
| **4** | API evolution (unify public names) | api-breaking (aliased) | M |
| **5** | Test/layer infrastructure | none → value-preserving | S |
| **6** | Structure/module moves | value-preserving → api-breaking | M–L |
| **7** | Efficiency/scalability stress map | (feeds the stress phase) | — |

---

## Phase 0 — Hygiene + latent defects (do first; unblocks everything)

### 0.1 Repo hygiene (`none`)
- **Remove the stray Split-Miner artifacts from inside the package tree**:
  `procposets/split-miner-all.zip`, `procposets/split-miner.zip`,
  `procposets/split-miner.md` (untracked scratch inside the importable package).
  Move to scratch/`docs/` or delete; add `procposets/**/*.zip` to `.gitignore`.
- **Exclude tests + stray files from the wheel**: `pyproject.toml` sets
  `packages = ["procposets"]` with no exclude, so `procposets/tests/` (the
  value-pinned golden corpus) and any stray file ship in the wheel. Add
  `[tool.hatch.build.targets.wheel] exclude = ["procposets/tests", "**/*.zip"]`.

### 0.2 Latent defects & safety (these are real; `changes-values` — verify + golden-gate)
1. **P0 — `extract_dp.extract_classes`: the documented `max_pomsets_per_state=512`
   cap is unimplemented** (`extract_dp.py:234`, `closing_pomsets` 296–341,
   `ExtractionResult` 353–355). `reps` grows unbounded; `truncated` is **always
   False**. A genuinely over-generating net has no valve → runs to `max_frontiers`
   or OOM, violating the resource rule. *Fix:* enforce the per-state cap in
   `closing_pomsets`, thread a `truncated` flag out, pass `truncated=True` into
   `ExtractionResult` (the field already exists at `class_extraction.py:214`). Add
   a bounded stress fixture under the cgroup cap.
2. **`extract_signature_fast` picks a nondeterministic representative Generator per
   CanonKey** (`engine_fast.py:204–214`: iterates unordered sets, `setdefault`-keeps
   first). The CanonKey *set* is stable but *which* `Generator` object represents it
   varies run-to-run. *Fix:* iterate `sorted(g.activities)` and sort `B`/`F` by a
   stable bundle key before `best.setdefault`. Changes which object is kept, not the
   key set — golden-gate.
3. **Import-time side effects** (`value-preserving`, but make explicit):
   - `adapters/outbound.py:23` — module-level `warnings.filterwarnings("ignore")`
     mutates process-global state on import. Scope it (context manager at call
     sites) or narrow to the specific warning.
   - `viz/spm_viz.py:20` / `viz/dag_render.py` — import-time `matplotlib.use("Agg")`
     (present in two of three viz modules). Consolidate into one viz init and apply
     consistently, or move into the render entry points.
   - ~~`grouping.group_by_key` prints diagnostics to stdout~~ — **corrected on
     inspection: this is intentional and contractual.** The docstring states the
     declared assumption "is printed," and `test_np_grouping_simulate.py` asserts
     on `capsys.readouterr().out`. Leave it as-is (a `logging` migration would be
     an api/behaviour change, not hygiene).
4. **Triage — 3 correctness smells that may be intentional byte-exact behaviour**
   (investigate against intent + goldens; do **not** blind-fix):
   - `compose` LoopBox branch bypasses the `seen` dedup and `end_label` filter
     (`compose.py:154–159`).
   - `morphism_schema.shape_key` replays firing with **unweighted**
     `Counter(g.left/right)`, diverging from the weighted `fire` used everywhere
     else (`morphism_schema.py:85–108`).
   - Timed sampler assumes all trees share `tree[0]`'s alphabet, undocumented
     (`simulate.py:83`). At minimum document the invariant + assert it.

---

## Phase 1 — Naming convention + safe renames

### 1.1 Adopt one package-wide convention table (`none` — the reference artifact)
Record the canonical name per concept (dominant spelling wins, verified against
`__all__`):

| Concept | Canonical | Replaces |
|--------|-----------|----------|
| e(P) extension count | `count_extensions` | `count_linear_extensions` (rel), `extension_count` (SPTree), `PosetClass.extension_count` |
| uniform sampler | `sample_extension` + input-typed variants | 3 same-named sigs (`_extensions`/`poset`/`rel`) + `sample_linear_extension` |
| SP composition | `series` / `parallel` (match `SPTree.kind`) | `then` / `par` (poset.py) |
| raw ordered pair-set (param) | `pairs` | `less`-as-param, stray `rel:` for raw pairs |
| poset receiver var | `P` (uppercase) | `p` (bridge.py) |
| model type | `Model = list[tuple[Poset, float]]` (one source) | 5 redefinitions incl. weak `Model = list` |
| mixture weights | `rho` | note `TrueMixture.weights`, internal `w` map to it |
| noise kernel | `noise_kernel` **or** `noise` (pick one) | the split Atom-`noise` / Oracle-`noise_kernel` |
| in-L indicator | `in_L` (method + param) | `inL`, `inL_cache` |
| activity / object type (occn) | `act` / `otype` | `a`/`b`/`name`/`t`, `ot`/`typ`/`o` |
| LMGraph / Generator vars | `lm`/`graph`, `gen` | overloaded single-letter `g` |
| our colour identifiers | British `colour(s)` | mixed (keep matplotlib `facecolor` American) |
| WL dag hash vs arity key | `canonical_key` (WL) / `canon_key` (CanonKey) | disambiguate `CompositeDiagram.canonical_key` |

### 1.2 **P0 — Fix the actively-misleading boundary predicates** (`value-preserving`, correctness hazard)
`is_boundary_label` (`signature_compare.py:43`, matches `gamma1/gamma2/START_/END_`)
and `_is_boundary_label` (`signature_diff.py:250`, matches `START_/END_` **only**)
are near-identical names with **different membership**. A wrong swap silently
changes which labels are boundary. *Fix:* introduce two unambiguously-named
predicates in `occurrence.py` — `is_start_end_marker` (== `BOUNDARY_PREFIXES`) and
`is_gamma_or_marker` (== `DAG_BOUNDARY_MARKERS`) — repoint each call site to the
correct one; replace open-coded prefix checks (`engine.py:211-212`,
`engine_fast.py:199-200`). **Do not merge the two predicates** — membership must
stay distinct.

### 1.3 Value-preserving internal-parameter sweep (`value-preserving`, one pass)
Private params/locals/aliases (not in any `__all__`): single `Model` alias
imported everywhere; `bridge` receiver `p → P`; cospan single-letter `g → lm/gen`;
occn `act`/`otype` spellings; splice `frags → by_name`. Largest low-risk
readability win; provably output-neutral.

### 1.4 Doc-only fixes for misleading-but-public names (`none`)
Rename is too risky, so pin the meaning in a docstring: `PosetClass` is called a
"Protocol" but is a concrete ABC (`rel.py:464`); `Marker.activity` names the
neighbour, not the owner (`markers.py:33`); `Parallel.atomic=True` reads backwards
(`moddecomp.py:125`); `estimate.reweight` is a Model constructor, not a reweighter
(`estimate.py:43`); distinguish `Atom.desc`/`Atom.describe()`/`rel.describe()`.

### 1.5 Real type aliases replacing runtime-string "aliases" (`none`)
`Step = "..."` (a plain string, never an annotation) at `class_extraction.py:97`,
`splice.py:59`; bare `Skeleton=tuple`, `State=tuple`, `Bundle=frozenset`
(duplicated `engine.py:31`/`typebalance.py:32`). Make real `TypeAlias`es or delete
if unused.

*(API-breaking renames — the e(P) spellings, `sample_extension` family,
`then/par`, `noise`/`noise_kernel`, `canonical_key` triple, viz `color_map`,
diagnostics report family — are deferred to **Phase 4** behind aliases.)*

---

## Phase 2 — Deduplication (value-preserving first; renderers untouched)

Ordered by confidence. Each keeps **iteration/accumulation order identical** so it
is byte-safe; the two float-sensitive exclusions are called out explicitly.

1. **Single-shot Bhattacharyya angle** (`value-preserving`):
   `distance.bhattacharyya_angle` (136–142) and `traces.trace_bhattacharyya`
   (48–53) are byte-identical bar the distribution source. Extract
   `_bhattacharyya_angle(p, q)` (same `set(p)|set(q)` order) + a `_clamp01` helper.
2. **Per-state squared-angle SMD kernel** (`value-preserving` — *with a guard*):
   `smd_rows` (69–77), `discrete._matrix_angle` (131–139), `discrete.order_angle`
   (100–107) all do `2·√(Σ acos²(clamp(bc)))` over a state list **in the same
   order** → share `angle_over(a1, a2, keys)`. **Exclude `distance._pairwise_rows`
   (100–110):** it iterates the *smaller* row dict after a swap — different float
   accumulation order → `changes-values`. Leave it as-is.
3. **Predecessor-set map → `_extensions.preds`** (`value-preserving`): the
   `{e: {a for (a,b) in rel if b==e}}` map is re-derived 5× (`simulate.py:87`
   rebuilt per trace, `likelihood.py:309`, `rel._ideals/_filters`,
   `traces.linear_extensions:20`). Route through the one canonical single-pass
   builder (frozenset vs set is membership-equivalent).
4. **Feasibility bounded-enumeration scaffold** (`value-preserving`):
   `feasibility.solve/all_solutions/ranges` share an identical prologue +
   `itertools.product` body; only the accumulation differs. Extract
   `_enumerate(cons, *, bound, lo, pinned, max_assignments)` → `solve = next(gen,
   None)`, `all_solutions = list(gen)`, `ranges = fold`. Same domains, product
   order, filter. (Also fixes the guard being maintained in three places.)
5. **OCCN→cospan lift helpers** (`value-preserving`): `occn/to_signature.py` and
   `occn/unroll_occn.py` share four **character-identical** helpers (`_types`,
   `_type_balanced`, `_in_port`, `_out_port`) + a near-identical boundary scaffold
   (its own docstring says "Mirrors …"). Extract `occn/_lift.py`. **Keep the
   constraint *tails* separate** (per-key partition vs per-type conservation).
6. **Cospan boundary constants single-source** (`value-preserving`): `gamma1/gamma2`
   live in ≥4 places with an explicit "kept in sync to avoid an import cycle"
   comment (`engine.py:38`). Add a dependency-free `cospan/_boundary.py`
   (`GAMMA1/GAMMA2/BOUNDARY_PREFIXES`) that breaks the cycle cleanly. (Predicates
   stay two — see 1.2.)
7. **Disjoint-set / components primitive** (`value-preserving` — *ordering-sensitive*):
   union-find bodies in `moddecomp._components` (158–175) and
   `engine_fast._components` (70–102) are near-identical. Share a path-halving
   `connected_components(nodes, edge)` **only if it reproduces output ordering
   byte-for-byte** (moddecomp's series `rank` relies on component insertion order).
   Leave `rel.decompose`'s DFS (275–291) alone unless byte-reproducible. **Do not
   touch either renderer.**
8. **Adapter build-and-validate wrappers + type-prefix** (`value-preserving`): six
   byte-structurally identical `lmgraph_from_*` wrappers (`from_bpmn.py:89`,
   `from_process_tree.py:110`, `from_petri.py:62`) → `_assemble` /
   `_assemble_single`; triplicated `pre = f"{otype}__" if otype else ""` →
   `_type_prefix(otype)` (the emitted string is load-bearing for signature
   comparison — keep it exact).
9. **rel canonical sort key + `_ideals`/`_filters`** (`value-preserving`): inline
   `lambda r: (len(r), sorted(r))` 3× → module-level `_canonical_key`; `_ideals`
   and `_filters` are the same 2ⁿ scan with down/up swapped → one
   `_closed_sets(elements, rel, direction)`.
10. **Single `Model` type alias** (`value-preserving`): define once, import in
    `estimate/loops/distance/matrix/adapters.outbound`; fix `traces.py:15` weak
    `Model = list`.

**Explicitly NOT deduplicated:** the two renderers (guardrail 2); the Poset/Rel
split (guardrail 3); `matrix.build`'s window/normalise vs
`discrete._build_refined`'s (string-keyed, byte-exact — merging is
`changes-values`).

---

## Phase 3 — Maintainability, types, docs

### 3.1 Decompose the god-functions (all `value-preserving` — pure extraction)
Long, multi-responsibility functions to split into named private steps (behaviour
unchanged, golden-gated):
- `Oracle.__init__` (~100 lines: regime dispatch + atom build + logF stacking) →
  `oracle.py:95`; also lift the brittle string-keyed regime state machine
  (`oracle.py:141`) into an enum/dispatch table.
- `npmle.fit` (long, nested re-pricing loop, many bare magic constants) —
  name the budgets (`INNER_ITERS=3000`, `tol=1e-13`, `INNER_REPRICE_ITERS=20`,
  entering weight `1e-3`) as constants so the coupling is visible (`npmle.py:194`).
- `extract_dp.extract_classes` (~120 lines of side-effecting nested closures over
  shared mutable frames) (`extract_dp.py:233`) — see also 0.2.1.
- `occurrence.to_event_dag` (~85 lines, nested closure, deep nesting)
  (`occurrence.py:99`).
- `splice.SpliceRepresentation.from_extraction_result` (~100 lines, five
  interleaved responsibilities) (`splice.py:161`).
- `discrete._build_refined` (long, string-keyed) — extract but **`changes-values`
  risk**: gate on the refinement goldens (`test_refinement.py`).
- `viz/string_diagram.render` (~170 lines) (`string_diagram.py:838`).

### 3.2 Type hints sweep (`none`)
18 medium type findings. Priorities: parametrize bare `dict`/`frozenset`/`tuple`
on public surfaces (`feasibility.py:65`, `signature.py:58`, splice/trace dataclass
fields, adapters); `Literal[...]` for string-enum params (`loops.py`, `distance`
`mode`, `discrete` weighting); return annotations on `moddecomp.decompose` (add a
`Node` union alias) and the distance/discrete/viz public functions; fix
`npmle.py:207` `Optional[object]`. Consider shipping `py.typed`.

### 3.3 Docs sweep (`none`)
12 medium doc findings. Add docstrings to undocumented public symbols (`Poset` +
methods, `Oracle`, `TrueMixture` + samplers, `trace_*`, `mine_occn`/`mine_ocdg`,
OCCN dataclass, `pairwise_emd`). Fix stale/broken Sphinx refs: `cpm.` namespace
throughout cospan/occn after extraction; broken `:func:occn_leg_constraints`
(`constraints.py:11`); `from_petri` cross-refs resolving to the wrong package.
Document value-contracts/units where only an inline comment holds the invariant
(the `Poset` transitive-closure invariant; the golden viz value-contract).

### 3.4 Dead code (`value-preserving`)
`moddecomp` `atomic` property (unused package-wide); `discrete` unused
`linear_extensions` import; `extract_dp._nontrivial_sccs` (defined, never called);
`grouping` unused `Callable`/`Sequence` imports; `equivalence._activity_pomsets`
dead `chain` payload + unreachable branch. Confirm no consumer touches each before
deleting.

---

## Phase 4 — API evolution (api-breaking; aliased, consumer-coordinated)

Land only behind **deprecation aliases** (old names re-exported from `__all__`),
removed at each consumer's cut-over — never rename-in-place:
- Unify e(P) on `count_extensions`; keep `count_linear_extensions`/`extension_count`
  as documented aliases.
- Input-typed `sample_extension_tree`/`_poset`; keep exported `sample_extension`
  aliased to the SPTree sampler.
- `series`/`parallel` canonical; `then`/`par` aliased.
- One spelling of `noise_kernel` (Atom/Oracle/fit) + one-release kwarg alias.
- `in_L` for method + param (`inL`, `inL_cache`).
- Disambiguate the three `canonical_key`/`canon_key` symbols; rename or delete
  `CompositeDiagram.canonical_key` (no in-repo consumer — confirm external first).
- viz public `color_map → colour_map` with back-compat alias; centralize the
  palette in `viz/palette.py` (same hexes → identical output).
- Normalize the diagnostics report family (`report_vs_trivial → trivial_report`
  aliased) and surface it consistently in `_LAZY`/`__all__` (`__init__.py:116,162`).

---

## Phase 5 — Test & layer infrastructure (`none` → `value-preserving`)

- **Extend `test_lazy_numpy.py`**: it guards only numpy. Add
  `test_stdlib_core_is_graph_and_viz_free` (assert `networkx`/`matplotlib` not in
  `sys.modules` after importing the eager core) — matters because
  `equivalence.py` imports networkx eagerly at the package top level. Make the
  layering contract executable in both directions.
- Establish the **golden-diff harness** as the gate for Phase 2/3: run the full
  suite (`systemd-run … timeout 300 uv run pytest -q`) before/after each
  `value-preserving`/`changes-values` change; a byte diff in any golden blocks the
  landing.

---

## Phase 6 — Structure & module moves (value-preserving → api-breaking)

- **Relocate `equivalence.py` → `cospan/equivalence.py`** (it's networkx-eager,
  `cospan.signature`-dependent — a B1 module mis-placed beside the core). Keep a
  thin re-export shim at the old path (public import path used by
  `test_golden_cospan.py:111`) → api-breaking without the shim.
- **`from_petri` placement**: it lives in `cospan/` while the other two inbound
  adapters live in `adapters/`. Either re-export `lmgraph_from_petri` from
  `adapters/__init__` (value-preserving) or move + shim.
- **Move `_log_mean_exp_rows`** (a shared numerical primitive) out of `oracle.py`
  into a small `_numerics` module imported by both `oracle` and `npmle`.
- **Split the two largest modules along existing seams** (value-preserving,
  re-export the public surface so imports are unchanged):
  - `viz/string_diagram.py` (1172 LoC) → `viz/_layout.py` (term DSL +
    `Layout`/`PlacedBox`/`Wire` + layout fns) vs matplotlib drawing. Figures are
    not byte-compared, so a structure split is safe. *(The ~14 module-level mutable
    style globals should become a config dataclass — but that's `api-breaking`:
    do it in Phase 4.)*
  - `rel.py` (582 LoC, ~6 responsibilities) → `rel_sp.py` (SPTree machinery) +
    `rel_classes.py` (`PosetClass` hierarchy), relation-set algebra + DP wrappers
    stay in `rel.py`; re-export everything so `__init__.py:29-54` is unchanged.
- Refresh drifted `__init__` docstrings: `cospan/__init__` says `unroll` (→
  `unroll_core`) and omits `engine_fast` (which it eagerly imports); `occn/__init__`
  still references pre-extraction `cpm.cospan` / `sim/occn_dev/`.

---

## Phase 7 — Efficiency & scalability stress map (seeds for the stress-test phase)

The follow-up work you flagged. **Do this after the refactor lands** (stable base
+ green goldens) and **under the resource rules** (serial, cgroup-capped, guarded;
a timeout is a finding, not a knob). The package already ships the reference
pattern to emulate: `engine.extract_signature`'s `|B|×|F|` blow-up is fixed by the
**guarded exact twin** `engine_fast.extract_signature_fast` (component decompose +
`_ENUM_CAP` fallback + golden CanonKey cross-check). Treat the slow engine as the
stress **baseline**, not a fix target, and clone that twin+cross-check shape for
the other seams.

Ranked hotspots (all speedups must be `value-preserving`; float-order ones flagged):

| # | Site | Cost / blow-up dimension | Fix | Risk |
|---|------|--------------------------|-----|------|
| 1 | `extract_dp.closing_pomsets` | unbounded WL+VF2 iso-dedup, **cap unimplemented** | enforce cap (see 0.2.1) | changes-values |
| 2 | `feasibility.solve/all_solutions/ranges` | `product(*domains)` exponential in free legs; ILP escape on only 1 of 3 | shared `_enumerate` + constraint-directed pruning; extend `solver=` hook | value-preserving |
| 3 | `_extensions.ideal_state_bound` / ideal DP | rebuilds `succs` every chain-peel; recursion to chain depth (RecursionError before width guard fires) | index `succs` once; iterative DP; note guard bounds width not depth | value-preserving |
| 4 | `simulate.one_timed` | rebuilds `preds` per trace (G·n_g×); `enabled` rescanned O(L²·deg) | hoist `preds` per component; incremental frontier (keep rng call order) | value-preserving |
| 5 | `discrete._pair_relation` | rebuilds label→element inverse per (pair × variant) | hoist inverse per Poset; cache `_covers` per prime | value-preserving |
| 6 | `traces.linear_extensions` / `trace_distribution` | **unguarded** factorial enumeration (bypasses the ideal-budget guard) | reuse `_extensions.preds` + `check_ideal_budget`; per-Poset memo | value-preserving |
| 7 | `distance.smd_rows` | O(\|X\|²) dense vs `_pairwise_rows`' sparse skip | share `_row_angle`, skip equal rows — **only if one identical loop order** | changes-values |
| 8 | `npmle._fully_corrective` | up to 3000 iters × FW × W14(≤20) × B(200 bootstrap) | warm-start `w` from prior FW iterate; **do not change logsumexp reduction order** | value-preserving |
| 9 | `initialiser.moment_seed` | recomputes all lower-order moments per escalation order (O(m!) each) | cache per (candidate, order); extend tensor when climbing k | value-preserving |
| 10 | `likelihood.swap_kernel` / timed `group_logf` | O(m!·m) (walled m≤8, cached); pure-Python timed double loop hot in polishing | vectorize/cache timed `group_logf`; keep swap off the hot path | value-preserving |
| 11 | `rel` order-algebra | `is_partial_order` O(\|rel\|²); `transitive_reduction` rebuilds vertex set per pair; `_ideals/_filters` 2ⁿ rebuilt per candidate in `enumerate_posets` | index by middle element; hoist node set; memoize closed-sets | value-preserving |
| 12 | `signature.Generator.weight/constrained_ports` | O(legs) scan per token in occurrence replay → O(legs²)/frame | build `dict[Port,int]` once (cached) | value-preserving |
| 13 | low-tier: `poset._transitive_closure` O(n³) fixpoint; `LMGraph` edge scans (`without_silent` O(silents·\|E\|²)); `markers._index` recomputed per direction + unused `objs_of_event`; topo-sort caps (`splice` `cap_nodes=11`, `trace_language`) | small inputs today; name the caps | Warshall index; index edges once; build OCEL index once | value-preserving |

**Stress harness note:** the caps in #13 (`cap_nodes=11`, `max_traces`,
`max_assignments=200_000`, `MAX_IDEAL_STATES`) are load-bearing correctness/safety
limits — probe *at* them, don't silently raise them; a hit is a finding.

---

## Sequencing summary

```
Phase 0  (safety + hygiene)        ──► unblocks a clean base
Phase 1  (naming + safe renames)   ──► readability, no value risk
Phase 2  (dedup, value-preserving) ──► golden-gated, one PR per cluster
Phase 3  (maintainability/types/docs)
Phase 5  (layer guard tests)       ──► run continuously as the gate
Phase 6  (module moves)            ──► behind re-export shims
Phase 4  (API unification)         ──► LAST, aliased, at consumer cut-over
Phase 7  (stress testing)          ──► after the refactor stabilises
```

Every non-`none` change ships with a golden cross-check confirming new == old on
the fixed corpus, per the migration discipline.

---

## Execution log

### Phase 0 — landed 2026-07-20 (suite: 339 passed, 20 skipped, 0 failures)
- **0.1 Hygiene** ✅ — moved `split-miner{,-all}.zip`/`split-miner.md` out of the
  package to `sandbox/split-miner/` (preserved, not deleted — they belong to the
  parked Split-Miner investigation); `.gitignore` now blocks `procposets/**/*.zip`,
  `procposets/**/*.md`, `sandbox/`; wheel `exclude = ["procposets/tests", "**/*.zip",
  "**/*.md"]`.
- **0.2.1 P0 — `extract_dp` per-state cap** ✅ — `closing_pomsets` now enforces
  `max_pomsets_per_state` and threads a `truncated` flag into `ExtractionResult`.
  Verified: default (512) → running example still 12 closings, `truncated=False`
  (byte-unchanged); cap 1/2/3 → bounded + `truncated=True`. New regression
  `test_cpm_extract_cap.py` (3 tests) pins both. *Note:* the cpm byte-exact
  cross-check for `extract_classes` skips here ("sim/cpm not checked out"); the
  default path is provably untouched, and that cross-check will confirm it when
  sim/cpm is present.
- **0.2.2 `extract_signature_fast` determinism** ✅ — iterate `sorted(g.activities)`
  and sort `B`/`F` by a stable bundle key before `best.setdefault`. Gated green by
  `test_cpm_extract_fast` (CanonKey set is order-invariant).
- **0.2.3 Import-time side effects** — `grouping` print is **intentional/pinned**
  (see above), left as-is. `outbound.filterwarnings("ignore")` and the
  `matplotlib.use("Agg")` inconsistency are **deferred**: value-preserving but live
  in [pm4py]/[graph] modules that don't import in this dev env, so they can't be
  golden-gated here — do them deliberately with those extras installed.
- **0.2.4 Three correctness smells** — **not touched** (need intent triage, may be
  deliberate byte-exact behaviour): `compose` LoopBox dedup/filter bypass;
  `morphism_schema.shape_key` unweighted `Counter`; timed-sampler shared-alphabet
  assumption. Recommend investigating against the geometry/estimation intent + the
  cpm goldens before any change.

### Phase 1a — boundary-predicate hazard (1.2), landed 2026-07-21 (suite: 343 passed, 20 skipped)
The P0 naming hazard: `signature_compare.is_boundary_label` (gamma-inclusive) and
`signature_diff._is_boundary_label` (`START_`/`END_` only) were near-identical
names with **different membership**. Fixed by renaming in place to self-documenting,
membership-matching names — **bodies byte-identical (value-preserving)**:
- `signature_compare.is_boundary_label` → **`is_gamma_or_marker`**; kept
  `is_boundary_label = is_gamma_or_marker` as a back-compat alias.
- `signature_diff._is_boundary_label` → **`_is_start_end_marker`**.
- New `test_cpm_boundary_predicates.py` (4 tests) pins the distinct membership +
  the alias.
- **Deferred:** single-sourcing the constants + predicates into a dependency-free
  `cospan/_boundary.py` (plan §2.6) — a bigger surface across byte-exact-critical
  modules whose cpm cross-checks skip in this checkout; do it with the consumer
  repos present for full gating.

### Phase 1 (cont.) — Model alias, landed 2026-07-21 (suite: 343 passed, 20 skipped)
`Model = list[tuple[Poset, float]]` was redefined 4× verbatim and weakened to
`Model = list` in `traces`. Single-homed in `poset.py`, imported everywhere;
dropped the now-unused `Poset` import from `distance`/`outbound`. Annotation-only
→ value-preserving. Commit `b081eb3`.

### Phase 2 — deduplication (gate-able parts), landed 2026-07-21 (suite: 343 passed, 20 skipped)
Each change gated two ways where float-sensitive: fixed-seed (`PYTHONHASHSEED=0`)
before/after `repr()` capture **byte-identical** + full suite green.
- **Bhattacharyya-angle kernel** (`c779f7c`) — 6 open-coded copies → 3 helpers
  (`_clamp01`, `_row_angle`, `_bhattacharyya_angle`) in `distance.py`; repointed
  `smd_rows`, `bhattacharyya_angle`, `discrete._matrix_angle`, `discrete.order_angle`,
  `traces.trace_bhattacharyya`, and the clamp in `_pairwise_rows`. **`_pairwise_rows`
  keeps its distinct smaller-row summation order — excluded on purpose** (folding it
  in is `changes-values`). Capture: 23/23 result lines identical.
- **preds routing** (`2667dfe`) — `traces.linear_extensions` + `simulate.one_timed`
  now use `_extensions.preds`. Capture: linear_extensions ×4 posets + timed sampler
  ×3 seeds byte-identical. **Remaining:** `likelihood._k_vectors`; `rel._ideals`/
  `_filters` (also want the `_closed_sets` down/up merge) — follow-up.
- **feasibility scaffold** (`90d9e90`) — `solve`/`all_solutions`/`ranges` triplicated
  enumeration → one private `_enumerate` generator; unified the `FeasibilityTooLarge`
  message (untested text). `test_cpm_feasibility` 6/6.

- **rel canonical key + `_ideals`/`_filters` merge** (`5733672`) — inline
  `(len(r), sorted(r))` ×3 → `_canonical_key`; `_ideals`/`_filters` → one
  `_closed_sets(elements, rel, *, down)`. Capture: enumerate_posets/enumerate_sp/
  meet_closure/_ideals/_filters byte-identical (15 lines).
- **likelihood preds** (`85a6b74`) — `_k_vectors` → `_extensions.preds` (3rd/last
  core preds site).
- **UnionFind primitive** (`bceb94c`) — `moddecomp._components` +
  `engine_fast._components` → shared `procposets/_unionfind.py`, grouping order
  preserved exactly. Capture: moddecomp `tiling()`/`decompose()` over 8 posets
  byte-identical. **Renderers stay separate — only the primitive is shared.**

**Phase 2 deferred (need consumer-repo gating or bigger surface):** OCCN `_lift.py`
helpers and adapter build/validate wrappers ([pm4py]/[graph] — skip standalone);
`cospan/_boundary.py` constants single-source; tree-block flatten skeleton
(`matrix._block_sequence`/`discrete._block_items` — windowing part is
`changes-values`). **→ The first three landed 2026-07-21 once the golden-path fix
made them byte-verifiable; see "Phase 2 leftovers" in the execution log below. Only
the `changes-values` tree-block flatten remains deferred.**

### Phase 3 — maintainability, types, docs, landed 2026-07-21 (suite: 343 passed, 20 skipped)
Each split is a pure extraction; float/structure-sensitive ones gated by a fixed-seed
before/after value capture **byte-identical** + full suite.
- **Dead code** (`a0b55f6`) — unused imports (discrete `linear_extensions`, grouping
  `Callable`/`Sequence`), dead `extract_dp._nontrivial_sccs`, dead `chain` payload +
  unreachable branch in `equivalence._activity_pomsets`. Kept public `.atomic`
  (no in-repo reads but a consumer may use it).
- **God-function splits** (5): `Oracle.__init__` → `_select_candidates` + `_build_atoms`
  (`92344ac`, captured across 7 regimes); `npmle.fit` → named FW budgets +
  `_tighten_restricted` (`3f6acd2`, weights@17dp identical); `discrete._build_refined`
  → `_refined_step` (`9094a2f`); `to_event_dag` → `_assemble_event_dag` (`f7388f4`,
  10 direct occurrence tests); `splice.from_extraction_result` → `_build_loop_fragments`
  (`b464805`, 464-line to_dict identical).
- **Docs** (`c6ef417`, `fb2b1c2`) — fixed all stale `cpm.` Sphinx refs (→ `procposets.`),
  demoted 4 external refs, fixed broken `:func:occn_leg_constraints`, `PosetClass`
  Protocol→ABC, drifted `cospan`/`occn` `__init__`; added docstrings to `Poset`,
  `TrueMixture`, `Oracle`, `moddecomp.decompose`, `traces.*`; clarified misleading
  `reweight`/`Marker.activity`; made the two runtime-string `Step` aliases real; typed
  the `mode`/`weighting` string-enums with `Literal`.

**Phase 3 deferred, with rationale:**
- `extract_classes` — left as-is: already factored into `_mint`/`closing_pomsets`
  nested closures over heavily-shared mutable state (memo/path/scc); further extraction
  is net-negative on risk. Its safety was already improved (the Phase-0 cap).
- `string_diagram.render` split — belongs with the Phase-6 `viz/_layout.py` file-split;
  its figures are **not** byte-pinned by any golden here, so a split can't be verified
  in this checkout.
- Long-tail type annotations on graph/viz internal helpers (bare `dict`/`tuple`/
  `frozenset` params) — lower value; the load-bearing ones already carry `# real-type`
  comments.

### Phase 5 — layer-guard tests, landed 2026-07-21 (suite: 349 passed, 20 skipped)
Test-only, `value-preserving` (no production code touched → no golden VALUE can move).
Made the layering contract executable in **both directions** for all three heavy
extras, extending `procposets/tests/test_lazy_numpy.py` from a numpy-only guard.
- **Generalized the subprocess helper** — the inline `_numpy_imported(code)` became
  `_fresh_import_pulls(setup, dep)`; the 3 existing numpy subprocess tests were
  repointed to it with **byte-identical generated code** (`{dep!r}` → `'numpy'`), so
  their pass/fail behaviour is unchanged; the in-process cache test is untouched. Two
  single-source-of-truth constants `_STDLIB_CORE` / `_COSPAN_ALGEBRA` back both the
  numpy and the graph/viz/pm4py negatives off the **same** module lists.
- **3 new negatives** (assert *absence* → correct on a minimal install, no
  `importorskip`): `test_stdlib_core_is_graph_and_viz_free` (the plan-named test:
  networkx + matplotlib), `test_stdlib_core_is_pm4py_free`, and
  `test_pure_cospan_algebra_is_graph_free` — the last is the non-obvious, drift-prone
  one: the B0 algebra (engine/signature/compose/signature_compare) stays networkx-free
  **even though** its direct siblings `cospan.occurrence`/`cospan.trace_language` each
  carry a top-level `import networkx as nx`.
- **3 new positives** (each `pytest.importorskip`s its extra → skip cleanly on a
  minimal install) witnessing non-vacuity: networkx via `equivalence` **and**
  `cospan.occurrence` (the latter pins the algebra guard), matplotlib via
  `viz.string_diagram`, pm4py via `adapters.from_bpmn`.
- **Empirically established** (serial, cgroup-capped, fresh-subprocess `sys.modules`
  probes) the ground truth the tests encode: eager core (`import procposets`, the
  stdlib set, the pure cospan algebra) pulls **none** of numpy/networkx/matplotlib/
  pm4py; touching each layer pulls its dep. **Non-vacuity mutation-checked**: injecting
  `import networkx` into the eager `__init__` made the graph/viz guard FAIL as designed
  (then reverted). All new positives run (not skip) in this checkout — the 20 skips are
  the unchanged [graph]/[pm4py] **cpm cross-check** goldens, orthogonal to these guards.
- **Scope note:** deliberately did **not** add matplotlib-free / pm4py-free *cospan*
  guards (no cospan-adjacent module pulls those, so they'd be vacuous; the stdlib-core
  negatives already cover their absence). Helper kept `.strip() == "yes"` (not a
  last-line compare) — traced-clean imports print no banner, so the simpler byte-exact
  form is preferred.

**Phase 5 remaining (from the plan, not done here):** the numpy-eager-guard already
lived; the plan's second bullet (establish the golden-diff harness as the Phase 2/3
gate) was operational throughout Phases 0–3 and needs no code. `equivalence.py`'s own
eager top-level networkx import is *not* a core leak (it is not on the eager path) and
its relocation to `cospan/` is Phase 6.

### Consumer-repo reality check, 2026-07-21 (corrects the "not checked out" framing)
All three consumer repos are **on disk** and two are **already cut over** to the
editable `../procposets` path dep, so procposets changes hit them immediately:
- `poset-mixture-npmle` (`main`) — ported; `poset_mixture/__init__` is a re-export
  shim, originals deleted. Suite **201 passed** against current procposets.
- `stochastic_process_mining/experiments` (`arl/main`) — ported; `spm/__init__` shim.
  Suite **54 passed** (2 files fail collection on a missing `demo/11_synthetic_fleet/
  signature_lift.py` — a pre-existing spm gap, unrelated to procposets).
- `string-diagram-process-mining/sim` (`arl/sosym`) — **NOT** ported; still runs its
  own `cpm/` copy (the source-of-truth for the [graph]/[pm4py] originals).

Consequence for the goldens: the repo was renamed `-DIAGRAM-…-v2/sim` →
`string-diagram-process-mining/sim`, so the cospan/unroll/viz goldens were skipping on
a **stale path**, not missing code (`c440f4c` repoints them → +7 byte-exact cross-checks
now green). The other 13 skips are estimation/poset goldens whose originals were
**deleted** from the two cut-over repos — vestigial (their "not checked out" messages
are now misleading; a future cleanup could drop or re-message them).

### Phase 6 — structure & module moves, landed 2026-07-21 (suite: 356 passed, 13 skipped)
Value-preserving, each behind a re-export shim; gated on the procposets suite (incl. the
now-live cospan goldens) and both ported consumer suites at the end (201 + 54, unchanged).
- **Golden path fix** (`c440f4c`) — see above; test-only, un-skipped 7 cross-checks.
- **`equivalence.py` → `cospan/equivalence.py`** (`10b028e`) — networkx-backed,
  cospan.signature-dependent B1 module; thin shim at `procposets.equivalence` keeps the
  public path. cpm.equivalence golden cross-check now green.
- **`_log_mean_exp_rows` → `procposets/_numerics.py`** (`10b028e`) — shared numpy
  primitive out of `oracle`; imported by both `oracle` and `npmle` (both re-import it so
  `oracle._log_mean_exp_rows` / `npmle._log_mean_exp_rows` still resolve).
- **`from_petri` re-export** (`10b028e`) — `lmgraph_from_petri`/`_nets` surfaced from
  `adapters/__init__` so all three inbound `lmgraph_from_*` share one namespace.
- **`rel.py` split** (`f148d02`) — byte-exact line-slice into `rel.py` (relation algebra
  + ideal-DP) + `rel_sp.py` (SP-tree view) + `rel_classes.py` (hypothesis classes);
  re-export at the base keeps `procposets.rel` and the `poset_mixture.posets` shim
  unchanged; numpy-free-core guard still green. Dropped a dead `math.comb` import.

**Phase 6 — `viz/string_diagram.py` split, LANDED 2026-07-21 (suite: 359 passed, 13 skipped):**
was blocked on Phase-4 item 9 (the 14 style globals → `LayoutStyle`+`DrawStyle` dataclasses);
that landed (`842bbd2`), so the layout functions are now fully parameterized by a
`LayoutStyle` arg and read no globals. Split the 1229-LoC file into a new **matplotlib-free
`viz/_layout.py`** (term DSL `Diagram`/`pick`/`D`/`gens`, the `Layout`/`PlacedBox`/`Wire`/`_Sub`
datatypes, `LayoutStyle`, the layout fns `_box_sub`/`_seq`/`_par`/`_gen_delta`/`_ports`, and the
two lowering paths `_finish`/`_layout_composite`/`_consumed_later`, plus the shared geometry
constants `_BW`…`_RISER`) + the matplotlib drawing half that stays in `string_diagram.py`
(`render`/`catalogue`/`_draw_wires`/bezier+crossing geometry/gid/`DrawStyle`/`StringDiagramStyle`/
the 14 legacy override globals + `_style_from_globals` bridge). `string_diagram.py` re-exports
every moved name so `string_diagram.NAME` resolves unchanged. **Verification (byte-exact on
values):** every one of the 38 original top-level def/class bodies moved VERBATIM (AST source-
segment equality, 0 changed); a layout-coordinate capture over 8 diagrams × 5 layout-knob
variants + 2 composite DAG runs is byte-identical (same sha256) before/after; new permanent
guard `test_string_diagram_layout_half_is_matplotlib_free` locks `viz._layout` backend-free
(paired with the existing `…pulls_matplotlib` positive). Consumer surface audit (10→4-repo
read-only fan-out): only external consumers are the procposets viz goldens (5 public names:
`D`/`render`/`StringDiagramStyle`/`LayoutStyle`/`DrawStyle`) and the off-disk SDPM demo
`recovery.py` (`catalogue`) — no private-helper use, no override sites anywhere. End-gate:
PMN **201**, SPM **54**, unchanged.

**Phase 6 note — override globals kept (Item-9 bridge, deliberate):** the 14 module globals
(`STRAIGHT_SPINE`…`TYPE_LANES`) + `render(style=None)→_style_from_globals()` stay in
`string_diagram.py`; the documented `sd.STRAIGHT_SPINE = True; render(...)` path is byte-identical.
Removing them is a separate api-break (an off-disk manuscript/tikz repo is still ungreppable for
overrides), not part of this value-preserving split.

### Phase 2 leftovers — the now-unblocked [graph]/[pm4py] dedups, landed 2026-07-21 (suite: 356 passed, 13 skipped)
Previously deferred as "can't byte-verify standalone"; the Phase-6 golden-path fix
(`c440f4c`) made them verifiable against the live `sim/cpm`, and all three run under the
now-live cospan goldens. Consumer end-gate re-run: poset-mixture-npmle **201 passed** +
spm/experiments **54 passed**, both unchanged.
- **OCCN→cospan lift helpers → `occn/_lift.py`** (`751852b`) — `to_signature.py` and
  `unroll_occn.py` carried char-identical `_types`/`_type_balanced`/`_in_port`/`_out_port`
  and a byte-identical `_boundary_generators` (unroll's docstring literally said
  "Mirrors …"). Extracted all five; constraint *tails* stay separate (`_leg_constraints`
  per-key partition vs `_firing_system` per-type conservation). `_lift` imports only down
  into `cospan.signature` (miner→algebra preserved). Gate: PYTHONHASHSEED=0 before/after
  canonical-key capture over a synthetic OCCN corpus (occn_to_signature bindings on/off,
  ground_occn, ground_run, gamma_boundary) byte-identical. `_in_port`/`_out_port` stay
  importable from `to_signature` (re-export) so `test_cpm_constraints` is unaffected.
- **Boundary labels → `cospan/_boundary.py`** (`d575ccb`) — `GAMMA1`/`GAMMA2`/
  `BOUNDARY_PREFIXES` lived in ≥2 places; `engine.py` kept a local `GAMMA2 = "gamma2"`
  with a "kept in sync … to avoid an import cycle" comment (the cycle being that
  `occurrence` imports networkx, so numpy-only `engine` couldn't import from it). New
  dependency-free leaf both layers import: engine/engine_fast (via re-export) stay
  networkx-free, occurrence/splice re-export the objects. `DAG_BOUNDARY_MARKERS` now
  derives as `BOUNDARY_PREFIXES + (GAMMA1, GAMMA2)`. Identity constants → value-neutral;
  layer guards + B0/B1 goldens green.
- **Adapter build-and-validate wrappers → `cospan/_lmgraph_build.py`** (`9853505`) — the
  six `lmgraph_from_*` wrappers (build/overlay/validate) and the triplicated
  `pre = f"{otype}__" if otype is not None else ""` → `_assemble`/`_assemble_single`/
  `_type_prefix`. Helper lives under `cospan/` (not `adapters/`) so B0 `from_petri` shares
  it without `import procposets.adapters` pulling pm4py — verified `from_petri` imports
  with none of pm4py/networkx/matplotlib in `sys.modules`. Gate: deterministic before/after
  LMGraph-key capture (process-tree + petri) byte-identical + in-process equivalence
  self-check for **all three** adapters incl BPMN (wrapper == old inline body on the same
  model object, pinning pm4py's per-process UUIDs) ALL-EQUAL. The `"{otype}__"` prefix is
  load-bearing for signature comparison and unchanged.

**Phase 2 fully landed except** the tree-block flatten skeleton
(`matrix._block_sequence`/`discrete._block_items`) — its windowing part is
`changes-values`, so it stays deferred (not a clean value-preserving dedup).

### Phase 4 — API evolution (api-breaking, aliased), landed 2026-07-21 (suite: 358 passed, 13 skipped)
Driven by a 10-agent read-only discovery across all 4 repos (consumer-breakage matrix +
per-item plan). All landed behind deprecation aliases where a consumer touches the old
name; end-gate re-run each consumer suite: PMN **201**, SPM **54**, unchanged. Two of the
plan's "aliases" were found IMPOSSIBLE as functional merges (different input types) and
landed as documentation instead — recorded honestly below.
- **Item 5 — `in_L` spelling** (`6ecd8a0`) — `likelihood.trace_p`/`group_logf` param `inL`
  → `in_L`, `oracle` local `inL_cache` → `in_L_cache`. All positional/internal (no kwarg
  callers anywhere), numpy arrays never rendered → no value risk, no alias.
- **Item 1 — e(P) (docs only)** (`6ecd8a0`) — the three spellings take three different input
  types (`Poset` / `(elements,rel)` / `SPTree`), so a functional `count_extensions` alias is
  impossible (immediate `TypeError`). Cross-referenced docstrings only; `count_linear_extensions`
  + `extension_count` kept first-class (PMN imports both — no DeprecationWarning).
- **Item 3 — series/parallel (docs only)** (`6ecd8a0`) — `then`/`par` build repeated-label
  `Poset`s, `series`/`parallel` build distinct-label `SPTree`s, and the package-root
  `series`/`parallel` are ALREADY the SPTree constructors, so a "canonical" alias is a footgun.
  Decision (owner): docstring notes only, no new symbols.
- **Item 6 — `CompositeDiagram.canonical_key` → `label_multiset_key`** (`3c03c17`) — dead in
  procposets + both cut-over consumers (every `canonical_key(...)` call is the free function
  `occurrence.canonical_key`). Rename resolves the 3-way ambiguity; `canon_key` /
  `occurrence.canonical_key` unchanged. sim/cpm keeps its own copy (reconnect at its port).
- **Item 8 — `report_vs_trivial` → `trivial_report`** (`8ceb3bf`) — module-level alias MANDATORY
  (PMN demos import the old name via `poset_mixture.diagnostics`); surfaced `trivial_report` in
  `_LAZY`/`__all__`; body byte-unchanged (format string byte-pinned in the PMN demo golden). The
  break only surfaces in the CONSUMER suite.
- **Item 2 — `sample_extension_tree`/`sample_extension_poset`** (`b60e305`) — the SPTree sampler
  and the ideal-DP engine both named `sample_extension`; renamed by input type. `sample_extension
  = sample_extension_tree` alias MANDATORY (PMN imports it and calls it directly). Byte-exact rng:
  the public name stays wired to the TREE sampler (the two consume rng differently).
- **Item 7 — `colour_map` + `viz/palette.py`** (`4272542`) — public `color_map` → `colour_map`
  (+ alias); every colour hex single-sourced into `palette.py` with the three list palettes kept
  SEPARATE (merging shifts modulo assignments). Gate: PYTHONHASHSEED=0 before/after over every
  palette path incl the byte-pinned dag `render_dag` DOT (vs live cpm) — byte-identical.
- **Item 4 — `noise_kernel`** (`de64d37`) — unified the spelling (Oracle/fit already used it);
  Atom field `noise` → `noise_kernel` with a deprecating `.noise` read-property (keeps PMN's
  `export_bridge.py` green) + `make_atom(noise=)` deprecated kwarg (TypeError if both). `describe()`
  keeps the byte-pinned `noise=` token. PMN stays green untouched (no `filterwarnings=error` on any
  `.noise` path; the demos error on warnings but never read it).
- **Item 9 — `StringDiagramStyle`** (`842bbd2`) — the 14 mutable viz style globals → two frozen
  dataclasses (`LayoutStyle`(4) + `DrawStyle`(10)) split along the layout/draw fault line, threaded
  through the term-DSL closures + layout/draw functions; `render`/`catalogue` gain an append-only
  `style=None` param that BRIDGES from the still-present globals (legacy override path byte-identical).
  **This unblocks the deferred Phase-6 `viz/string_diagram.py` file-split** (layout/draw knobs no
  longer cross-reference). Gate: before/after capture over every rendered artist (default output
  byte-identical) + a NEW override test (goldens only cover defaults). File-split itself left to Phase 6.

**Phase 4 deferred:** none of the 9 items deferred. Consumer-side alias REMOVAL (migrating PMN/SPM
off the old names, then dropping the aliases) is a per-consumer cut-over task, not done here — the
aliases stay until then.

### Finish-work — alias removal, override-globals removal, reproduce-check (2026-07-21)
Post-refactor cleanup items 2 & 4 (of the review's remaining-work list); item 3 (port sim/cpm)
deferred, flatten skipped.
- **Item 2 — Phase-4 alias removal** (procposets `27b6744`; PMN `82e7af7` on `arl/alias-cutover`).
  Dropped all 4 deprecation aliases: `rel_sp.sample_extension = sample_extension_tree` (+ its
  re-exports in rel.py / __init__.py / decompose.py __all__), `diagnostics.report_vs_trivial`,
  `viz/occn_vis.color_map`, and `Atom.noise` property + `make_atom(noise=)` kwarg (+ now-unused
  `import warnings`; dropped the `test_noise_kernel_deprecation_shim` regression). The
  `sample_extension` alias was load-bearing INTERNALLY (simulate.py, rel_classes.py,
  test_np_posets.py → `sample_extension_tree`); the unrelated `poset.sample_extension(P,rng)` and
  `PosetClass.sample_extension(self,…)` are NOT the alias and stay. PMN migrated: poset_mixture shim
  + test_posets (sample_extension→_tree), 6 demos (report_vs_trivial→trivial_report), export_bridge +
  tests (.noise→.noise_kernel, make_atom(noise=)→(noise_kernel=)), leaving the 3-arg method calls and
  the already-new fit/Oracle(noise_kernel=) alone. SPM used none. Gate: procposets 358, PMN 201, SPM 54.
- **Item 4 — string_diagram override globals removed** (`3e75156`). The 14 legacy module globals
  (`STRAIGHT_SPINE`…`TYPE_LANES`) + `_style_from_globals()` bridge deleted; `render(style=None)` →
  `DEFAULT_STYLE`. A read-only grep of the string-diagram-process-mining manuscript/tikz/figure code
  found ZERO overrides anywhere (its figures use the pre-knob cpm/vis.py), so the bridge was dead.
  `StringDiagramStyle` is the sole style API. Defaults byte-identical; suite 358. Item-4 flatten
  dedup: scoped and SKIPPED (byte-exact but net-zero value; windowing is changes-values).
- **Item 3 — port sim/cpm: DEFERRED past Phase 7** (owner decision). sim/cpm is the last independent
  byte-exact ORACLE for the cospan/occn/viz stack (the `*_matches_cpm` goldens); porting it retires
  that oracle, which Phase 7's speedup-gating needs. Do it as the final teardown, after Phase 7.
- **Reproduce-check — ALL GREEN.** procposets==cpm confirmed both directions (16 procposets golden
  cross-checks + sim's ed_chest_pain procposets-cross-check demos); cpm own suite **209 passed**;
  PMN 201; SPM 54; procposets 358/13. The refactor reproduces end-to-end across all repos.

**Next:** Phase 7 (efficiency stress map). The HOLD is lifted — everything reproduces.

### Phase 7 — efficiency stress map, LANDED 2026-07-21 (suite 362 passed / 13 skipped)
A 13-agent read-only stress-map recon (wf_af2f9c31-515) characterised each §7 seed against the current
code. **6 of 13 were already dead** (#1 Phase-0 cap; #8 warm-start already implemented; #10 walled/cached
/linear; #11 small-m + Phase-4 dedup; #5/#12 polynomial/O(1)-default). Of the 7 live, only #6/#13(D)/#9
were genuinely factorial. All speedups value-preserving, golden- or capture-gated:
- **#6 `traces.linear_extensions`** (`73dd2a7`) — guarded materialisation twin: `count_extensions` pre-check
  (golden pins == len) + `MAX_LINEAR_EXTENSIONS=1e6`; measured OOM cliff at N=11 (39.9M words), now refuses
  in 0.007s.
- **#13(D) `splice._count_linear_extensions`** (`df32f6c`) — ideal-DP swap for `nx.all_topological_sorts`
  count (O(2^width) vs width! sorts); byte-exact, cap_nodes=11 kept.
- **#5 `discrete._pair_relation`** (`913030e`) — hoist label->element inv out of the pair×variant loop.
- **#4 `simulate.one_timed`** (`1b570b5`) — hoist preds/succs + incremental frontier; rng-free, seeded
  output byte-identical (capture).
- **#7 `distance.smd_rows`/`_matrix_angle`** (`6498bed`) — sparse BC row-angle twin (row1∩row2∩keys ordered
  by position); byte-exact (dropped terms are sqrt(0*x)=0.0). NOT `_pairwise_rows`' changes-values path.
- **#3 `ideal_state_bound`** (`72c0343`, GUARD-ONLY) — succs built once (was O(m^3)/peel) + iterative
  longest-path (fixes RecursionError on a chain > recursion limit). Left `count_extensions`/`sample`'s
  ideal-DP recursion untouched: making it iterative turns their fast catchable RecursionError on a
  pathological tall poset into a multi-minute O(m^3) HANG — worse.
- **#2 `feasibility._enumerate` + #9 `initialiser.moment_seed`** — MEASURED, left BASELINE (a finding).
  #2 is bounded by `max_assignments=200k` (measured F=2..5 = 0.2..143ms, F>=6 caps out); realistic boxes
  tiny; a pruned DFS needs byte-exact lex-order for a ~143ms already-capped worst case. #9 is a one-time
  seed that (measured) doesn't escalate on realistic non-tied candidates (91ms); only a rare near-tie
  climbs (1.5s once). Neither justifies a delicate byte-exact rewrite on a consumer path.

Gate: procposets 362; PMN 201; SPM 54; cpm goldens green. Byte-exact gates: seeded/deterministic captures
(sim 630 lines, distance 1488, _extensions 8901) all identical before/after (sha256).
