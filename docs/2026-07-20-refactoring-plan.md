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
`changes-values`).
