"""Generate a model's **trace language** (valid runs γ1→γ2) from its
:class:`~procposets.cospan.splice.SpliceRepresentation`, up to a finite loop cut-off.

Design: ``CLASS_EXTRACTION.md`` §27c. The splice representation is the finite
generating grammar: a run is a family baseline with its anchored loops spliced
in (any multiplicity). This module realises that grammar to a bounded depth.

Two honest scope limits, by design (§27c, user-accepted):

* **Bounded, not complete.** Enumerating all linear extensions of a pomset is
  #P-hard; we bound loop unrollings by ``max_loops`` and (optionally) cap the
  number of traces. No completeness is claimed -- a heavier exact pass is left
  for later.
* **Algebraic (layered) vs causal.** For *series-parallel* families
  (``Family.sp_exact``) the cheap algebraic linearization of the step-skeleton
  equals the causal pomset, so it is exact. For genuinely non-SP families the
  baseline (m=0) is linearised exactly from the concrete pomset; m≥1 unrollings
  use the layered reading, which can over-order -- those families are reported
  in :attr:`TraceLanguage.approx_families`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import networkx as nx

from .splice import Family, Pomset, SpliceRepresentation


@dataclass
class TraceLanguage:
    """Traces per family (each a tuple of activity labels), with the loop bound
    used and the families whose m≥1 traces are a layered over-approximation."""

    traces: dict
    max_loops: int
    approx_families: tuple

    def all_traces(self) -> set:
        out: set = set()
        for ts in self.traces.values():
            out |= ts
        return out


def model_traces(
    rep: SpliceRepresentation, *, max_loops: int = 1, max_traces_per_family: int | None = 20000
) -> TraceLanguage:
    """Trace language of ``rep`` allowing up to ``max_loops`` total loop
    traversals per run. Returns a :class:`TraceLanguage`."""
    loops_by_id = {lp.loop_id: lp for lp in rep.loops}
    traces: dict[str, set] = {}
    approx: list[str] = []
    for fam in rep.families:
        ts, is_approx = _family_traces(fam, loops_by_id, max_loops, max_traces_per_family)
        traces[fam.spine_id] = ts
        if is_approx:
            approx.append(fam.spine_id)
    return TraceLanguage(traces=traces, max_loops=max_loops, approx_families=tuple(approx))


# --- per-family -------------------------------------------------------------


def _family_traces(fam: Family, loops_by_id, max_loops, cap) -> tuple[set, bool]:
    out: set = set()
    approx = False
    for seq, n_loops in _expanded_sequences(fam, loops_by_id, max_loops):
        if n_loops == 0 and not fam.sp_exact:
            # exact baseline from the concrete pomset (the layered reading would
            # over-order a non-SP family)
            out |= _pomset_linearizations(fam.pomset, cap)
        else:
            if n_loops > 0 and not fam.sp_exact:
                approx = True
            out |= _layered_linearizations(seq)
        if cap is not None and len(out) > cap:
            return out, approx
    return out, approx


def _expanded_sequences(fam: Family, loops_by_id, max_loops):
    """Yield ``(step_sequence, total_loop_count)`` for every way of splicing the
    family's anchored loops at their sites with total count ≤ ``max_loops``."""
    spine = list(fam.term.steps)
    sites = list(fam.splices)
    if not sites:
        yield tuple(spine), 0
        return
    per_site = [_site_words(s.loop_ids, max_loops) for s in sites]
    for combo in product(*per_site):
        total = sum(len(w) for w in combo)
        if total > max_loops:
            continue
        seq = list(spine)
        # splice high sites first so earlier insertion indices stay valid
        for site, word in sorted(zip(sites, combo), key=lambda sw: -sw[0].site):
            insert: list = []
            for lid in word:
                insert.extend(loops_by_id[lid].term.steps)
            seq[site.site : site.site] = insert
        yield tuple(seq), total


def _site_words(loop_ids: tuple, max_len: int) -> list:
    """All words over a site's loop ids up to length ``max_len`` (the free
    monoid the site can host, bounded)."""
    words: list = []
    for length in range(max_len + 1):
        words.extend(product(loop_ids, repeat=length))
    return words


# --- linearization ----------------------------------------------------------


def _layered_linearizations(steps: tuple) -> set:
    """Linear extensions of a layered step sequence: ``;``-ordered across steps,
    all permutations within a ``@``-concurrent tensor group."""
    per_step = [list(permutations(s)) if isinstance(s, tuple) else [(s,)] for s in steps]
    out: set = set()
    for combo in product(*per_step):
        out.add(tuple(lab for group in combo for lab in group))
    return out


def _pomset_linearizations(pomset: Pomset, cap: int | None) -> set:
    """Exact linear extensions of a concrete pomset (topological sorts of the
    occurrence-net DAG), labels projected from node ids."""
    g = nx.DiGraph()
    label = dict(pomset.events)
    g.add_nodes_from(label)
    g.add_edges_from((u, v) for (u, v, _typs) in pomset.edges)
    out: set = set()
    for order in nx.all_topological_sorts(g):
        out.add(tuple(label[n] for n in order))
        if cap is not None and len(out) > cap:
            break
    return out
