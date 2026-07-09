"""Render occurrence-net DAGs to Graphviz DOT (and PNG/SVG via the ``dot``
binary). Design: ``CLASS_EXTRACTION.md`` §19.

Two products:

* :func:`render_dag` -- one fragment's boundary-rooted labelled DAG.
* :func:`render_overlay` -- the headline three-way comparison: the *union* of
  all models' occurrence nets, with every node/edge coloured by which models
  exhibit it (green = all, amber = a strict subset of >1, red = exactly one),
  so the structural diff reads like a process-map.

No Python Graphviz wrapper is needed (``graphviz``/``pydot`` are absent); DOT
text is emitted directly and the system ``dot`` binary does layout.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import Counter

from ..cospan.class_extraction import ExtractionResult, NamedMorphism, _fire, _to_counter
from ..cospan.dag_diff import DagDiffReport, closing_spine
from ..cospan.morphism_schema import _expand, _frames
from ..cospan.occurrence import IN, OUT, EventDag, anchor_types, canonical_key, history_keys, to_event_dag

# the all-models combo is always green; other combos draw from _PALETTE below
_ALL = "#2ca02c"      # green: every model agrees
_BOUNDARY = "#bbbbbb"  # grey: the IN/OUT roots


def _esc(s: str) -> str:
    return s.replace('"', '\\"')


def _typ_label(typs) -> str:
    """Edge label that collapses repeated object types to exponents: a wire carrying
    the same type k times reads ``order^5``, not ``order,order,order,order,order``.
    Distinct types stay comma-joined in first-appearance order (e.g. ``order^2,box``)."""
    counts = Counter(typs)
    return ",".join(t if counts[t] == 1 else f"{t}^{counts[t]}" for t in dict.fromkeys(typs))


def _collapse_bundles(events, edges):
    """Collapse interchangeable **sibling** events into one node, summing their parallel
    edges' multiplicities -- a faithful compaction under within-type fungibility (object
    tokens of one type carry no identity). Two events bundle iff they share a label, the
    same in-edges and the same out-edges (each as a multiset of ``(neighbour, type-tuple)``);
    such events are interchangeable in the DAG. So one ``r``'s ``k-1`` *not-investigated*
    orders draw as a single ``n`` reached by an ``order^(k-1)`` wire, not ``k-1`` parallel
    ``n`` boxes -- the ``r``-has-two-out-edges view. (Distinct ``r`` parents keep distinct
    ``n`` bundles: their in-edges differ, so they never merge.) Returns
    ``(events, edges, id_map)`` with ``id_map`` sending every original id to its kept rep."""
    label = dict(events)
    ins: dict = {}
    outs: dict = {}
    for u, v, typs in edges:
        outs.setdefault(u, []).append((v, tuple(typs)))
        ins.setdefault(v, []).append((u, tuple(typs)))

    def sig(n):
        key = lambda e: (str(e[0]), e[1])
        return (label.get(n), tuple(sorted(ins.get(n, []), key=key)),
                tuple(sorted(outs.get(n, []), key=key)))

    groups: dict = {}
    for n, _ in events:
        groups.setdefault(sig(n), []).append(n)
    id_map = {m: members[0] for members in groups.values() for m in members}

    events2 = [(n, lab) for n, lab in events if id_map[n] == n]
    agg: dict = {}
    for u, v, typs in edges:
        agg.setdefault((id_map[u], id_map[v]), Counter()).update(typs)
    edges2 = [(u, v, list(c.elements())) for (u, v), c in agg.items()]
    return events2, edges2, id_map


def render_dag(dag: EventDag, *, rankdir: str = "TB") -> str:
    """DOT for a single occurrence-net DAG. Event nodes are boxes labelled by
    activity; the :data:`IN`/:data:`OUT` boundary roots are small grey points;
    edges are labelled by object type."""
    g = dag.graph
    lines = [f'digraph "{_esc(dag.name)}" {{', f"  rankdir={rankdir};", "  node [fontname=Helvetica];"]
    for n, d in g.nodes(data=True):
        if d["label"] in (IN, OUT):
            lines.append(f'  "{n}" [shape=point, color="{_BOUNDARY}", width=0.1];')
        else:
            lines.append(f'  "{n}" [shape=box, style=rounded, label="{_esc(d["label"])}"];')
    for u, v, d in g.edges(data=True):
        lab = _esc(_typ_label(d.get("typs", ())))
        lines.append(f'  "{u}" -> "{v}" [label="{lab}", fontsize=9];')
    lines.append("}")
    return "\n".join(lines)


# Distinct colour per *exact* model combo (so "master+occn" and "occn+ocpn"
# are told apart, not both lumped as an ambiguous "subset"). The all-models
# combo is always green; the rest draw from this palette in a deterministic
# order, so the legend names exactly which models each colour means.
_PALETTE = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#d62728"]


def _combo_order_key(combo: frozenset, full: frozenset):
    return (combo != full, -len(combo), sorted(combo))


def _assign_combo_colours(membership_sets, full: frozenset) -> dict:
    """Map each distinct model-membership combo that actually appears to a
    stable colour (all-models = green, others from :data:`_PALETTE`)."""
    uniq = {frozenset(s) for s in membership_sets}
    out: dict = {}
    pi = 0
    for combo in sorted(uniq, key=lambda c: _combo_order_key(c, full)):
        if combo == full:
            out[combo] = _ALL
        else:
            out[combo] = _PALETTE[pi % len(_PALETTE)]
            pi += 1
    return out


def _combo_label(combo: frozenset, full: frozenset) -> str:
    name = "+".join(sorted(combo))
    return f"{name} (all)" if combo == full else name


def _combo_legend_lines(colours: dict, full: frozenset) -> list:
    lines = ['  subgraph cluster_legend {', '    label="model agreement"; fontsize=10; style=dotted;']
    for i, combo in enumerate(sorted(colours, key=lambda c: _combo_order_key(c, full))):
        col = colours[combo]
        lines.append(
            f'    lg{i} [shape=box,style="filled",fillcolor="{col}33",color="{col}",'
            f'label="{_esc(_combo_label(combo, full))}"];'
        )
    lines.append("  }")
    return lines


def render_overlay(report: DagDiffReport, *, loops: bool = False, rankdir: str = "TB") -> str:
    """DOT for the colour-coded *union* of all models' occurrence nets (the
    headline three-way visual). Identical structure across models is drawn
    once; each node/edge is coloured by how many models contain it.

    The union is taken over the canonical DAG classes (so isomorphic fragments
    from different models overlay onto the same drawn structure). Within a
    class, the representative DAG is drawn; its colour records the class's
    model-membership."""
    classes = report.loop_classes if loops else report.closing_classes
    full = frozenset(report.model_names)
    colours = _assign_combo_colours([cls.models for cls in classes], full)

    # node-key = (canonical-class-index, node-id); we draw every class's rep
    # but share the IN/OUT roots across the whole figure for a compact union.
    lines = ["digraph overlay {", f"  rankdir={rankdir};", "  node [fontname=Helvetica];", "  compound=true;"]
    lines.append(f'  "{IN}" [shape=point, color="{_BOUNDARY}", width=0.12];')
    lines.append(f'  "{OUT}" [shape=point, color="{_BOUNDARY}", width=0.12];')

    for ci, cls in enumerate(classes):
        colour = colours[frozenset(cls.models)]
        tag = _combo_label(frozenset(cls.models), full)
        g = cls.rep.graph

        def nid(n):
            return n if n in (IN, OUT) else f"c{ci}_{n}"

        for n, d in g.nodes(data=True):
            if d["label"] in (IN, OUT):
                continue
            lines.append(
                f'  "{nid(n)}" [shape=box, style="rounded,filled", '
                f'fillcolor="{colour}33", color="{colour}", '
                f'label="{_esc(d["label"])}", tooltip="{_esc(tag)}"];'
            )
        for u, v, d in g.edges(data=True):
            lab = _esc(_typ_label(d.get("typs", ())))
            lines.append(f'  "{nid(u)}" -> "{nid(v)}" [color="{colour}", label="{lab}", fontsize=9];')

    lines.extend(_combo_legend_lines(colours, full))
    lines.append("}")
    return "\n".join(lines)


def render_merged_overlay(report: DagDiffReport, *, loops: bool = False, rankdir: str = "TB") -> str:
    """The true process-map overlay (§20d): the **causal-prefix-merged** union
    of all models' occurrence nets. Shared structure is drawn *once* and
    coloured by which models exhibit it (green = all, amber = a strict subset
    >1, red = one) -- unlike :func:`render_overlay`, which draws each distinct
    structure separately. Merging is by :func:`history_keys`, so a node is
    shared only when its full causal cone (labels + typed wiring) agrees; this
    never conflates differently-wired events, it only collapses genuinely
    common prefixes."""
    classes = report.loop_classes if loops else report.closing_classes

    node_models: dict[str, set] = {}
    node_label: dict[str, str] = {}
    edge_models: dict[tuple, set] = {}
    for cls in classes:
        hk = history_keys(cls.rep)
        g = cls.rep.graph
        for n, d in g.nodes(data=True):
            k = hk[n]
            node_models.setdefault(k, set()).update(cls.models)
            node_label[k] = d["label"]
        for u, v, d in g.edges(data=True):
            ek = (hk[u], hk[v], tuple(d["typs"]))
            edge_models.setdefault(ek, set()).update(cls.models)

    full = frozenset(report.model_names)
    memberships = list(node_models.values()) + list(edge_models.values())
    colours = _assign_combo_colours(memberships, full)

    lines = ["digraph merged {", f"  rankdir={rankdir};", "  node [fontname=Helvetica];"]
    for k, label in node_label.items():
        if label in (IN, OUT):
            lines.append(f'  "{k}" [shape=point, color="{_BOUNDARY}", width=0.12];')
            continue
        combo = frozenset(node_models[k])
        colour = colours[combo]
        lines.append(
            f'  "{k}" [shape=box, style="rounded,filled", fillcolor="{colour}33", '
            f'color="{colour}", label="{_esc(label)}", tooltip="{_esc(_combo_label(combo, full))}"];'
        )
    for (ku, kv, typs), models in edge_models.items():
        colour = colours[frozenset(models)]
        lab = _esc(_typ_label(typs))
        lines.append(f'  "{ku}" -> "{kv}" [color="{colour}", label="{lab}", fontsize=9];')

    lines.extend(_combo_legend_lines(colours, full))
    lines.append("}")
    return "\n".join(lines)


def _splice_points(closing: NamedMorphism, loops, by_name) -> dict[int, list[str]]:
    """Where each loop can be spliced into ``closing`` (§19e, Q1): a loop with
    anchor frontier ``F`` attaches at every cut whose frontier ⊇ ``F``. Replay
    the closing's atomic firings (frame order, which is exactly
    ``to_event_dag``'s node order), and after each frame test every loop's
    anchor against the running frontier. Returns ``{last-node-of-cut: [loop
    names that anchor there]}``."""
    frames = _frames(_expand(closing.body, by_name))
    loop_anchors = [(nm.name, _to_counter(nm.boundary)) for nm in loops if nm.boundary]
    frontier: Counter = Counter()
    node_id = -1
    out: dict[int, list[str]] = {}
    for frame in frames:
        for g in frame:
            frontier = _fire(frontier, g)
            node_id += 1
        here = [name for name, anchor in loop_anchors
                if all(frontier[p] >= c for p, c in anchor.items())]
        if here:
            out.setdefault(node_id, []).extend(here)
    return out


def _spine_families(result: ExtractionResult) -> list[tuple[NamedMorphism, dict[int, set]]]:
    """Collapse a model's closings into ``M(m,σ)`` families by loop-free spine
    (§22h) and return one ``(baseline, splices)`` pair per family.

    A discovered model often lists the loop unrollings (``M(2,σ),...``) as
    *separate* closings built from *committed* generators: the unrolling's port
    replay exposes the loop splice-point (its committed ``examine_1`` re-enables
    the loop), while the family's loop-free baseline -- committed to an outcome
    -- exposes none. Drawing every closing therefore makes a discovered
    catalogue look unlike the master's even when they agree at the splice-aware
    family level. Collapsing onto one baseline per spine, and projecting the
    *whole family's* splice-points onto it, is what makes the two coincide:
    every family member shares the pre-splice prefix with its baseline, so a
    sibling's splice node (same id, same label) lands on the baseline's node.

    ``baseline`` is the family's shortest (loop-free) member; ``splices`` maps a
    baseline node id to the loop name(s) that anchor there."""
    loops = result.loops()
    families: dict[tuple, list[NamedMorphism]] = {}
    for nm in result.closing():
        families.setdefault(closing_spine(nm, loops, result.fragments), []).append(nm)

    def size(nm: NamedMorphism) -> int:
        return len(to_event_dag(nm, result.fragments, strip_prefixes=()).graph)

    out: list[tuple[NamedMorphism, dict[int, set]]] = []
    for members in families.values():
        base = min(members, key=lambda nm: (size(nm), nm.name))
        base_labels = {n: d["label"] for n, d in
                       to_event_dag(base, result.fragments, strip_prefixes=()).graph.nodes(data=True)}
        marks: dict[int, set] = {}
        for nm in members:
            m_labels = {n: d["label"] for n, d in
                        to_event_dag(nm, result.fragments, strip_prefixes=()).graph.nodes(data=True)}
            for node, names in _splice_points(nm, loops, result.fragments).items():
                # project onto the baseline only where the shared prefix agrees,
                # so a sibling's splice mark lands on the matching baseline node
                if base_labels.get(node) is not None and base_labels.get(node) == m_labels.get(node):
                    marks.setdefault(node, set()).update(names)
        out.append((base, marks))
    return sorted(out, key=lambda bm: bm[0].name)


def _loop_rep_names(loops: list[NamedMorphism], by_name: dict) -> dict[str, str]:
    """Map each loop name to the representative (shortest-name) loop of its
    distinct **structure** class -- the ``(anchor_types, canonical_key)``
    iso-class -- so a splice mark can list distinct loop *structures*, not every
    raw provenance duplicate. OCPN carries 79 raw loops over only 11 structures
    (§22d free-product + B4 self-bounces); without this, one splice site would
    pile a dozen names. On master/OCCN (a single loop ``L1``) it is the
    identity, so their catalogues are unchanged."""
    by_struct: dict[tuple, list[str]] = {}
    for lp in loops:
        key = (anchor_types(lp), canonical_key(to_event_dag(lp, by_name)))
        by_struct.setdefault(key, []).append(lp.name)
    rep: dict[str, str] = {}
    for names in by_struct.values():
        r = min(names, key=lambda s: (len(s), s))
        for n in names:
            rep[n] = r
    return rep


def _splice_node(term, site: int) -> int:
    """The pomset node id of the last atom before insertion index ``site`` -- the
    cut a loop anchors at. The concrete pomset's node ids run in frame order
    (one per atom, wrappers absorbed), matching the algebraic step skeleton."""
    return sum(len(s) if isinstance(s, tuple) else 1 for s in term.steps[:site]) - 1


def render_splice_catalogue(rep, *, title: str = "", collapse: bool = True) -> str:
    """One model's closing catalogue as a **view over its**
    :class:`~cpm.cospan.splice.SpliceRepresentation` (§27d): one cluster per
    ``M(m,σ)`` family (its concrete pomset) with a dashed loop **splice-arc** at
    each anchor cut. Drawing from the rep means the figure and the serialized
    splice artifact share one source and cannot drift.

    Each splice arc lists the **ℓ ids** of the loop structures that attach at
    that cut (``↺ ℓ1,ℓ3``); look each up in the loop gallery
    (:func:`render_loop_gallery`) to see its body. Family/loop ids (``σ``/``ℓ``)
    are the canonical ids of the rep, so they match the algebraic skeleton and
    the gallery exactly."""
    lines = [f'digraph "{_esc(title)}" {{', "  rankdir=TB; node [fontname=Helvetica]; compound=true;",
             '  labelloc=t;' + (f' label="{_esc(title)}";' if title else "")]
    # a splice arc names the loop's cut-invariant **cycle** (Cn), so the rotation phasings of
    # one loop read as the single loop they are; the phasing ℓ ids stay internal (generation).
    cycle_of = {lp.loop_id: (lp.cycle or lp.loop_id) for lp in rep.loops}
    for ci, fam in enumerate(rep.families):
        if collapse:
            events, edges, id_map = _collapse_bundles(fam.pomset.events, fam.pomset.edges)
        else:
            events, edges, id_map = fam.pomset.events, fam.pomset.edges, {}
        labels = dict(events)
        lines.append(f"  subgraph cluster_{ci} {{")
        lines.append(f'    label="{_esc(fam.spine_id)}"; style=dotted; fontsize=10;')

        def nid(n):
            return f"c{ci}_{n}"

        for n, lab in events:
            lines.append(f'    "{nid(n)}" [shape=box, style=rounded, label="{_esc(lab)}"];')
        for u, v, typs in edges:
            lines.append(f'    "{nid(u)}" -> "{nid(v)}" [label="{_esc(_typ_label(typs))}", fontsize=8];')
        for s in fam.splices:
            node = _splice_node(fam.term, s.site)
            node = id_map.get(node, node)
            if node not in labels:
                continue
            mark = "↺ " + ",".join(sorted({cycle_of.get(lid, lid) for lid in s.loop_ids}))
            lines.append(
                f'    "{nid(node)}" -> "{nid(node)}" '
                f'[style=dashed, color="#8c564b", fontcolor="#8c564b", '
                f'label="{_esc(mark)}", fontsize=9];'
            )
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def render_loop_gallery(rep, *, title: str = "") -> str:
    """The model's distinct loop **cycles**, one cluster per cycle (``Cn``) -- the same ids the
    closing catalogue's ``↺ Cn`` splice arcs use. A cycle's rotation phasings (cuts) are one
    loop categorically (a trace ``Tr(g)``), so they share a cluster; a representative phasing's
    pomset is drawn (events + typed edges), with the phasing count and anchor in the sub-label."""
    lines = [f'digraph "{_esc(title)}" {{', "  rankdir=TB; node [fontname=Helvetica]; compound=true;",
             '  labelloc=t;' + (f' label="{_esc(title)}";' if title else "")]
    if not rep.loops:
        lines.append('  none [shape=note, label="(no loops)"];')
    by_cycle: dict = {}
    for lp in rep.loops:
        by_cycle.setdefault(lp.cycle or lp.loop_id, []).append(lp)
    for ci, (cyc, phs) in enumerate(by_cycle.items()):
        lp = phs[0]  # representative phasing of the cycle
        anchor = ",".join(sorted(t for (_b, t), _n in lp.anchor))
        nph = f", {len(phs)} phasings" if len(phs) > 1 else ""
        lines.append(f"  subgraph cluster_{ci} {{")
        lines.append(f'    label="{_esc(cyc)}  @[{_esc(anchor)}]{_esc(nph)}"; style=dotted; fontsize=10;')

        def nid(n):
            return f"l{ci}_{n}"

        for n, lab in lp.pomset.events:
            lines.append(f'    "{nid(n)}" [shape=box, style=rounded, label="{_esc(lab)}"];')
        for u, v, typs in lp.pomset.edges:
            lines.append(f'    "{nid(u)}" -> "{nid(v)}" [label="{_esc(_typ_label(typs))}", fontsize=8];')
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def render_catalogue_dags(result: ExtractionResult, *, title: str = "") -> str:
    """Splice catalogue for a raw :class:`ExtractionResult` -- a thin wrapper
    that builds the :class:`~cpm.cospan.splice.SpliceRepresentation` and renders
    it via :func:`render_splice_catalogue` (the single source of truth, §27d).
    Retained for callers/tests that start from an ``ExtractionResult``."""
    from ..cospan.splice import SpliceRepresentation

    return render_splice_catalogue(
        SpliceRepresentation.from_extraction_result(result, name=title or "model"), title=title
    )


def has_dot() -> bool:
    return shutil.which("dot") is not None


def write_dot(dot: str, out_path: str, *, fmt: str = "svg") -> str:
    """Write ``dot`` to ``<out_path>.dot`` and, if the ``dot`` binary is
    available, render ``<out_path>.<fmt>`` beside it. Returns the rendered
    path (or the ``.dot`` path if no binary). Raises nothing on a missing
    binary -- the DOT source is always written so it can be rendered later."""
    dot_path = out_path if out_path.endswith(".dot") else out_path + ".dot"
    with open(dot_path, "w") as f:
        f.write(dot)
    if not has_dot():
        return dot_path
    rendered = out_path.rsplit(".dot", 1)[0] + f".{fmt}"
    subprocess.run(["dot", f"-T{fmt}", dot_path, "-o", rendered], check=True)
    return rendered
