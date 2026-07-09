"""Our OCCN (object-centric causal net) visualiser (graphviz) -- the
heuristic/causal-net rendering that is our alternative to the Java/React
reference (Liss et al.; driven separately by ``occn_dev/native_viz.py``).

``draw_occn`` takes a :class:`cpm.occn.OCCN` (from :func:`cpm.occn.mine_occn`)
and saves a PNG. Visual vocabulary (mirrors the reference's input-left /
output-right model):

* activity  -> rounded box;  ``START_<ot>`` -> triangle;  ``END_<ot>`` -> square (typed)
* marker    -> small node, circle if max cardinality 1, square if >1, filled with
               the object-type colour; labelled with its range if not ``1..1``
* a marker group -> its markers chained by a dotted line (one binding)
* shared key (object-distribution split) -> bold red link between same-key markers
* causal flow -> coloured (by type) edge from an output marker of ``a`` to the
                 matching input marker of ``b`` (START/END attach to the boundary)
"""

from __future__ import annotations

import contextlib
import math

import graphviz

from ..occn.markers import Marker, OCCN

_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def color_map(otypes) -> dict[str, str]:
    return {ot: _PALETTE[i % len(_PALETTE)] for i, ot in enumerate(sorted(otypes))}


@contextlib.contextmanager
def _themed_ocpn_colors(otypes):
    """Temporarily override pm4py's OCPN object-type colouring (which is a
    hash-of-the-name -> hex, hence ugly *and* hash-seed-unstable) with our
    :func:`color_map`, so a discovered OCPN renders in the same stable palette as
    :func:`draw_occn` (same colour per object type across both views)."""
    from pm4py.visualization.ocel.ocpn.variants import wo_decoration as w

    cmap = color_map(otypes)
    orig = w.ot_to_color
    w.ot_to_color = lambda ot: cmap.get(ot, orig(ot))
    try:
        yield
    finally:
        w.ot_to_color = orig


def save_vis_ocpn(ocpn, out_path: str) -> str:
    """``pm4py.save_vis_ocpn`` but with our palette (see
    :func:`_themed_ocpn_colors`). ``ocpn`` is a pm4py ``OCPetriNet``."""
    import pm4py

    otypes = getattr(ocpn, "object_types", None) or []
    with _themed_ocpn_colors(otypes):
        pm4py.save_vis_ocpn(ocpn, str(out_path))
    return str(out_path)


def _card_label(m: Marker) -> str:
    if m.cmin == 1 and m.cmax == 1:
        return ""
    hi = "*" if m.cmax == math.inf else str(m.cmax)
    return f"{m.cmin}..{hi}"


def draw_occn(occn: OCCN, out_path: str, title: str | None = None, fmt: str = "png") -> str:
    """Render ``occn`` to ``<out_path>.<fmt>`` (graphviz appends the extension);
    returns the written path."""
    colors = color_map(occn.ocdg.otypes)
    g = graphviz.Digraph("OCCN", format=fmt)
    g.attr(rankdir="LR", nodesep="0.25", ranksep="0.9", splines="spline")
    if title:
        g.attr(
            label=f"{title}\\l(○ card 1, □ card >1; coloured edge = typed flow; bold red link = shared-key object split)\\l",
            labelloc="t",
            fontsize="16",
        )

    for a in sorted(occn.ocdg.activities):
        g.node("A|" + a, a, shape="box", style="rounded,filled", fillcolor="#f0f0f0")
    for ot, nm in occn.ocdg.starts.items():
        g.node("A|" + nm, nm, shape="triangle", style="filled", fillcolor=colors[ot], fontsize="9")
    for ot, nm in occn.ocdg.ends.items():
        g.node("A|" + nm, nm, shape="square", style="filled", fillcolor=colors[ot], fontsize="9")

    nid: dict[tuple, str] = {}

    def marker(side: str, act: str, gi: int, m: Marker) -> str:
        key = (side, act, gi, m.activity, m.otype)
        if key in nid:
            return nid[key]
        n = f"{side[0]}{len(nid)}"
        nid[key] = n
        lab = _card_label(m)
        g.node(
            n, lab, shape=("circle" if m.cmax == 1 else "square"),
            width="0.16", height="0.16", fixedsize=("true" if not lab else "false"),
            style="filled", fillcolor=colors[m.otype], fontsize="8",
        )
        return n

    def group_chain(members: list[str]):
        for x, y in zip(members, members[1:]):
            g.edge(x, y, dir="none", style="dotted", color="#bbbbbb", constraint="false")

    def shared_key(group, members):
        # an object-distribution split is WITHIN a type: link only same-type
        # markers that share a key (cross-type common partition indices are an
        # encoding artefact, not a split; OCCN_DEV.md Finding F6).
        by_tk: dict[tuple[str, int], list[str]] = {}
        for m, n in zip(group, members):
            by_tk.setdefault((m.otype, m.key), []).append(n)
        for nodes in by_tk.values():
            if len(nodes) > 1:  # bold red link = object-XOR / shared-key split
                for x, y in zip(nodes, nodes[1:]):
                    g.edge(x, y, dir="none", style="bold", color="#d62728", constraint="false")

    in_index: dict[tuple[str, str, str], list[str]] = {}
    for a, groups in occn.input_groups.items():
        for gi, (group, _c) in enumerate(groups):
            gl = sorted(group, key=lambda m: (m.otype, m.activity))
            members = [marker("IN", a, gi, m) for m in gl]
            for m, n in zip(gl, members):
                in_index.setdefault((a, m.activity, m.otype), []).append(n)
                g.edge(n, "A|" + a, color="#999999", arrowsize="0.5")
            group_chain(members)
            shared_key(gl, members)

    for a, groups in occn.output_groups.items():
        for gi, (group, _c) in enumerate(groups):
            gl = sorted(group, key=lambda m: (m.otype, m.activity))
            members = [marker("OUT", a, gi, m) for m in gl]
            for m, n in zip(gl, members):
                g.edge("A|" + a, n, color="#999999", arrowsize="0.5")
                col = colors[m.otype]
                if m.activity.startswith("END_"):
                    g.edge(n, "A|" + m.activity, color=col, penwidth="1.6")
                else:
                    targets = in_index.get((m.activity, a, m.otype))
                    if targets:
                        for tn in targets:
                            g.edge(n, tn, color=col, penwidth="1.6")
                    else:
                        g.edge(n, "A|" + m.activity, color=col, penwidth="1.6")
            group_chain(members)
            shared_key(gl, members)

    for (a, src, ot), nodes in in_index.items():
        if src.startswith("START_"):
            for n in nodes:
                g.edge("A|" + src, n, color=colors[ot], penwidth="1.6")

    try:
        return g.render(out_path, cleanup=True)
    except graphviz.backend.execute.CalledProcessError:
        # work around a dot mincross assertion on dense label-free flat edges
        g.attr(splines="ortho")
        return g.render(out_path, cleanup=True)
