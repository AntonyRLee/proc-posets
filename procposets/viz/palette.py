"""Centralized colour palettes for the viz layer -- one source for the hexes.

The lists are kept SEPARATE per renderer on purpose: ``OCCN_PALETTE`` (7 items,
includes green), ``DAG_COMBO_PALETTE`` (8 items, EXCLUDES green -- green is reserved
for the all-models combo), and ``STRING_DIAGRAM_FALLBACK`` (a third list) differ, so
merging them would shift every modulo colour assignment and break the byte-exact dag
DOT golden. Hexes and list order are verbatim copies of the per-module originals;
changing any value changes rendered output. (Note the two case conventions are
intentional and load-bearing for the byte-exact DOT: dag_render emits lowercase hex,
string_diagram/signature_diagram use the uppercase "Red, Black & Blue" variants.)
"""
from __future__ import annotations

# --- named hexes shared across >1 renderer (deduplicated here) --------------
ALIZARIN = "#D2292D"          # red variant   (string_diagram, signature_diagram)
COTTON_BLUE = "#1761B0"       # blue variant  (string_diagram, signature_diagram)
OLD_BLACK = "#282828"         # black variant (string_diagram, signature_diagram)
ALL_MODELS_GREEN = "#2ca02c"  # every model agrees (dag_render all-models combo)
BOUNDARY_GREY = "#bbbbbb"     # IN/OUT / gamma boundary roots (dag_render, signature_diagram)
SPLICE_BROWN = "#8c564b"      # dag_render dashed loop/splice edge

# --- occn_vis object-type palette (7 items, includes green) -----------------
OCCN_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

# --- dag_render model-combo palette (8 items, EXCLUDES green) ----------------
DAG_COMBO_PALETTE = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#d62728"]

# --- string_diagram type -> colour map + overflow fallback list -------------
# Red/black/blue variants from the "Red, Black & Blue" palette
# (color-hex.com/color-palette/26562): one variant of each for the loop-family ports.
STRING_DIAGRAM_TYPE_COLOURS: "dict[str | None, str]" = {
    "pat": "#1f77b4",     # patient  -- blue
    "lab": "#2ca02c",     # lab      -- green
    "img": "#ff7f0e",     # imaging  -- orange
    "bed": "#d62728",     # bed      -- red
    "alpha": ALIZARIN,    # loop-family alpha -- red
    "beta": COTTON_BLUE,  # loop-family beta  -- blue
    "gamma": OLD_BLACK,   # loop-family gamma -- black
    None: "#888888",      # untyped  -- grey
}
STRING_DIAGRAM_FALLBACK = ["#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
