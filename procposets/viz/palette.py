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

# --- the single default categorical palette for string-diagram outputs ------
# The data-viz reference palette (light mode): 8 CVD-safe hues in a fixed order
# chosen to maximise the minimum adjacent colour-blind separation (worst adjacent
# ΔE 24.2, well above the >=12 target; pre-validated by the dataviz skill's
# validate_palette.js). EVERY string-diagram type colour comes from here now --
# the named map below pins known types to fixed slots, and any other type falls
# through to this list in sorted order (_colour_map). Fixed order IS the CVD-safety
# mechanism: do not reorder. (aqua/yellow/magenta sit below 3:1 line-contrast on
# white; the legend + numbered-port labels carry identity, so colour is never the
# sole channel.)
CATEGORICAL_DEFAULT = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                       "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
#   slot:                0 blue    1 teal    2 amber   3 green
#                        4 violet  5 red     6 magenta 7 orange

# Known types pinned to fixed CATEGORICAL_DEFAULT slots -- stable per entity across
# figures (colour follows the type, not its rank). eICU types keep their
# blue/green/orange/red intent; the loop-family ports move onto the unified palette.
STRING_DIAGRAM_TYPE_COLOURS: "dict[str | None, str]" = {
    "pat": CATEGORICAL_DEFAULT[0],    # patient -- blue
    "lab": CATEGORICAL_DEFAULT[3],    # lab     -- green
    "img": CATEGORICAL_DEFAULT[7],    # imaging -- orange
    "bed": CATEGORICAL_DEFAULT[5],    # bed     -- red
    "alpha": CATEGORICAL_DEFAULT[4],  # loop-family alpha -- violet
    "beta": CATEGORICAL_DEFAULT[1],   # loop-family beta  -- teal
    "gamma": CATEGORICAL_DEFAULT[2],  # loop-family gamma -- amber
    None: "#888888",                  # untyped -- grey (neutral, not a category)
}
STRING_DIAGRAM_FALLBACK = CATEGORICAL_DEFAULT  # back-compat alias (its consumers)
