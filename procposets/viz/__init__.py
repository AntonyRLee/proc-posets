"""procposets.viz -- renderers ([viz] extra: matplotlib + graphviz).

Rendering, not calculation: kept out of the core so the numpy-only install
carries no matplotlib/graphviz.  Modules are imported by full path.

- string_diagram : cospan Signature -> string-diagram figure (matplotlib)
- compare_vis    : signature ComparisonReport -> comparison matrix (matplotlib)
- dag_render     : occurrence-net / diff -> Graphviz DOT (needs the `dot`
                   binary at runtime; `dag_render.has_dot()` guards it)
- occn_vis       : OCCN -> drawing (python graphviz lib; pm4py for OCPN)
- signature_diagram : model signature catalogue + morphism / SMD-heatmap figures (matplotlib)
"""
