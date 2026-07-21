"""Phase-6 (viz layer) checks.  Rendering is hard to byte-compare, so these
are import + headless-render smoke tests (the real risk is a botched import
rewrite), plus a value check where a viz helper returns data not a figure.

All under the [viz] marker; skipped without matplotlib/graphviz.
"""

import importlib
import os
import pathlib
import sys

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

SIM = pathlib.Path("/home/arl/Research/string-diagram-process-mining/sim")

pytestmark = pytest.mark.viz


def _cpm(module):
    if not (SIM / "cpm").is_dir():
        pytest.skip("sim/cpm not checked out")
    if str(SIM) not in sys.path:
        sys.path.insert(0, str(SIM))
    return importlib.import_module(module)


def test_viz_modules_import():
    pytest.importorskip("matplotlib")
    for m in ("string_diagram", "compare_vis", "dag_render", "spm_viz"):
        importlib.import_module(f"procposets.viz.{m}")
    # occn_vis needs the graphviz python lib
    pytest.importorskip("graphviz")
    importlib.import_module("procposets.viz.occn_vis")


def _signature():
    import procposets.cospan.engine as eng
    import procposets.cospan.lmgraph as lm
    g = lm.LMGraph()
    for a in "abc":
        g.add_activity(a)
    m = g.add_mediator("m", lm.Kind.XOR)
    g.add_edge("a", m)
    g.add_edge("m", "b")
    g.add_edge("m", "c")
    g.validate()
    return eng.extract_signature(g)


def test_string_diagram_renders_headless():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import procposets.viz.string_diagram as sd
    sig = _signature()
    diagram = sd.D(sig, "a", 0)          # 'a' spans 2 generators; pick the first
    fig = sd.render(diagram, sig, title="smoke")
    assert isinstance(fig, Figure)


def test_spm_viz_extract_generators_pinned():
    # Retired golden seam: this began as a cross-check against SPME's spm.viz /
    # spm.poset.  Both became pure shims that re-export procposets (spm/viz.py does
    # `from procposets.viz.spm_viz import *`; spm/__init__ aliases
    # sys.modules["spm.poset"] = procposets.poset), so the old
    # `== oviz.extract_generators(model(opos))` side re-entered the SAME procposets
    # function object on an equal input -- a tautology that could no longer catch a
    # regression, hung off a hardcoded external path.  It survives as a
    # self-contained regression on the pinned generator-cospan signature (value
    # computed from procposets.viz.spm_viz.extract_generators).
    pytest.importorskip("matplotlib")
    import procposets.viz.spm_viz as nviz
    import procposets as pp

    def model(mp):
        return [(mp.then(mp.leaf("a"), mp.par(mp.leaf("b"), mp.leaf("c")), mp.leaf("d")), 1.0)]

    # a -> (b || c) -> d: each activity's (label, in-ports, out-ports); a source legs
    # in from gamma_1, a sink legs out to gamma_2 (the SPME spm.viz cross-check agreed
    # on this signature).
    assert nviz.extract_generators(model(pp)) == [
        ("a", (("g1", "a"),), (("a", "b"), ("a", "c"))),
        ("b", (("a", "b"),), (("b", "d"),)),
        ("c", (("a", "c"),), (("c", "d"),)),
        ("d", (("b", "d"), ("c", "d")), (("d", "g2"),)),
    ]


def test_dag_render_dot_matches_cpm():
    pytest.importorskip("networkx")
    import procposets.cospan.extract_dp as ndp
    import procposets.viz.dag_render as ndr
    odp = _cpm("cpm.cospan.extract_dp")
    odr = _cpm("cpm.cospan.dag_render")
    import procposets.cospan.occurrence as nocc
    oocc = _cpm("cpm.cospan.occurrence")
    sig = _signature()
    res = ndp.extract_classes(sig)
    frags = res.closing()               # list[NamedMorphism]
    if not frags:
        pytest.skip("no closing fragment to render")
    dag_new = nocc.to_event_dag(frags[0], res.fragments)
    ores = odp.extract_classes(sig)     # signatures are shared datatypes
    ofrags = ores.closing()
    dag_old = oocc.to_event_dag(ofrags[0], ores.fragments)
    assert ndr.render_dag(dag_new) == odr.render_dag(dag_old)
