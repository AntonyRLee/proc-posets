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

SIM = pathlib.Path("/home/arl/Research/-DIAGRAM-String-diagrams-for-process-mining-v2/sim")
SPME = pathlib.Path("/home/arl/Research/stochastic_process_mining/experiments")

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


def test_spm_viz_extract_generators_matches_spme():
    pytest.importorskip("matplotlib")
    import procposets.viz.spm_viz as nviz
    import procposets as pp
    if not (SPME / "spm").is_dir():
        pytest.skip("SPME not checked out")
    if str(SPME) not in sys.path:
        sys.path.insert(0, str(SPME))
    oviz = importlib.import_module("spm.viz")
    opos = importlib.import_module("spm.poset")

    def model(mp):
        return [(mp.then(mp.leaf("a"), mp.par(mp.leaf("b"), mp.leaf("c")), mp.leaf("d")), 1.0)]

    assert nviz.extract_generators(model(pp)) == oviz.extract_generators(model(opos))


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
