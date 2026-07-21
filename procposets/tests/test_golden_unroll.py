"""Polish check: the unroll -> occn dependency inversion preserved behaviour.

`cospan/unroll.py` was split into the pure `cospan/unroll_core.py`
(unroll_generator/unroll_signature -- no OCCN dependency) and
`occn/unroll_occn.py` (the OCCN grounding, imports *down* into cospan).  This
verifies (a) cospan no longer imports occn, (b) unroll_generator reproduces
cpm's, (c) the OCCN grounding still imports and exposes the same surface.
"""

import importlib
import inspect
import pathlib
import sys

import pytest

SIM = pathlib.Path("/home/arl/Research/string-diagram-process-mining/sim")


def _cpm(module):
    if not (SIM / "cpm").is_dir():
        pytest.skip("sim/cpm not checked out")
    if str(SIM) not in sys.path:
        sys.path.insert(0, str(SIM))
    return importlib.import_module(module)


def test_cospan_does_not_import_occn():
    """The inversion's whole point: the cospan algebra no longer depends on
    the OCCN miner (dependency runs miner -> algebra only)."""
    import procposets.cospan as cospan
    pkg_dir = pathlib.Path(cospan.__file__).parent
    for py in pkg_dir.glob("*.py"):
        src = py.read_text()
        # allow docstring mentions, but no actual import of the occn package
        assert "from ..occn" not in src, f"{py.name} imports up into occn"
        assert "import procposets.occn" not in src, f"{py.name} imports occn"


def test_unroll_generator_matches_cpm():
    import procposets.cospan.unroll_core as ncore
    import procposets.cospan.constraints as ncon
    import procposets.cospan.signature as nsig
    ocore = _cpm("cpm.cospan.unroll")
    ocon = _cpm("cpm.cospan.constraints")
    osig = _cpm("cpm.cospan.signature")

    def constrained(sig_mod, con_mod):
        # a generator whose single input leg admits 1..3 objects
        p_in = sig_mod.Port("b", "order", "s")
        g = sig_mod.Generator("s", frozenset({p_in}), frozenset(),
                              con_mod.cset(*con_mod.interval(p_in, 1, 3)))
        return g

    gn = ncore.unroll_generator(constrained(nsig, ncon), order=5)
    go = ocore.unroll_generator(constrained(osig, ocon), order=5)
    # compare the weighted-leg multisets (Generator identity is by frozenset)
    key = lambda gens: sorted(sorted((p.src, p.typ, p.tgt, w) for p, w in gg.weights)
                              for gg in gens)
    assert key(gn) == key(go)


def test_occn_grounding_surface_exposed():
    import procposets.occn as occn
    for name in ("ground_occn", "ground_run", "gamma_boundary"):
        assert hasattr(occn, name), f"occn.{name} not exposed"
    import procposets.occn.unroll_occn as uo
    # it imports down into cospan.unroll_core
    assert "unroll_core" in inspect.getsource(uo)
