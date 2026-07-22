"""The dependency-free-core property: importing procposets and its stdlib
modules (and the pure B0 cospan algebra) must pull *no* third-party package.
numpy sits behind the [estimate] extra and loads lazily on first access to an
NPMLE estimator name (PEP 562); networkx / matplotlib / pm4py sit behind the
[graph] / [viz] / [pm4py] extras and load only when their layer module is
touched.

Each check runs in a *fresh* subprocess and probes ``sys.modules`` there, so
the import side effect is measured cleanly and is never contaminated by a
module pytest itself already imported (numpy / networkx / matplotlib / pm4py
are all installed in the dev venv -- we assert whether a package was *imported*
by a given import scenario, not whether it is available).

Two directions keep the guards honest:

- negatives ("the core does not pull X") assert *absence* from ``sys.modules``,
  so they hold on a minimal install where X is not even installed;
- positives ("touching the X layer pulls X") certify each negative is
  non-vacuous -- X really is reachable from the package -- and need the extra,
  so they ``pytest.importorskip(X)`` and skip cleanly on a minimal install.
"""

import subprocess
import sys

import pytest

# The eager, dependency-free import sets, kept as single-source-of-truth
# constants so the numpy negatives and the graph/viz/pm4py negatives probe the
# *same* module lists.
_STDLIB_CORE = (
    "import procposets, procposets.poset, procposets.rel, "
    "procposets.moddecomp, procposets.traces, procposets.grouping, "
    "procposets.distance, procposets.matrix, procposets.estimate, "
    "procposets.loops, procposets.bridge"
)
_COSPAN_ALGEBRA = (
    "import procposets.cospan.engine, procposets.cospan.signature, "
    "procposets.cospan.compose, procposets.cospan.signature_compare"
)


def _fresh_import_pulls(setup: str, dep: str) -> bool:
    """Run ``setup`` (imports and/or attribute accesses) in a *fresh*
    interpreter -- so the measurement is never polluted by a module pytest
    itself already imported -- and report whether ``dep`` landed in
    ``sys.modules`` as an import side effect.  The probe prints the
    ``yes``/``no`` sentinel; we compare it exactly as the original helper did.
    """
    code = (
        f"{setup}; import sys; "
        f"print('yes' if {dep!r} in sys.modules else 'no')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip() == "yes"


# --- numpy / [estimate]: the dependency-free-core property (unchanged) ------


def test_stdlib_core_import_is_numpy_free():
    assert not _fresh_import_pulls(_STDLIB_CORE, "numpy"), (
        "the stdlib core must not import numpy"
    )


def test_pure_cospan_algebra_is_numpy_free():
    assert not _fresh_import_pulls(_COSPAN_ALGEBRA, "numpy"), (
        "the B0 cospan algebra must not import numpy"
    )


def test_touching_the_estimator_pulls_numpy():
    # attribute access on a lazy name resolves it (PEP 562) -> imports numpy
    assert _fresh_import_pulls(
        "import procposets; procposets.fit; procposets.GroupedLog", "numpy"
    ), "reaching the NPMLE estimator must pull numpy"


def test_lazy_names_are_still_reachable_and_cached():
    import procposets
    fit1 = procposets.fit
    fit2 = procposets.fit                    # second access hits the cache
    assert fit1 is fit2
    assert procposets.fit.__module__ == "procposets.npmle"
    # a genuinely unknown attribute still errors
    with pytest.raises(AttributeError):
        procposets.definitely_not_a_name


# --- networkx / matplotlib / pm4py: the same contract for the other extras --
#
# Negatives assert *absence* from a fresh sys.modules, so they are correct on a
# minimal install where the extra is not even present -- they never import the
# extra.  Each is non-vacuous because the paired positive touch-test below
# proves the layer really does pull the dep (direction B): so "the core does
# not pull X" is a genuine separation, not a dep that could never appear.


def test_stdlib_core_is_graph_and_viz_free():
    # Same eager-core module list as test_stdlib_core_import_is_numpy_free.
    # The numpy-only stdlib core installs at py3.10; networkx/matplotlib sit
    # behind the [graph]/[viz] extras and must NOT be pulled by importing the
    # core.  Witnessed by test_touching_the_graph_layer_pulls_networkx
    # (equivalence.py: top-level `import networkx as nx`) and
    # test_touching_the_viz_layer_pulls_matplotlib (viz/string_diagram.py:
    # top-level `import matplotlib...`).
    for dep in ("networkx", "matplotlib"):
        assert not _fresh_import_pulls(_STDLIB_CORE, dep), (
            f"the stdlib core must not import {dep}"
        )


def test_stdlib_core_is_pm4py_free():
    # pm4py sits behind the [pm4py] extra.  Witnessed by
    # test_touching_the_pm4py_layer_pulls_pm4py (adapters/from_bpmn.py:
    # top-level `from pm4py.objects.bpmn.obj import BPMN`).
    assert not _fresh_import_pulls(_STDLIB_CORE, "pm4py"), (
        "the stdlib core must not import pm4py"
    )


def test_pure_cospan_algebra_is_graph_free():
    # Same B0 module list as test_pure_cospan_algebra_is_numpy_free.  This is
    # the least-obvious, most drift-prone guard in the batch: the algebra core
    # (engine/signature/compose/signature_compare) stays networkx-free *even
    # though* its direct siblings cospan.occurrence and cospan.trace_language
    # each carry a top-level `import networkx as nx`.  Witnessed by the
    # cospan.occurrence assertion in test_touching_the_graph_layer_pulls_networkx.
    assert not _fresh_import_pulls(_COSPAN_ALGEBRA, "networkx"), (
        "the B0 cospan algebra must not import networkx"
    )


def test_touching_the_graph_layer_pulls_networkx():
    # Non-vacuity witness for both graph-free negatives above.
    # `cospan.equivalence` certifies the stdlib-core guard; `cospan.occurrence`
    # is a direct sibling of the B0 algebra, so it pins the (non-obvious)
    # cospan-algebra guard.  Needs the [graph] extra, so skip cleanly on a
    # minimal install; the measurement itself still runs in a fresh subprocess.
    pytest.importorskip("networkx")
    assert _fresh_import_pulls("import procposets.cospan.equivalence", "networkx"), (
        "the [graph] layer (cospan.equivalence) must pull networkx"
    )
    assert _fresh_import_pulls(
        "import procposets.cospan.occurrence", "networkx"
    ), "a B0-algebra sibling (cospan.occurrence) must pull networkx"


def test_touching_the_viz_layer_pulls_matplotlib():
    # Non-vacuity witness for the viz-free half of the stdlib-core negative.
    # Needs the [viz] extra, so skip cleanly on a minimal install.
    pytest.importorskip("matplotlib")
    assert _fresh_import_pulls(
        "import procposets.viz.string_diagram", "matplotlib"
    ), "the [viz] layer (string_diagram) must pull matplotlib"


def test_string_diagram_layout_half_is_matplotlib_free():
    # The point of the string_diagram file-split: the geometry half (viz/_layout
    # -- term DSL + Layout/PlacedBox/Wire + _finish/_layout_composite) stays
    # backend-free so a TikZ (or other) backend can reuse it without dragging in
    # matplotlib.  The drawing half (string_diagram) still pulls matplotlib -- see
    # the paired positive test above, which is this negative's non-vacuity witness.
    pytest.importorskip("matplotlib")  # so the paired positive is meaningful
    assert not _fresh_import_pulls(
        "import procposets.viz._layout", "matplotlib"
    ), "the string-diagram layout half (viz._layout) must stay matplotlib-free"


def test_touching_the_pm4py_layer_pulls_pm4py():
    # Non-vacuity witness for the pm4py-free negative.  Importing the adapter
    # submodule runs adapters/__init__.py, which eagerly imports from_bpmn
    # (top-level `from pm4py.objects.bpmn.obj import BPMN`), landing 'pm4py' in
    # sys.modules.  Needs the [pm4py] extra, so skip cleanly on a minimal
    # install.
    pytest.importorskip("pm4py")
    assert _fresh_import_pulls(
        "import procposets.adapters.from_bpmn", "pm4py"
    ), "the [pm4py] layer (adapters.from_bpmn) must pull pm4py"
