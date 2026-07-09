"""The dependency-free-core property: importing procposets and its stdlib
modules must not import numpy; only touching the NPMLE estimator (the
[estimate] layer) pulls it.  Run in a fresh subprocess so the import side
effect is measured cleanly (numpy is installed in the dev venv -- we assert
whether it was *imported*, not whether it is available)."""

import subprocess
import sys

import pytest


def _numpy_imported(code: str) -> bool:
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip() == "yes"


def test_stdlib_core_import_is_numpy_free():
    code = (
        "import procposets, procposets.poset, procposets.rel, "
        "procposets.moddecomp, procposets.traces, procposets.grouping, "
        "procposets.distance, procposets.matrix, procposets.estimate, "
        "procposets.loops, procposets.bridge; "
        "import sys; print('yes' if 'numpy' in sys.modules else 'no')"
    )
    assert not _numpy_imported(code), "the stdlib core must not import numpy"


def test_pure_cospan_algebra_is_numpy_free():
    code = (
        "import procposets.cospan.engine, procposets.cospan.signature, "
        "procposets.cospan.compose, procposets.cospan.signature_compare; "
        "import sys; print('yes' if 'numpy' in sys.modules else 'no')"
    )
    assert not _numpy_imported(code), "the B0 cospan algebra must not import numpy"


def test_touching_the_estimator_pulls_numpy():
    # attribute access on a lazy name resolves it (PEP 562) -> imports numpy
    code = (
        "import procposets; procposets.fit; procposets.GroupedLog; "
        "import sys; print('yes' if 'numpy' in sys.modules else 'no')"
    )
    assert _numpy_imported(code), "reaching the NPMLE estimator must pull numpy"


def test_lazy_names_are_still_reachable_and_cached():
    import procposets
    fit1 = procposets.fit
    fit2 = procposets.fit                    # second access hits the cache
    assert fit1 is fit2
    assert procposets.fit.__module__ == "procposets.npmle"
    # a genuinely unknown attribute still errors
    with pytest.raises(AttributeError):
        procposets.definitely_not_a_name
