"""Let the regression suite run on any install, down to a minimal
``pip install procposets`` with no extras: a test module whose module-level
imports need an optional extra is skipped at collection when that extra is
absent, so collection never errors on a missing dependency. The numpy-free
stdlib core (poset / cospan algebra / bridge / traces) always collects.

Gating is by CAPABILITY (``importlib.util.find_spec``); the list below is the
test modules that pull the ``[numeric]`` (numpy) layer, derived from their
actual top-level imports. Note the wheel excludes ``tests/``, so this matters
for a source checkout, not a wheel install.
"""
import importlib.util

# [numeric] (numpy): the comparison stack, which builds numpy block matrices
# and stochastic-matrix distances.
_NUMPY_TESTS = [
    "test_comparison_discrete.py",
    "test_comparison_distance.py",
    "test_comparison_loops.py",
]

collect_ignore: list[str] = []
if importlib.util.find_spec("numpy") is None:
    collect_ignore += _NUMPY_TESTS
