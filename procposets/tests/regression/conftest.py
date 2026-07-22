"""Let the regression suite run on any install, down to a minimal
``pip install procposets`` with no extras: a test module whose module-level
imports need an optional extra is skipped at collection when that extra is
absent, so collection never errors on a missing dependency. The numpy-free
stdlib core (poset / cospan algebra / bridge / traces / grouping) always
collects.

Gating is by CAPABILITY (``importlib.util.find_spec``); the lists below are the
test modules that pull the ``[estimate]`` (numpy) or ``[graph]`` (networkx)
layer, derived from their actual top-level imports (plus the capped extractor,
which needs networkx at run time). Note the wheel excludes ``tests/``, so this
matters for a source checkout, not a wheel install.
"""
import importlib.util

# [estimate] (numpy): the NPMLE estimator tests + the comparison stack that
# builds numpy block matrices / stochastic-matrix distances.
_NUMPY_TESTS = [
    "test_npmle.py",
    "test_npmle_initialiser.py",
    "test_npmle_likelihood.py",
    "test_npmle_oracle.py",
    "test_comparison_context_depth.py",
    "test_comparison_discrete.py",
    "test_comparison_distance.py",
    "test_comparison_estimate.py",
    "test_comparison_loops.py",
    "test_comparison_pairwise.py",
]
# [graph] (networkx): the cospan B1 layer (occurrence / splice / trace-language
# and the capped DP extractor).
_NETWORKX_TESTS = [
    "test_cospan_occurrence.py",
    "test_cospan_splice.py",
    "test_cospan_trace_language.py",
    "test_cospan_extract_cap.py",
]

collect_ignore: list[str] = []
if importlib.util.find_spec("numpy") is None:
    collect_ignore += _NUMPY_TESTS
if importlib.util.find_spec("networkx") is None:
    collect_ignore += _NETWORKX_TESTS
