"""P0 regression: ``extract_dp.extract_classes`` enforces ``max_pomsets_per_state``.

The documented per-frontier-state cap on loop-free closing iso-classes was
declared in the signature/docstring but never implemented, so
``ExtractionResult.truncated`` was *always* ``False`` and a genuinely
over-generating net (a free-product execution space) had no safety valve --
it ran to ``max_frontiers`` or OOM instead of truncating. This pins the guard:
a low cap truncates and flags it, while the default cap leaves a well-formed
model's full catalogue untouched (byte-identical to the pre-fix behaviour).

Needs the [graph] extra (networkx, via ``occurrence``); the conftest ignores
this file on a numpy-only install.
"""
from procposets.cospan import extract_dp
from procposets.cospan.engine import extract_signature

from .test_cospan_engine_running_example import build_running_example

_FULL_CLOSINGS = 12  # the running example's full loop-free closing catalogue


def _running_signature():
    return extract_signature(build_running_example())


def test_default_cap_leaves_well_formed_model_untruncated():
    res = extract_dp.extract_classes(_running_signature())
    assert res.truncated is False
    assert len(res.closing()) == _FULL_CLOSINGS


def test_low_cap_truncates_and_sets_the_flag():
    sig = _running_signature()
    for cap in (1, 2, 3):
        res = extract_dp.extract_classes(sig, max_pomsets_per_state=cap)
        assert res.truncated is True
        assert len(res.closing()) <= cap < _FULL_CLOSINGS


def test_high_cap_matches_the_uncapped_catalogue():
    sig = _running_signature()
    high = extract_dp.extract_classes(sig, max_pomsets_per_state=10_000)
    assert high.truncated is False
    assert len(high.closing()) == _FULL_CLOSINGS
