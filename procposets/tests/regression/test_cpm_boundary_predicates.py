"""Regression: the two boundary-label predicates have DISTINCT membership and
now carry self-documenting names.

They were once both spelled ``[_]is_boundary_label`` in sibling modules while
matching DIFFERENT sets -- ``signature_compare`` is gamma-inclusive
(``gamma1``/``gamma2``/``START_``/``END_``), ``signature_diff`` is ``START_``/
``END_`` only -- so a copy-paste or wrong import silently changed which labels
counted as boundary. This pins the membership and the intended difference.
"""
from procposets.cospan.signature_compare import is_boundary_label, is_gamma_or_marker
from procposets.cospan.signature_diff import _is_start_end_marker

_START_END = ["START_o", "END_o", "START_", "END_"]
_GAMMA = ["gamma1", "gamma2"]
_INTERIOR = ["a", "b", "gamma", "start", "MID_x", ""]


def test_gamma_inclusive_predicate_membership():
    for lab in _START_END + _GAMMA:
        assert is_gamma_or_marker(lab) is True
    for lab in _INTERIOR:
        assert is_gamma_or_marker(lab) is False


def test_start_end_only_predicate_membership():
    for lab in _START_END:
        assert _is_start_end_marker(lab) is True
    for lab in _GAMMA + _INTERIOR:  # gamma1/gamma2 are NOT start/end markers
        assert _is_start_end_marker(lab) is False


def test_the_two_predicates_differ_exactly_on_gamma():
    for lab in _GAMMA:
        assert is_gamma_or_marker(lab) and not _is_start_end_marker(lab)


def test_back_compat_alias_is_the_gamma_inclusive_predicate():
    assert is_boundary_label is is_gamma_or_marker
