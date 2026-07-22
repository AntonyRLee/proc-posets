"""Release-gate guard (WS 4.3): the *shipping tree* must not leak any consumer
repo, developer-machine path, or consumer-doc filename.

``procposets`` releases on its own track and is **not** any paper's artifact, so
none of the consumer coupling (cpm/sim, poset-mixture, the SPM/ED-pipeline docs)
may appear in the code that actually ships in the wheel. The wheel packages the
``procposets`` package source and excludes ``procposets/tests`` + ``*.md`` (see
``[tool.hatch.build.targets.wheel]``), so we scan exactly that surface: every
non-test file under the package directory.

This is a fast, dependency-free grep; it runs in the base (numpy-less) suite.
"""

import pathlib

# The package dir is the parent of this tests/ directory.
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS_DIR = pathlib.Path(__file__).resolve().parent

# Substrings that betray consumer/developer coupling. Kept literal (not this
# module's own text -- tests/ is excluded from both the wheel and this scan).
FORBIDDEN = (
    "/home/arl",                 # developer-machine absolute paths
    "sim/cpm",                   # the cpm consumer copy
    "string-diagram-process-mining",
    "poset-mixture",             # the PMN consumer
    "poset_mixture",
    "stochastic_process_mining", # the SPM consumer
    "CLASS_EXTRACTION.md",       # consumer-only docs
    "OCCN_DEV.md",
    "RUNNING_EXAMPLE.md",
    "MINED_COSPANS.md",
)


def _shipping_files():
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if TESTS_DIR in path.parents or path == TESTS_DIR:
            continue
        if path.suffix in {".pyc", ".zip"}:
            continue
        yield path


def test_no_consumer_leak_in_shipping_tree():
    offenders = []
    for path in _shipping_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for needle in FORBIDDEN:
            if needle in text:
                rel = path.relative_to(PACKAGE_ROOT)
                offenders.append(f"{rel}: contains {needle!r}")
    assert not offenders, (
        "shipping tree leaks consumer/developer coupling:\n  "
        + "\n  ".join(offenders)
    )
