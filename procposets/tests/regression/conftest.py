"""Let the ported regression suite run on a numpy-only install: the few
networkx-backed cospan tests (the [graph] layer) are ignored when networkx is
absent, so `pip install procposets && pytest` doesn't error on collection.
The core (poset / estimation / distance / pure-cospan) tests always run."""
import importlib.util

collect_ignore = []
if importlib.util.find_spec("networkx") is None:
    collect_ignore += [
        "test_cpm_occurrence.py",
        "test_cpm_trace_language.py",
        "test_cpm_splice.py",
    ]
