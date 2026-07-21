"""Small shared numerical primitives for the numpy NPMLE layer.

Kept out of ``oracle`` so ``npmle`` can use it without importing a private name
from ``oracle``.  numpy-only; imported only by the [estimate]-layer modules, so
it does not affect the numpy-free core.
"""

from __future__ import annotations

import numpy as np


def _log_mean_exp_rows(ratios: np.ndarray) -> np.ndarray:
    """Per-row log((1/G) sum_g exp(ratios[., g])), with all-(-inf) rows
    mapping to -inf instead of NaN."""
    G = ratios.shape[1]
    hi = ratios.max(axis=1)
    out = np.full(ratios.shape[0], -np.inf)
    ok = np.isfinite(hi)
    if ok.any():
        out[ok] = (
            hi[ok]
            + np.log(np.exp(ratios[ok] - hi[ok, None]).sum(axis=1))
            - np.log(G)
        )
    return out
