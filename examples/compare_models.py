"""Minimal, dependency-free example: canonically decompose a process model and
compare two models by the stochastic-matrix distance (Result 3).

Runs on a bare ``pip install procposets`` (no extras).

    uv run python examples/compare_models.py
"""
from procposets import leaf, modular_decompose, par, smd, then


def main() -> None:
    # A "model" is a weighted list of labelled posets (its variants).
    seq = [(then(leaf("a"), leaf("b"), leaf("c")), 1.0)]        # a < b < c
    conc = [(then(leaf("a"), par(leaf("b"), leaf("c"))), 1.0)]  # a < (b ∥ c)

    # The total modular decomposition is unique -> a canonical block tiling.
    tiling = modular_decompose(then(leaf("a"), par(leaf("b"), leaf("c")))).canonical()
    print("canonical block tiling:", tiling)  # (a ; (b * c))

    # Stochastic-matrix distance + the per-state (per-block) angle breakdown.
    dist, per_state = smd(seq, conc, normalize=True)
    print(f"normalized SMD(seq, conc) = {dist:.4f}")
    for state, angle in sorted(per_state.items()):
        print(f"  {state:>8}: {angle:.4f}")


if __name__ == "__main__":
    main()
