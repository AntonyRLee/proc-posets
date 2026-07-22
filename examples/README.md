# Examples

Small, self-contained, deterministic scripts demonstrating proc-posets. Each is
runnable with `uv run python examples/<name>.py`.

- **`compare_models.py`** — dependency-free core: canonical modular
  decomposition + the stochastic-matrix distance between two models.

(More per-layer examples — an `[estimate]` NPMLE fit, a `[pm4py]`
discovery → signature — are welcome; keep them small and seeded so they can
double as CI smoke tests.)
