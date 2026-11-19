# Quantum likelihood estimation

All 21 source stages, four one-qubit Hamiltonians, standard particle selection, and eight entropy-annealing steps with 20 neighbors per update. Each real experiment draws one native outcome and runs a complete posterior update.

Run from the repository root with `python -m examples.synthesis.algorithms.search_and_optimization.quantum_likelihood_estimation.overview.main`. The [Notebook](walkthrough.ipynb) shows the actual sampled posterior histories.

Specialize each experimental setting through native qfunc. Execute the proposal likelihoods using FlagQuantum batched Circuit gates, with explicit per-item angles. Verify both paths against independent complex amplitude formulas. Keep the source stopping limits and use a reproducible local seed for the physical outcome stream.

[Source provenance](provenance.json).
