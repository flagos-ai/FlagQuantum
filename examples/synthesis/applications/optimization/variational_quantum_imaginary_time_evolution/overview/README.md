# Variational imaginary-time MaxCut

All 22 source stages, a fresh four-node weighted graph, six ordered generators, fourteen imaginary-time steps and 128 shots per histogram. Report update energy, independently measured energy, rounded cut and the exact four-node comparison.

Run from the repository root with `python -m examples.synthesis.applications.optimization.variational_quantum_imaginary_time_evolution.overview.main`. Open the [Notebook](walkthrough.ipynb) for the graph and finite-shot trajectory.

The source graph generator has n=k=4, so the native workflow constructs the complete four-node graph and draws the weights from seed zero. It recomputes the sorted generators from that fresh graph and specializes the six angles before each local execution. A separate local shot seed replaces hosted randomness. The rounded partition keeps its actual measured cost.

[Source provenance](provenance.json).
