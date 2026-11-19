# Symmetric-state compression

All twelve original source stages, five-qubit uniform input, three-qubit single-excitation input, three-qubit nonsymmetric input and 2048 actual native samples per case.

Run from the repository root with `python -m examples.synthesis.community.paper_implementation_project.quantum_compression_algorithm_for_symmetric_states.overview.main`. The [Notebook](walkthrough.ipynb) compares actual sampled counts with exact native probabilities.

The single-excitation source lists three equal amplitudes of 1/3. Their squared norm is 1/3, so the native workflow explicitly normalizes those amplitude ratios before preparation. Full absolute complex vectors are compared with an independent Circuit. Native shot seeds are local, with no claim of matching earlier hosted counts.

[Source provenance](provenance.json).
