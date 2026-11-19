# Thermal state preparation

All 16 source stages, both six-bit Fourier windows with 10,000 samples each, and 50 delta cycles with the original four-bit mixing window and 1,000 thermal samples. All 400 resets execute as channels.

Run from the repository root with `python -m examples.synthesis.algorithms.quantum_state_preparation.gibbs.quantum_thermal_state_preparation.overview.main`. The [Notebook](walkthrough.ipynb) presents both measured windows and the thermal comparison.

Compute the windows, signed Glauber table and time controls from the original source constants. Use a local sample seed for each circuit. Compare both absolute complex Fourier states and the complete final density matrix, including all reset bits. The Gibbs comparison uses the source tolerance of 0.1.

[Source provenance](provenance.json).
