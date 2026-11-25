# Qubitized Hamiltonian evolution

All 15 source stages, the original four Pauli terms on two data qubits, time 22, degree 33 and 10 qubits for evolution. Compare both complex projected vectors using the declared scales.

Run from the repository root with `python -m examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.main`. Open the [Notebook](walkthrough.ipynb) to compare both complex projections.

Recompute the Jacobi-Anger coefficients by cosine quadrature at the source degree. Native synth prepares the amplitudes at double precision. Public Circuit SELECT, reflection and inverse preparation implement the 63 controlled walk steps. Use a reproducible local input seed in place of the unseeded source draw. Circuit resources describe the emitted native IR.

[Source provenance](provenance.json).
