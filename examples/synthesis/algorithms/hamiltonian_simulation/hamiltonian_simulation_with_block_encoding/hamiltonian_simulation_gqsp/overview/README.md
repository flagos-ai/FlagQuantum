# Hamiltonian evolution with fresh GQSP phases

Run the original time-22, degree-33 Laurent approximation for the four-term
Hamiltonian. The workflow computes all 67 projected rotations by spectral
completion and stable endpoint stripping, verifies the polynomial
reconstruction, and executes the original five-qubit GQSP program.

Reuse the native synth amplitude preparation and public Pauli SELECT
operations from the paired qubitization example. Reflection conjugation
gives the inverse walk directly from the forward walk and its Hermitian
block encoding. Compare the absolutely scaled complex amplitudes with a fresh
dense operator exponential.

Run the Notebook or invoke
`python -m examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_gqsp.overview.main`.
