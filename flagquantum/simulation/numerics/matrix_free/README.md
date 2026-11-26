# Matrix-free numerical primitives

This directory owns linear-adjoint/autograd helpers, Krylov exponential action,
GMRES and exact Pauli-sum action. It consumes tensor operations and callable
linear actions; Dynamics owns time-grid integration and Runtime owns execution
policy and evidence. No resource selection or Runtime dependency belongs here.

Use `exponential.py`, `linear_solve.py`, `autograd.py` or `hamiltonian.py` for
algorithm changes. Imports outside the family use these actual owners. The old
Hamiltonian module only locates the two concrete types embedded in historical
serialized public Lindblad plans; it does not forward numerical operations.

From the repository root, run:

```sh
python -m pytest tests/unit/test_matrix_free_structure.py tests/unit/test_lindblad_matrix_free_hamiltonian.py tests/unit/test_lindblad_batching_and_gradients.py tests/unit/test_lindblad_integrators.py -q
```

Keep residual, gradient and memory-bound assertions intact. This relocation
adds no integrator, approximation, fallback, Rust implementation or performance
claim.
