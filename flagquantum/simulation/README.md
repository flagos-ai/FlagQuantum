# Simulation

Numerical algorithms for quantum state evolution, observables, noise, and
differentiation. The design goal is a shared numerical foundation whose
precision and approximation behavior remain explicit across execution paths.

Simulation consumes Core semantics. It does not choose devices, schedule ranks,
submit jobs, authorize fallbacks, or construct public execution evidence.
Runtime owns those decisions and calls the appropriate numerical primitives.

## Choose the numerical owner

| Work | Entry point |
| --- | --- |
| Dense gates, sampling, and adjoints | [statevector/](statevector/README.md) |
| MPS updates, factorization, and truncation | [mps/](mps/README.md) |
| Contraction, slicing, and pullbacks | [tensor_network/](tensor_network/README.md) |
| Exact noisy evolution | [density_matrix.py](density_matrix.py) |
| Reusable arithmetic and precision | [numerics/](numerics/README.md) |
| Optional JAX kernels | [jax/](jax/README.md) |

For a typical local change, run the matching suite from the repository root:

```bash
python -m pytest tests/test_native_circuit.py -q
python -m pytest tests/test_mps.py -q
python -m pytest tests/test_tensor_network.py -q
python -m pytest tests/test_noise.py -q
```

Choose the suite for the affected representation, then broaden by the
[testing policy](../../docs/development/TESTING.md). Verify values and gradients
against independent references; truncation and emulated precision need their
own error envelope. Local numerical changes must not introduce a dependency on
Runtime orchestration.

[Detailed source map](IMPLEMENTATION.md) locates shared gate primitives,
rank-local kernels, specialized precision paths, and numerical migration rules.
