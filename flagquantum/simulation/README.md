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
| Continuous-time Lindblad evolution | [lindblad.py](lindblad.py) |
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

## Continuous-time open-system evolution

`flagquantum.simulation.evolve_density_matrix` integrates the Lindblad master
equation on an explicit, strictly increasing output grid. Collapse entries pair
an operator with a physical rate; the named `amplitude_damping` operator is
`sqrt(rate) * |0><1|`, so its rate is never confused with the dimensionless
probability accepted by the instruction-level noise channel.

The deterministic CPU implementation uses two fourth-order Runge--Kutta steps
per output interval. Results report the actual method, order, maximum internal
step, precision, basis populations, requested observables, maximum trace drift,
and whether populations stayed within the documented tolerance (`1e-9` for
complex128 and `2e-5` for complex64). Density matrices are returned only when
requested. `plan_density_matrix_evolution` validates the identical request and
reports dimensions and retained-trajectory memory without evolving it.

The issue #234 reference problem can be run directly:

```bash
python examples/lindblad_evolution.py
```

The minimal SDK call is:

```python
import torch
import flagquantum as fq
import flagquantum.lindblad as lindblad

result = lindblad.solve(
    0.5 * fq.X(0) + 0.1 * fq.Z(0),
    "1",
    torch.linspace(0.0, 8.0, 161),
    collapse_operators=[lindblad.amplitude_damping(rate=0.1, wire=0)],
    observables={"z": fq.Z(0)},
)

print(result.populations.shape)          # torch.Size([161, 2])
print(result.maximum_trace_drift)
print(result.population_bounded)
```

Set `return_density_matrices=True` when the complete density trajectory is
needed. Otherwise it is omitted from the returned result.

[Detailed source map](IMPLEMENTATION.md) locates shared gate primitives,
rank-local kernels, specialized precision paths, and numerical migration rules.
