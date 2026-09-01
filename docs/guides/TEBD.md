# Constrained MPS TEBD

FlagQuantum exposes `flagquantum.experimental.simulation.run_tebd` as an experimental, fail-closed
single-device MPS path for second-order imaginary-time evolution.

```python
import flagquantum as fq

hamiltonian = fq.transverse_field_ising(8, coupling=1.0, field=0.7)
result = fq.experimental.simulation.run_tebd(
    hamiltonian,
    n_wires=8,
    total_time=2.0,
    time_step=0.02,
    max_bond=64,
    cutoff=1e-12,
    initial_state="+x",
)
print(result.final_energy)
print(result.to_dict())
```

## Supported boundary

- static real Pauli Hamiltonians with one-site or adjacent two-site terms;
- open chains, batch size one, `complex64` or `complex128`;
- second-order Strang splitting and imaginary-time evolution;
- product initial states: `+x`, `+z`, `-z`, `neel_z`, and `domain_wall`;
- CPU or one GPU through the native MPS implementation.

The result records the complete execution configuration, program hash, energy
history, squared norms before normalization, normalization errors, cumulative
discarded weight, observed bond dimension, actual device, precision, and the
explicit `single_device_fast_path` execution semantics. `to_dict()` is JSON
serializable and never materializes the dense state.

## Unsupported boundary

Real-time evolution, periodic or nonlocal interactions, terms wider than two
sites, differentiable coefficients, batching, TDVP, distributed execution, and
production/scalability claims are unsupported. The implementation never falls
back to a dense statevector. It raises an error if truncation weight cannot be
measured or if a numerical result becomes non-finite.
