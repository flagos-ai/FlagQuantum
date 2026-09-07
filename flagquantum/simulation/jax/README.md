# JAX numerics

This package owns JAX-specific numerical kernels shared by the PyTorch-facing
runtime. It does not select devices, compile runtime executables, organize
distributed collectives, or produce execution evidence.

- Start in `primitives.py` for compute dtype, instruction matrices, local gate
  application, and basic observable kernels.
- Start in `statevector/kernels.py` for rank-local initialization, gate updates,
  pair-exchange assembly, and sharded observable loss.
- Start in `tensor_network.py` for JAX node construction, contraction,
  observables, slicing, and contracted-output loss.
- Start in `mps/` for JAX matrix-product-state kernels and differentiation.
- Import the owning submodule directly; this package does not re-export a JAX
  simulation facade.
- Run `python -m pytest tests/unit/test_jax_simulation_boundary.py -q` after a
  boundary-only change.
