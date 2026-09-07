# JAX numerics

This package owns JAX-specific numerical kernels shared by the PyTorch-facing
runtime. It does not select devices, compile runtime executables, organize
distributed collectives, or produce execution evidence.

- Start in `primitives.py` for compute dtype, instruction matrices, local gate
  application, and basic observable kernels.
- Import the owning submodule directly; this package does not re-export a JAX
  simulation facade.
- Run `python -m pytest tests/unit/test_jax_simulation_boundary.py -q` after a
  boundary-only change.
