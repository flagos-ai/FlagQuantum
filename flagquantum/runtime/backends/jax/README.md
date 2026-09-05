# JAX Runtime Backend

This directory owns JAX backend selection, PyTorch/JAX bridging, execution
policy, distributed coordination, and evidence collection.

It does not own statevector, MPS, or tensor-network numerical kernels. Local
numerical operations live under `flagquantum/simulation/` and are imported
explicitly; this boundary does not proxy Simulation internals dynamically.

For local execution, start with `kernel.py`. For MPS circuit lowering and the
nearest-neighbor CX fast-path decision, start with `mps_kernel.py`. The latter
may recognize and lower circuit structure, but delegates tensor initialization,
updates, contraction, and observable evaluation to Simulation.

Run the boundary checks with:

```bash
pytest tests/unit/test_jax_simulation_boundary.py
```
