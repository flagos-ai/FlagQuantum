# JAX tensor-network numerics

This package owns dependency-light JAX tensor-network mathematics. It does not
select devices, organize collectives, consume Runtime plans, or emit evidence.

- Start in `kernels.py` for node construction, contraction, slicing,
  observables, and contracted-output loss.
- Import the owning module directly; this package does not re-export a facade.
