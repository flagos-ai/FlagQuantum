# JAX statevector numerics

This package owns dependency-light JAX statevector mathematics. It does not
select devices, organize collectives, consume Runtime plans, or emit evidence.

- Start in `kernels.py` for shard initialization, gate updates, pair exchange,
  all-to-all assembly, and observable loss.
- Import the owning module directly; this package does not re-export a facade.
