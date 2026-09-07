# JAX statevector runtime

This package owns JAX statevector execution orchestration, distributed
collectives, Runtime records, and evidence. Numerical statevector operations
remain in `simulation/jax/statevector.py`.

- Start in `execution.py` for sharded execution and parameter gradients.
- Import the owning module directly; this package does not add another facade.
