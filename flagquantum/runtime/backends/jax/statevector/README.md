# JAX statevector runtime

This package owns JAX statevector execution orchestration, distributed
collectives, Runtime records, and evidence. Numerical statevector operations
remain in `simulation/jax/statevector/kernels.py`.

- Start in `execution.py` for sharded execution and parameter gradients.
- Start in `records.py` for statevector shard, execution, and training records.
- Start in `gradient_records.py` for parameter-gradient results and planning data.
- Start in `kernels.py` for plan-aware collective execution.
- Start in `training.py` for training preflight and plan construction.
- Import the owning module directly; this package does not add another facade.
