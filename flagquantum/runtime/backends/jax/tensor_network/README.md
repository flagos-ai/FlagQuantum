# JAX tensor-network runtime

This package owns JAX tensor-network planning, execution orchestration,
collectives, Runtime records, and evidence. Numerical tensor-network
operations remain in `simulation/jax/tensor_network.py`.

- Start in `records.py` for tensor-network nodes and result records.
- Start in `planning.py` for slice-task and representation planning.
- Start in `contraction.py` for plan-aware slice contraction and collectives.
- Start in `execution.py` for sharded forward execution.
- Import the owning module directly; this package does not add another facade.
