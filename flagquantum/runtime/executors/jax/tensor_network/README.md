# JAX tensor-network runtime

This package owns JAX tensor-network planning, execution orchestration,
collectives, Runtime result records, and evidence. Numerical nodes and local
contraction live in `simulation/jax/tensor_network/`.

- Start in `simulation/jax/tensor_network/models.py` for numerical nodes.
- Start in `records.py` for Runtime result records.
- Start in `planning.py` for slice-task and representation planning.
- Start in `contraction.py` for plan-aware slice contraction and collectives.
- Start in `execution.py` for sharded forward execution.
- Start in `gradients.py` for sliced reverse mode and parameter gradients.
- Import the owning module directly; this package does not add another facade.
