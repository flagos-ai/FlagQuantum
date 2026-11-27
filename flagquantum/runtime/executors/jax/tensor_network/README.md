# JAX tensor-network runtime

This package owns JAX tensor-network planning, execution orchestration,
collectives, Runtime result records, and evidence. Numerical nodes and local
contraction live in `simulation/tensor_network/jax/`.

- Start in `simulation/tensor_network/jax/models.py` for numerical nodes.
- Start in `records.py` for Runtime result records.
- Start in `planning.py` for slice-task and representation planning.
- Start in `contraction.py` for plan-aware slice contraction and collectives.
- Start in `execution.py` for sharded forward execution.
- Start in `gradients.py` for sliced reverse mode and parameter gradients.
- Import the owning module directly; this package does not add another facade.

Parameterized physical graph construction and local reduction arithmetic belong
to the existing Simulation owners. `gradients.py` selects precision and bridges
parameters; `contraction.py` admits ranks and performs real collectives. Run
`python -m pytest tests/unit/test_jax_tn_graph_ownership.py -q` with optional JAX
for the local owner-replacement path, then the distributed-plan tests for
changes to slicing or collective orchestration. No private graph forwarding
name is retained in Runtime.
