# JAX MPS runtime

This package owns JAX MPS planning, execution orchestration, collectives,
Runtime records, and evidence. Numerical MPS operations remain in
`simulation/jax/mps/`.

- Start in `planning.py` for representation and training planning.
- Start in `result.py` for forward execution results.
- Start in `training_records.py` for training, parameter-flow, and shard records.
- Start in `gradient_result.py` for parameter-gradient execution results.
- Start in `execution.py` for sharded forward execution.
- Start in `gradients.py` for parameter-gradient execution.
- Import the owning module directly; this package does not add another facade.
