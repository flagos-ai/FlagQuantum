# JAX MPS runtime

This package owns JAX MPS planning, execution orchestration, collectives,
Runtime records, and evidence. Numerical MPS operations remain in
`simulation/jax/mps/`.

- Start in `planning.py` for representation and training planning.
- Start in `result.py` for forward execution results.
- Import the owning module directly; this package does not add another facade.
