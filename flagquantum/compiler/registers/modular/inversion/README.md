# Modular inverse steps

This package owns arithmetic, ordering, parity, exchange and termination steps of the inverse algorithm.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [inverse_algorithm.py](inverse_algorithm.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_inverse_algorithm.py -q
```

The [register package guide](../../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.
