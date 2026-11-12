# Integer arithmetic

This package owns addition, subtraction, integer products, squares and their component costs.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [addition.py](addition.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

```bash
python -m pytest tests/unit/test_register_addition.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

[Factorized addition programs](../../../../docs/development/ADDITION_COMPONENTS.md)
compose typed CNOT/AND calls over the same logical sequence as the gate emitter.
Physical binding retains the existing append workflow; explicit numerical
suppliers can contract large closed graphs without a full addition matrix.
Run `tests/unit/test_addition_component.py` for the complete contributor scenario.
