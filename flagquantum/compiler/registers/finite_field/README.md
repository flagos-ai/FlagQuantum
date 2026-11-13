# Binary field arithmetic

This package owns polynomial-basis field operations, linear maps, product and inverse chains and erasure.

Core, shared Compiler facilities and the register components used by these
implementations are allowed dependencies. Backend selection and numerical
execution belong to Runtime and Simulation. Public APIs and IR schemas follow
their existing change-proposal process.

Start with [coefficients.py](coefficients.py). Preserve register order, feedback slots and caller-owned
workspace. From the repository root, run the focused scenario tests.

`addition_model.py` owns the concrete and symbolic source declarations, including
zero-count X edges and the unknown-constant width bound. Numerical field
construction and scalar callbacks live in Simulation's `finite_field_addition`.
`coefficients.py` remains the executable gate authority. Both additions are
big-endian, self-inverse and require zero work wires and zero measurements.
`examples/notebooks/binary_field_addition.ipynb` and
`binary_field_add_constant.ipynb` show metadata, declared graphs and normal
`circuit.append(op)` execution. See `docs/development/FIELD_ADDITION_FILE_AUDIT.md`
for the whole-file mapping and the separate source fixture boundaries.

```bash
python -m pytest tests/unit/test_finite_field_addition.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.
