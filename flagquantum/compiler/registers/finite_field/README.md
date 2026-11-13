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

`linear_resource_model.py` evaluates concrete binary PLU costs independently of
gate emission. `field_linear.py` retains the symbolic Frobenius bound and exact
mathematical emitter. Simulation's `finite_field_frobenius` owns numerical
matrices and field values. The complete `binary_field_square.ipynb` demonstrates
normal append, free output mapping, explicit binding and inverse restoration.
See `docs/development/FIELD_SQUARE_FILE_AUDIT.md` for source parity ranges and
the preserved matrix/adjoint differences.

```bash
python -m pytest tests/unit/test_finite_field_addition.py -q
```

The [register package guide](../README.md) provides executable consumer examples,
source-parity requirements, resource-counting boundaries and the relocation map.

`field_polynomial_addition.py` and `field_polynomial_layout.py` own descending
coefficient traversal and gate-free word views. Source-specific `adjoint=True`
reverses coefficient calls while retaining each self-adjoint field XOR child's
bit order. Simulation's `finite_field_polynomial_addition` owns value callbacks
and constant field/degree validation. The complete polynomial addition, constant
addition and view notebooks under `examples/notebooks` demonstrate the normal
append path. See `docs/development/FIELD_POLYNOMIAL_FILE_AUDIT.md` for the complete
nine-file mapping, literal inverse ordering and remaining generic boundaries.

`field_multiplication.py` owns source-ordered quadratic multiplication and its
arbitrary-target `plus_equal_prod=True` variant, alongside the existing exact
Karatsuba owner. `multiplication_model.py` owns recursive polynomial declarations
and independent field gate-category models. Numerical products, monomial values,
coefficient arrays and reduction matrices live in Simulation's
`finite_field_multiplication`. The complete `binary_field_multiplication.ipynb`
retains all nine source examples and uses the ordinary append protocol. See
`docs/development/FIELD_MULTIPLICATION_FILE_AUDIT.md` for the complete definition
map, immutable-source adjoint witness and retained generic boundaries.
