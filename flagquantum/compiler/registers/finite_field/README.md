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

`linear_component.py` stores the shared source-order PLU child calls and their
free output permutation. `frobenius_component.py` composes one such child between
coefficient-order reversals, using the same mathematical matrix as
`field_linear.py`. Both use zero additional quantum work and classical slots.
The exact inverse preserves every coherent amplitude and restores the original
physical input. Frozen source matrix-power and ignored-adjoint defects remain
explicit in the square audit. To bind, execute and restore a complete word, run
`python -m examples.algorithms.register_field_frobenius` and the corresponding
`test_binary_linear_component.py` and `test_field_frobenius_component.py` tests.

`polynomial_addition_component.py` composes the existing typed coefficient views
and field-addition calls, preserving highest-degree-first order and literal
source adjoints. An explicit `packed_dtype` retains the canonical quantum
binary-polynomial descriptor approved by Proposal 109. The default keeps its
existing opaque view. Both routes share the same coefficient split/join owner.
The internal field constant
adder's `adjoint=True` reverses X calls, while quantum field XOR retains its
self-adjoint bit order. Run
`python -m examples.algorithms.register_field_polynomial_addition` and
`tests/unit/test_field_polynomial_addition_component.py` for coherent append,
original source and clean-spectator restoration checks.

`monomial_component.py` and `constant_multiplication_component.py` retain stored
shift and shared PLU calls. `polynomial_product_component.py` owns the recursive
coefficient-array multiply-accumulate chains and clean high padding.
`multiplication_component.py` owns quadratic RIGHT/THRU/LEFT products.
`karatsuba_component.py` composes the shared polynomial, constant and shift
owners into the source implementation and public allocation shell. Existing
`product_erasure_program.py` supplies the measured inverse channel. Numerical
codecs and independent source models retain their original owners. Run
`python -m examples.algorithms.register_field_multiplication` for normal append
and returned physical aliases. The multiplication audit records original
linear-adjoint defects and unsupported domains separately from accepted native
mathematical execution and resource models.

`inverse_component.py` composes the original Itoh--Tsujii schedule from shared
addition, Frobenius and Karatsuba stored children. Its RIGHT junk array is
retained logical output, distinct from temporary clean padding. Run
`python -m examples.algorithms.register_field_inverse` for the normal append
path and returned aliases. `FIELD_INVERSE_FILE_AUDIT.md` records the complete
source map, exact-native adjoint adaptation and remaining admission boundaries.

`field_binding.py`, `product_binding.py`, `inverse_binding.py` and
`division_binding.py` compose the complete source Python wrapper semantics over
these existing physical generators. Targets, scaffold and clean leases belong
to the caller Program. The shared private binding helpers preserve init reset
order, component ranges and fresh classical feedback after caller storage.
External field objects are consumed by Ecosystem `binary_fields`; Python scalar
addition, powers and Frobenius counts belong to Simulation `binary_field_scalar`.
Run `tests/unit/test_field_physical_binding.py` for exact original marked streams,
normal append and exhaustive small complex measurement branches. See
`docs/development/BINARY_FIELD_BINDING_AUDIT.md` for the complete definition map,
source allocation translation and separate physical/model accounting.


The literal source profile has a separate owner from mathematical field powers:
`square_component.py` preserves the frozen square schedule and ignored physical
inverse flag; `field_frobenius_program` retains correct powers and exact inverse.
Both reuse `linear_component.field_linear_program` for MSB word / coefficient-array
views. `addition_component.field_addition_decomposition_program` preserves the
actual source degree-prefix graph on admitted raw field storage; the mathematical
binary adder retains its binary-only contract. These transforms allocate no work
and own no backend execution. Simulation's explicit `resolve_quantum_field_storage`
realizes exact scalar aliases without changing the literal metadata projection.

To work on the source profile, change its small component, keep the mathematical
owner separate, and run `tests/unit/test_field_square_decomposition.py` or
`tests/unit/test_field_addition_literal.py`. They show the shortest normal
`Circuit.append(FieldOperation(...))` path, actual source operators, virtual
output aliases, coherent spectators and inverse restoration. Their audits retain
source defects and independently measured primitive/model/binding costs.


For literal Karatsuba execution, use the separately named
`karatsuba_decomposition_implementation` (arbitrary THRU h) and
`karatsuba_decomposition_program` (RIGHT result) in `karatsuba_component.py`.
They share the mathematical owner's schedule and polynomial dependencies while
retaining the original physical linear-adjoint defect. Run
`tests/unit/test_karatsuba_literal.py` for actual untouched-source operands,
complex operators and every degree-five raw target with clean-padding checks.
The literal output is not promised to be a numerical field product. Complete
catalog adjoint-identity integration remains in the multiplication audit.


`binary_linear_decomposition_program` is the original physical recipe on any
nonempty square binary matrix, including missing pivots. It shares the packed
Core factorization and Compiler gate schedule with invertible synthesis, but
never promises the matrix's numerical action for singular input. Its declared
inverse uses its own named entry; bind the returned first function explicitly
when that entry is selected. `constant_multiplication_decomposition_program`
composes this child and includes zero. `monomial_decomposition_program` uses
source signed loop counts, keeping negative counts distinct from mathematical
negative powers. The normal catalog resolves zero constants and negative shifts
to these owners. Run `tests/unit/test_field_linear_literal.py` for the executable
append, source-operator, alias and clean-work scenarios. Measured primitive
counts and defective source declarations are separate in the multiplication audit.


Raw nonbinary physical consumers use existing Core storage, not mathematical
binary admission. `constant_matrix_decomposition_program` receives a numerical
matrix already constructed by Simulation. `monomial_decomposition_program`
retains the source field-degree conditional and raw-width rotations.
`field_multiplication_decomposition_program` shares the quadratic schedule and
supports RIGHT, arbitrary THRU and owned LEFT profiles. The degree-one branch
in `inverse_decomposition.py` allocates a result and calls literal prefix
addition. None claims numerical nonbinary arithmetic or owns a backend.

Run `tests/unit/test_raw_field_components.py` for a short source constructor,
Program binding and `Circuit.append(FieldOperation(...))` scenario. It covers
complete raw complex states, all small constants, source aliases, matching LEFT
ownership/release and an SDK-blocked replacement execution. The two whole-file
audits record separate primitive/model costs and remaining declared graph / erasure
work; successful raw execution does not close those broader parents by itself.


Declared constant and linear inverse identities survive literal compilation even
when their primitive recipes coincide. Examples qualify them only against a
complete matching caller, with generated names treated as opaque. Quadratic
left/right calls likewise retain separate named entries when needed. The normal
source catalog selects the original literal Karatsuba graph; mathematical
FieldProduct execution still uses the correct product owner.

`product_erasure_program` accepts either existing explicit polynomial metadata
or an explicit modulus with a concrete raw field descriptor. The existing
FieldProduct example supports `adjoint=True` for measured retirement and reserves
private classical slots after caller storage. Run
`python -m examples.algorithms.register_field_multiplication` for normal forward
and measured-adjoint append. `tests/unit/test_field_erasure_all_branches.py`
reuses the common exact phase checker to certify all256 eight-bit histories on
all65536 input pairs, including common Kraus amplitude and clean physical clearing.
The same test retains the defective original product/erase channel as a failure.

`inverse_component.py` owns one Itoh–Tsujii root schedule with explicit child
profiles. `field_inverse_program` uses mathematically correct square/product
children; `inverse_decomposition.field_inverse_decomposition_program` preserves
the complete literal source graph and its declared directions. Logical junk is
a RIGHT output, never claimed as cleared private work.

The field-power catalog draws the literal graph even when it contains an
original invalid clean release. The example `FieldInverse(..., literal=True)`
binds that graph, then asks Simulation's `_clean_releases` owner to prove every
actual clean free before ordinary append. Its default remains the mathematical
profile. Compiler does not execute the proof kernel or choose a backend. There
is no degree whitelist or corrected-source substitute. Run
`tests/unit/test_field_inverse_literal.py` and
`tests/unit/test_clean_release_analysis.py`; the complete inverse audit separates
source defects, exact complex/source evidence, proof costs and resource models.

`polynomial_addition_component.py` shares one coefficient-call schedule between
concrete arithmetic and `logical_field_polynomial_addition_program`. The latter
retains finite polynomial extent over symbolic fields using signature-only
atomic children. Declarations have no executable body; resolving a raw storage
width alone does not supply a field-addition implementation. The field-polynomial
catalog preserves actual source nominal-binding failures, while the separate
mathematical owner remains available for supported native inputs. Run
`tests/unit/test_symbolic_field_polynomial.py` for source traces and declaration
admission, and `tests/unit/test_raw_field_polynomial_views.py` for concrete raw
binary/nonbinary identity operators and complex spectators.
