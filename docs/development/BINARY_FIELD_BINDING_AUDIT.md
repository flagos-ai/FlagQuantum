# Binary field Python binding ownership and remaining closure

## Frozen source and batch boundary

The complete source group is K-0226 through K-0228 at immutable Kickmix revision
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, Apache-2.0. The implementation
file has blob `5f111727a1e383394ef856e7104ff4817ae12f31`. The declaration and
original Python test files have blobs
`c38de4b029b7bc82c5b172f026377daac639b342` and
`45bd9641d58825ae820e7c24469035f63127a370`. All nine recorded binding and
utility source blobs were reverified after field construction publication.
The earlier intake ran all 78 unchanged original Python tests with zero skips.
The complete parent group stays open until every physical wrapper is compared.

The accepted Core polynomial, matrix and field utilities supply scalar values.
`ecosystem/binary_fields.py` consumes foreign field/poly/index objects and
returns only owned `BinaryField` or integer values. `simulation/binary_field_scalar.py`
owns the Python integer argument boundary, reduced scalar addition and signed
arbitrary-width powers. Compiler retains existing primitive generators. This
module introduces no foreign object dependency in Core, Compiler or Simulation.

## Full definition map

| Original top-level definition | Current owner and disposition |
| --- | --- |
| py_int_to_gf2_poly | `binary_field_operand` for scalar arguments and `binary_polynomial_value` for construction; retained 1024-bit bound |
| gf2_poly_to_py_int | Core integer values, no limb copy required |
| is_py_int | External constructor rejects bool while scalar int arguments accept it as the source does |
| py_repr | Python repr at external diagnostics; native error wording is owned locally |
| is_galois_field_obj | `_external_field`, three-attribute dispatch |
| is_galois_poly_obj | `binary_polynomial_value`, field/degree/coeffs dispatch |
| require_characteristic_2 | `_characteristic`, integer two with bool refusal |
| parse_degree_arg | `binary_field_value`, source integer degree range before construction |
| parse_poly_arg | `binary_polynomial_value`, original string, polynomial, class characteristic and __index__ admission order |
| make_gf2_field | `binary_field_value` delegates to shared Core construction |
| gf2_field_from_galois_obj | `binary_field_value`, declared modulus and optional primitive retained |
| resolve_gf2_field | `resolve_binary_field`, owned value/default/external field and exact register width |
| py_int_to_field_element | Core is_element plus existing Compiler constant bounds; complete wrapper comparison pending |
| alloc_qubit_array_obj | Existing caller-owned Program allocation; allocation-order wrapper comparison pending |
| resolve_init_target_obj | Existing caller target allocation/binding; reset/admission wrapper comparison pending |
| check_register_size | Existing operand-word and field validation; complete wrapper comparison pending |
| register_gf2_field_class | Core owned field identity and external construction conversion; no parallel upstream-branded class |
| register_gf2_field_methods | Core arithmetic plus Simulation scalar adapters, constructor fixture and existing value observations; representation adaptation below |
| append_gf2_iadd_obj | physical_field_addition and operand/control binding; complete wrapper comparison pending |
| append_gf2_imul_obj | physical_field_constant_scale; constant/field argument comparison pending |
| append_gf2_idiv_obj | physical_field_constant_scale inverse route; constant/field argument comparison pending |
| append_init_gf2_mul_obj | physical_field_product plus target reset/allocation; binding closure pending |
| append_ixor_gf2_mul_obj | physical_field_product and typed control; binding closure pending |
| append_del_gf2_mul_obj | physical_field_product erase route; binding closure pending |
| append_init_gf2_inverse_obj | physical_field_inverse and target reset/clean leases; binding closure pending |
| append_init_gf2_inverse_with_scaffold_obj | physical_field_inverse retained chain and target/scaffold reset; binding closure pending |
| append_del_gf2_inverse_obj | physical_field_inverse erase; binding closure pending |
| append_del_gf2_inverse_with_scaffold_obj | physical_field_inverse retained erase; binding closure pending |
| append_init_gf2_div_obj | physical_field_division and target reset; binding closure pending |
| append_init_gf2_div_with_scaffold_obj | physical_field_division retained chain; binding closure pending |
| append_ixor_gf2_div_obj | physical_field_division ordinary accumulation; binding closure pending |
| append_del_gf2_div_obj | physical_field_division erase; binding closure pending |
| append_del_gf2_div_with_scaffold_obj | physical_field_division retained erase; binding closure pending |
| append_gf2_phase_by_product_obj | physical_field_product_phase; mask binding comparison pending |

The native field dataclass equality and hash retain degree/modulus/generator
identity. Native dataclass representation and Python hash values follow their
native owners. Original `km.GF2Field` repr spelling and C++ word-hash numbers are
recorded representation differences. They carry no quantum primitive claim.

## Scalar and conversion evidence

The committed generator records 601 unchanged original constructor/add/power
observations. These cover string/integer/index/polynomial inputs, explicit
modulus/generator choices, malformed degrees and moduli, odd characteristics,
boolean distinctions, all constructor error types, widths 1/2/4/8/65/128/512,
unreduced operands and positive/negative 160-bit exponents. The native fixture
test compares all results and TypeError/ValueError outcomes.

A separate live probe uses real galois/NumPy objects at degrees 1/4/8/16. All
sixteen successful constructor routes and six nonbinary refusal routes agree.
An object with divergent __int__ and __index__ confirms original int conversion
after index admission. A native regression preserves this distinction. Receipt
`ftqc-reference-env/g2-gf-binding-real-conversion-20261122.json` retains results.
Additional native validation covers foreign characteristic None, resolver
widths, retained object identity and a 4097-bit signed exponent.

Core raw polynomial addition preserves all coefficient bits. The original
Python field addition first reduces both operands. These contracts have
separate owners and an explicit regression. Signed Python powers invert before
calling the nonnegative Core power scan. The original zero inverse convention
is retained. Scalar argument bools are converted to integer values while
constructor polynomial bools remain refused.

```sh
python tests/unit/fixtures/generate_binary_field_binding.py \
  --output tests/unit/fixtures/binary_field_binding.json
python -m pytest tests/unit/test_binary_field_binding.py -q
```

All scalar conversions emit zero quantum primitives and allocate zero quantum
or classical workspace. The concentrated field execution suite reuses full
complex amplitudes, spectator/inverse restoration, measured-erasure branches,
clean-work and exact primitive evidence. Gate/resource models retain their
existing separate owners. The scalar module does not extend a physical model.

Local module verification includes 495 concentrated and 29 integration tests,
strict typing on four Python files, Black, Ruff, Integration scope, architecture,
IR, protected API and whitespace checks. Logs use the
`ftqc-reference-env/g2-gf-binding-scalar-*-20261122.log` prefix.

## Remaining complete-file acceptance

Compare all init/xor/delete/scaffold wrapper streams, operand/control inputs,
allocation and reset ordering, and rejection-before-mutation behavior against
the frozen binding. Reuse prior complete product, inverse, division and erasure
evidence for unchanged kernels. All K-0226 through K-0228 parents remain open
until that comparison closes. Pending Proposals 116 through 119, source defects
and unrelated G2 files retain their existing status.
