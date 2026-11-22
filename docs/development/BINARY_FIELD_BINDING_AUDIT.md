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
All 34 top-level definitions have native ownership below. Complete physical wrapper comparison and native acceptance are recorded in the final section.

The accepted Core polynomial, matrix and field utilities supply scalar values.
`ecosystem/binary_fields.py` consumes foreign field/poly/index objects and
returns only owned `BinaryField` or integer values. `simulation/binary_field_scalar.py`
owns the Python integer argument boundary, reduced scalar addition, signed
arbitrary-width powers and the Python Frobenius count boundary. Compiler retains existing primitive generators. This
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
| py_int_to_field_element | `field_constant_binding`, exact integer element admission before existing scale emission |
| alloc_qubit_array_obj | Caller-owned target/work ports and existing Program allocation; fresh and LIFO-reused source allocation records matched |
| resolve_init_target_obj | `field_word` and `prefix_field_resets`; explicit and source target="alloc" routes matched using the same owned target ports |
| check_register_size | `field_word`, shared operand conversion and width validation before returning any compiled artifact |
| register_gf2_field_class | Core owned field identity and external construction conversion; no parallel upstream-branded class |
| register_gf2_field_methods | Core arithmetic plus Simulation scalar adapters, constructor fixture and existing value observations; representation adaptation below |
| append_gf2_iadd_obj | `field_addition_binding`, shared integer/typed offset and positive quantum/classical/literal controls |
| append_gf2_imul_obj | `field_constant_binding`, multiply route with explicit field element checks |
| append_gf2_idiv_obj | `field_constant_binding`, divide route with explicit field element checks |
| append_init_gf2_mul_obj | `field_product_binding(mode="init")`, output reset outside control scope and caller-owned target |
| append_ixor_gf2_mul_obj | `field_product_binding(mode="xor")`, existing product generator and whole available work word |
| append_del_gf2_mul_obj | `field_product_binding(mode="erase")`, measured unmul or source controlled XOR dispatch |
| append_init_gf2_inverse_obj | `field_inverse_binding`, output reset and ordinary clean-work generator |
| append_init_gf2_inverse_with_scaffold_obj | `field_inverse_binding(retain_chain=True)`, output then original word-order scaffold resets |
| append_del_gf2_inverse_obj | `field_inverse_binding(erase=True)`, existing self-cleaning erase generator |
| append_del_gf2_inverse_with_scaffold_obj | `field_inverse_binding(retain_chain=True, erase=True)`, exact caller-retained chain consumption |
| append_init_gf2_div_obj | `field_division_binding(mode="init")`, output reset then existing ordinary quotient generator |
| append_init_gf2_div_with_scaffold_obj | `field_division_binding(mode="init", retain_chain=True)`, caller-retained inverse and chain |
| append_ixor_gf2_div_obj | `field_division_binding(mode="xor")`, including shared dividend/divisor input |
| append_del_gf2_div_obj | `field_division_binding(mode="erase")`, ordinary measured quotient deletion |
| append_del_gf2_div_with_scaffold_obj | `field_division_binding(mode="erase", retain_chain=True)`, retained inverse/chain deletion |
| append_gf2_phase_by_product_obj | `field_product_phase_binding`, classical-only masks and parity storage after caller classical words |

The native field dataclass equality and hash retain degree/modulus/generator
identity. Native dataclass representation and Python hash values follow their
native owners. Original `km.GF2Field` repr spelling and C++ word-hash numbers are
recorded representation differences. They carry no quantum primitive claim.

## Scalar and conversion evidence

The committed generator records 610 unchanged original constructor/add/power/Frobenius
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

## Complete physical binding acceptance

The complete wrapper fixture records 394 exact original marked streams through
degrees 1, 2, 3, 4, 8 and 17. It covers all init/xor/delete/scaffold wrappers,
integer/quantum/classical addition, constant scaling, product phase, shared
quotient inputs, all source control families, default and explicit AES moduli,
explicit and auto-allocated targets, fresh and reused LIFO clean pools, and
unused caller classical storage. Every opcode, operand, measurement slot,
feedback predicate and nested component mark matches without renaming. Complete
KMX marked streams also roundtrip through the existing codec.

`_binding.reserve_field_bits` places fresh feedback after existing caller
classical storage while retaining phase-mask reads. `prefix_field_resets` moves
component ranges after the original init resets. Source allocation remains in
the caller Program. The native wrappers accept already-owned target/work ports,
with MSB-first words. Source little-endian allocation arrays map by reversing
each word. Reused pool leases retain their actual LIFO allocation order. An init
target is reset even under an inactive enable. Retained inverse scaffold resets
retain source word order. A quantum-controlled delete follows the source XOR
route because its output need not contain the uncontrolled product.

The new 35-case wrapper regression exhausts every reset/measurement history for
product init/xor/delete at degrees one and two under literal, quantum and both
classical guard states. Inverse/division init-plus-erase, with ordinary or
retained scaffold, exhaust all histories at degrees one and two, including all
4096 degree-two ordinary division histories. Every branch compares unnormalized
complex amplitudes, including the outcome-dependent sign from resetting an
initial one target. Entangled spectators survive. Target and work storage return
to zero after deletion. Normal DynamicCircuit.append(FieldOperation(...))
executes the source-matched quantum-controlled initialized product.

Original collision refusals were independently probed for equal multiplication
factors, equal phase operands, self-addition and overlapping inverse input/output
at degrees one through four. The source refuses all sixteen at finish. Native
ownership checks may refuse earlier. The source-supported shared quotient inputs
remain accepted. Existing 78 unchanged original Python tests and complete
utility tests are reused. Kernel source fixtures and wider measured-erasure
acceptance retain their previous scope.

The Python Frobenius binding accepts bool/index/int-convertible counts with a
64-bit unsigned bound on the verified source platform. This boundary has its own
Simulation function and nine additional unchanged-source observations. The
positive-exponent C++ utility owner keeps its existing contract.

### Measured primitives

The degree-two default-field fresh allocation samples have these exact source
and native opcode counts. INFO_TAG marks are excluded from primitive totals.

| Wrapper | Actual primitive operations | Work ownership |
| --- | --- | --- |
| Initialized quantum-controlled product | 4 R, 9 CX, 3 SWAP, 5 CCX, 2 HMR, 4 CZ, 4 BIT_INVERT | Two clean temporary qubits, returned zero |
| Initialized inverse | 2 R, 3 CX | No temporary work at this degree |
| Initialized ordinary quotient | 4 R, 15 CX, 3 SWAP, 3 CCX, 2 HMR | Two clean temporary qubits, returned zero |
| Initialized retained quotient | 4 R, 12 CX, 3 SWAP, 3 CCX | Two retained inverse qubits, cleared by matching deletion |

### Analytical accounting

The frozen Python wrappers declare no independent analytical resource formula.
Existing Qualtran declaration models remain in their dedicated Compiler owners.
For example, the degree-two source product-erasure declaration estimates four
logical CZ calls and omits its measurement/parity overhead. The actual physical
wrapper fixture reports those measurements and classical operations separately.
No model estimate is reported as an executed primitive count.

### Final batch checks and boundaries

530 concentrated tests and 29 integration tests pass. Strict typing covers all
ten changed Python files. Black, Ruff, Integration scope, architecture, IR,
protected API and whitespace checks pass. The 394-stream and 610-scalar fixtures
retain the unchanged source binding blob. Evidence uses the
`ftqc-reference-env/g2-gf-binding-physical-*-20261122.log` prefix.

K-0226 through K-0228 are accepted as complete-file native ownership. Source
branding/C++ hash values and mutable allocator objects are adapted to native
representations and caller-owned Program allocation as described above. Higher
degree full statevector enumeration remains outside the small-branch tests and
uses the accepted wider kernel/stream evidence. Pending Proposals 116 through
119, original upstream defects and unrelated G2 parents retain their status.
