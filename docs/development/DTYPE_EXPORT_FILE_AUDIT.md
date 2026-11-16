# Register data-type export source audit

The Qualtran v0.7.0 initializer at
`bcb17bb25360ed68ce3bc084645db3e929799389` has Git blob
`bb233e268a3baee166641123f9d927ac64f1adde`. Its 22 exports point to the
original definitions in `qualtran._infra.data_types`.

| Original exports | Native declaration owner | Numerical owner |
| --- | --- | --- |
| `QCDType`, `CDType`, `QDType` | `RegisterType` in `core/ir/program/registers.py` represents both storage kinds. Quantum and classical widths and the storage parameter express the category. | Each encoding owns its numerical contract. |
| `QAny` | `opaque_register_type` | Opaque register metadata |
| `QBit`, `CBit` | `quantum_bit_type`, `classical_bit_type` | `simulation/bit_encoding.py` |
| `QInt`, `CInt` | `signed_register_type`, `classical_signed_register_type` | `simulation/integer_encoding.py` |
| `QIntOnesComp`, `CIntOnesComp` | `ones_complement_register_type`, `classical_ones_complement_register_type` | The shared ones-complement integer codec |
| `QUInt`, `CUInt` | `unsigned_register_type`, `classical_unsigned_register_type` | The shared unsigned integer codec |
| `BQUInt`, `BCUInt` | `bounded_unsigned_register_type`, `classical_bounded_unsigned_register_type` | The bounded integer validation and codec modules |
| `QFxp`, `CFxp` | `fixed_point_register_type`, `classical_fixed_point_register_type` | `simulation/fixed_point_encoding.py` |
| `QMontgomeryUInt`, `CMontgomeryUInt` | `montgomery_register_type`, `classical_montgomery_register_type` | `simulation/montgomery_values.py` and the shared unsigned codec |
| `QGF`, `CGF` | `FiniteFieldEncoding.register_type(classical=...)` projects the declaration to a neutral descriptor. | `simulation/finite_field_encoding.py` |
| `QGFPoly`, `CGFPoly` | `FiniteFieldPolynomialEncoding.register_type(classical=...)` retains the coefficient-field descriptor. | `simulation/finite_field_polynomial_encoding.py` |

The scalar factories above live in `core/ir/program/register_types.py`.
Field factories retain optional galois objects inside the Simulation codecs.
Callers use the native owner imports. Executable Program ports are admitted
by the existing register verifier, independently of metadata construction.

The export audit verifies the frozen module bytes, the exact ordered
export list, and the identity of every imported definition. All 19 concrete
factories are exercised through that source package on concrete and
symbolic fixtures. Width totals, storage and symbolic behavior are compared
with the native descriptors. Zero-width signed and unsigned fixtures retain
classical identity through the storage parameter.

The concentrated regression reuses the original constructor, validation,
bit ordering, codec, and append evidence for the mapped families. It checks
that register metadata and numerical ownership agree without broadening
execution support. The initializer emits no physical operations and has no
resource model. The larger original `_infra/data_types.py` execution and
validation scope remains tracked by Q-0160.
