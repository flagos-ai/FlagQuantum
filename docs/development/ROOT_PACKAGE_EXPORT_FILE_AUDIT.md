# Complete root package exports

Q-0143 covers the complete root initializer of Qualtran v0.7.0 at revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, copyright Google LLC,
Apache-2.0, Git blob `a0f2b5e2d219176be9c7a0017968e6cb25c0cb98`.
Its executable content consists of 52 ordered re-exports from nine modules.
The source test hashes the actual imported file, checks every statement and
verifies every object's identity with its defining module, including private
`_ControlledBase`. The module docstring and tier comments describe ownership.

## Complete native ownership

Paths in the table are relative to `flagquantum/`. Each original name appears
once. These are responsibility mappings between different architectures.
Behavioral limits remain in the owning source-file audits.

| Original exports | Native responsibility and consumer evidence |
| --- | --- |
| Bloq, GateWithRegisters | Core `ir/program/model.py` FunctionRecord and ProgramRecord, typed signatures and existing `circuit.append(op)` component binding. `test_declared_component_contract.py` covers declaration-only calls. |
| DecomposeTypeError, DecomposeNotImplementedError, DidNotFlattenAnythingError | Compiler `registers/component_expansion.py` NoComponentExpansionError and expansion validation. `test_component_graph.py` and existing component expansion tests retain atomic/refused expansion behavior. |
| BloqError | Compiler `registers/component_bindings.py` RegisterBindingError and Core validation errors, selected at the owning failure boundary. |
| CompositeBloq, BloqBuilder | Core ProgramRecord and Compiler `registers/component_builder.py` ComponentProgramBuilder. Stored graph construction, copy and execution retain one IR. |
| SoquetT, ConnectionT, BloqInstance, Connection, DanglingT, LeftDangle, RightDangle, Soquet | Core `ir/program/model.py` Operation, ValueRef and Edge, plus `ir/program/register_bindings.py` RegisterBinding and typed function inputs/outputs. Compiler `registers/component_graph.py` owns traversal and copying. `test_component_graph.py` covers visitation and directional lifetimes. |
| QCDType, CDType, QDType | Core `ir/program/registers.py` RegisterType with separate quantum/classical widths. Logical validation and physical admission remain separate owners. |
| QAny, QBit, CBit | Core `ir/program/register_types.py` opaque_register_type, quantum_bit_type and classical_bit_type. |
| QInt, CInt, QIntOnesComp, CIntOnesComp | Same type owner, signed and ones-complement quantum/classical constructors. Logical register-type regression covers metadata. |
| QUInt, CUInt, BQUInt, BCUInt | Same type owner, unsigned and bounded unsigned quantum/classical constructors. |
| QFxp, CFxp | Same type owner, fixed_point_register_type and classical_fixed_point_register_type. Existing fixed-point suites and approved Proposals 113–115 distinguish logical descriptors and physical storage. |
| QMontgomeryUInt, CMontgomeryUInt | Same type owner, montgomery_register_type and classical_montgomery_register_type. Existing Montgomery type regression covers modulus metadata. |
| QGF, CGF, QGFPoly, CGFPoly | Core RegisterType field/polynomial descriptors and register verifier. `test_finite_field_register_types.py` covers the numerical-to-neutral projection. Approved Proposals 109–112 govern packed and classical physical consumers. |
| QDTypeCheckingSeverity, check_dtypes_consistent | Core `ir/program/register_compatibility.py` RegisterTypeCheckingSeverity and check_register_types_consistent. |
| Register, Signature, Side | Core `ir/program/registers.py` Register and RegisterSide, `ir/program/signatures.py` Signature. Existing declaration and logical type suites cover shapes, sides and symbolic metadata. |
| Adjoint | Compiler `registers/component_adjoint.py` adjoint_component_program, adjoint_component_calls and adjoint_component_hierarchy. Existing exact-adjoint regressions retain measurement cleanup. |
| Controlled, _ControlledBase | Compiler `registers/component_control.py` control_component_program and control_component_hierarchy with existing control-system owners. |
| CtrlSpec | Compiler `registers/control/control_predicate.py` ControlRegister and Core typed signatures. |
| AddControlledT | Compiler ControlAdder callable in the existing component-control boundary. |
| make_ctrl_system_with_correct_metabloq | Compiler `registers/control/component_control_selection.py` select_component_control_system and existing default/specialized control owners. |
| BloqExample, bloq_example, BloqDocSpec | `testing/component_examples.py` ComponentExample, component_example and ComponentDocumentation. `test_component_examples.py` covers factory and documentation consumers. |

The native root retains its existing lazy public exports and FlagQuantum
release identity. The source algorithm/analysis namespace imports terminate
at their existing native owners. Protected exports and their snapshots remain
governed by the approved API contract.

## Acceptance scope

This initializer has no gate body, work allocation, measurement branch or
resource formula. Literal primitive counts and source resource estimates are
therefore inapplicable to this file. Existing consumer suites supply amplitude,
work ownership and feedback evidence at their implementation boundaries.

Acceptance closes the complete re-export responsibility of Q-0143. It leaves
the underlying controlled, symbolic, typed admission and external adapter
source-file boundaries in their existing ledger entries. Pending Proposals
116–119 and the general reflection dirty-junk counterexample retain their
recorded status. Package build and documentation configuration have separate
parents Q-0942 and Q-0134. No site-build or wheel-build claim follows from this
initializer audit.

Batch results are retained in `ftqc-reference-env/g2-root-version-*.log`.

## Batch verification

The concentrated run passes 2107 checks and retains one pre-existing failure
in test_core_import_boundary.py. Seven NumPy imports in Core value_math.py
violate its dependency-policy test. The tested file, policy and failure test
are byte-identical to base cbf07e5fcbabc00ee0f36e5055b1fff79f5c090c.
The dedicated three root/version checks and 29 local integration checks pass.
Strict typing of both added Python files, Black, Ruff, team scope, architecture,
IR and public API checks pass. Logs and the unchanged-input blob certificate
are retained under ftqc-reference-env/g2-root-version-. The broader regression
retains that failure in its original log.
