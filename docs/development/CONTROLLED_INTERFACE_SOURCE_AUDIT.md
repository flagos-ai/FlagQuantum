# Complete controlled interface intake

This audit inventories Q-0158 `qualtran/_infra/controlled.py` and Q-0159
`controlled_test.py` at frozen Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`. Copyright 2023 Google LLC,
Apache-2.0. Source blobs are `6fabe760e793f1b39f825d76f80e92b6e5fa5c28` and
`0a909d79efa343dc78ae67ebab896e8dc1fd187c`. Both complete parent items remain
open. The following owner map records implemented behavior and actual gaps.

## Interface and ownership

Core `Register`, `RegisterType` and `Signature` retain type, shape and direction.
Compiler `ControlRegister` retains activation words and encoding. Array values
are row-major and words are MSB-first. Compiler constructs a single activation
predicate, propagates a positive control through existing component calls, then
erases the private predicate work. Linear handles and output order remain with
`ComponentProgramBuilder`. The caller supplies workspace to the physical binder.
Simulation owns value activation, classical callbacks and controlled tensors.
Drawer owns symbols. Ecosystem owns foreign SDK transport. The user appends the
bound operation through `circuit.append(op)`.

This batch admits two's-complement signed control registers to the existing
typed predicate path because Core and the flat emitter already support them.
The tests cover scalar, array and mixed unsigned/signed inputs, exact inverse
histories, full complex amplitudes and clean work. Descriptor/codec disagreement
still fails validation.

## Every source definition

All 47 class/function/method occurrences are listed. Class rows include their
member scope. Source-only object identity is distributed across immutable
native records and explicit transforms. An owner assignment alone establishes
no whole-file acceptance.

| Source definition | Lines | Native owner or outstanding boundary |
| --- | --- | --- |
| `_cvs_convert` | 60–69 | `simulation/classical_control_patterns.py` preserves conversion and iterator behavior. |
| `CtrlSpec` | 73–312 | Core register metadata plus Compiler activation records. Typed single-clause SDK transport is implemented. Metadata constructor/count/identity ownership is audited in `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`, including source mutation and symbolic-shape defects. |
| `CtrlSpec.__attrs_post_init__` | 122–123 | Existing Register/Signature metadata and Simulation pattern/activation owners. Full metadata mapping and source lifetime/shape defects are audited in `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`. Symbolic typed execution remains open. |
| `CtrlSpec.num_ctrl_reg` | 126–127 | Existing Register/Signature metadata and Simulation pattern/activation owners. Full metadata mapping and source lifetime/shape defects are audited in `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`. Symbolic typed execution remains open. |
| `CtrlSpec.shapes` | 130–132 | Existing Register/Signature metadata and Simulation pattern/activation owners. Full metadata mapping and source lifetime/shape defects are audited in `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`. Symbolic typed execution remains open. |
| `CtrlSpec.concrete_shapes` | 135–140 | Existing Register/Signature metadata and Simulation pattern/activation owners. Full metadata mapping and source lifetime/shape defects are audited in `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`. Symbolic typed execution remains open. |
| `CtrlSpec.num_bits` | 143–145 | Existing Register/Signature metadata and Simulation pattern/activation owners. Full metadata mapping and source lifetime/shape defects are audited in `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`. Symbolic typed execution remains open. |
| `CtrlSpec.num_qubits` | 148–152 | Existing Register/Signature metadata and Simulation pattern/activation owners. Full metadata mapping and source lifetime/shape defects are audited in `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`. Symbolic typed execution remains open. |
| `CtrlSpec.num_cbits` | 155–157 | Existing Register/Signature metadata and Simulation pattern/activation owners. Full metadata mapping and source lifetime/shape defects are audited in `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`. Symbolic typed execution remains open. |
| `CtrlSpec.is_symbolic` | 159–160 | Existing Register/Signature metadata and Simulation pattern/activation owners. Full metadata mapping and source lifetime/shape defects are audited in `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`. Symbolic typed execution remains open. |
| `CtrlSpec.activation_function_dtypes` | 162–172 | Existing Register/Signature metadata and Simulation pattern/activation owners. Full metadata mapping and source lifetime/shape defects are audited in `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`. Symbolic typed execution remains open. |
| `CtrlSpec.is_active` | 174–204 | Existing Register/Signature metadata and Simulation pattern/activation owners. Full metadata mapping and source lifetime/shape defects are audited in `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`. Symbolic typed execution remains open. |
| `CtrlSpec.wire_symbol` | 206–220 | `drawer/control_symbols.py`; exact bit, word, signed metadata and unknown-value symbols. |
| `CtrlSpec.__cvs_tuple` | 223–231 | Existing immutable Register/ControlRegister tuples provide shape/type/value identity, with SymbolicArrayShape for unknown values. Source live-array/cached-identity divergence is explicitly tested in `test_control_descriptor_metadata.py`. |
| `CtrlSpec.__eq__` | 233–241 | Existing immutable Register/ControlRegister tuples provide shape/type/value identity, with SymbolicArrayShape for unknown values. Source live-array/cached-identity divergence is explicitly tested in `test_control_descriptor_metadata.py`. |
| `CtrlSpec.__hash__` | 243–244 | Existing immutable Register/ControlRegister tuples provide shape/type/value identity, with SymbolicArrayShape for unknown values. Source live-array/cached-identity divergence is explicitly tested in `test_control_descriptor_metadata.py`. |
| `CtrlSpec.to_cirq_cv` | 246–262 | Ecosystem `_control_values.py` provides typed single-clause transport. The multi-register decoder repair and source defect are explicit in `REGISTER_CONTROL_VALUES_AUDIT.md`. |
| `CtrlSpec.from_cirq_cv` | 265–297 | Ecosystem `_control_values.py` provides typed single-clause transport. The multi-register decoder repair and source defect are explicit in `REGISTER_CONTROL_VALUES_AUDIT.md`. |
| `CtrlSpec.get_single_ctrl_val` | 299–312 | `compiler/registers/control/component_control_selection.py`; physical typed single-bit descriptors now retain admitted word types and singleton shapes, with the source matrix-shape defect recorded separately. |
| `AddControlledT` | 315–334 | `compiler/registers/component_control.py` ControlAdder and builder handle/output ordering. |
| `AddControlledT.__call__` | 332–334 | `compiler/registers/component_control.py` ControlAdder and builder handle/output ordering. |
| `_get_nice_ctrl_reg_names` | 337–356 | `compiler/registers/component_control.py::_control_names` shared naming and collision rules. |
| `_ControlledBase` | 359–580 | `compiler/registers/component_control.py`, control predicate/selection owners and typed builder. THRU checks, controls-first ordering and stored call propagation are tested. |
| `_ControlledBase.subbloq` | 367–368 | `compiler/registers/component_control.py`, control predicate/selection owners and typed builder. THRU checks, controls-first ordering and stored call propagation are tested. |
| `_ControlledBase.ctrl_spec` | 372–373 | `compiler/registers/component_control.py`, control predicate/selection owners and typed builder. THRU checks, controls-first ordering and stored call propagation are tested. |
| `_ControlledBase._thru_registers_only` | 376–377 | `compiler/registers/component_control.py`, control predicate/selection owners and typed builder. THRU checks, controls-first ordering and stored call propagation are tested. |
| `_ControlledBase._make_ctrl_system` | 380–401 | `compiler/registers/component_control.py`, control predicate/selection owners and typed builder. THRU checks, controls-first ordering and stored call propagation are tested. |
| `_ControlledBase.ctrl_reg_names` | 404–412 | `compiler/registers/component_control.py::_control_names` shared naming and collision rules. |
| `_ControlledBase.ctrl_regs` | 415–419 | `compiler/registers/component_control.py`, control predicate/selection owners and typed builder. THRU checks, controls-first ordering and stored call propagation are tested. |
| `_ControlledBase.signature` | 422–424 | `compiler/registers/component_control.py`, control predicate/selection owners and typed builder. THRU checks, controls-first ordering and stored call propagation are tested. |
| `_ControlledBase.on_classical_vals` | 426–445 | `simulation/classical_controlled_values.py`; unchanged-source callback/phase evidence. |
| `_ControlledBase.basis_state_phase` | 447–459 | `simulation/classical_controlled_values.py`; unchanged-source callback/phase evidence. |
| `_ControlledBase._tensor_data` | 461–484 | `simulation/tensor_network/controlled_tensor.py` and component tensor binding; full complex tensors and measurement histories. |
| `_ControlledBase._unitary_` | 486–510 | `simulation/tensor_network/controlled_tensor.py` and component tensor binding; full complex tensors and measurement histories. |
| `_ControlledBase.my_tensors` | 512–523 | `simulation/tensor_network/controlled_tensor.py` and component tensor binding; full complex tensors and measurement histories. |
| `_ControlledBase.wire_symbol` | 525–544 | `drawer/control_symbols.py` plus register score callbacks. Generic child/control symbols and all four original SDK diagrams are audited in `CONTROLLED_REGISTER_EXPORT_AUDIT.md`. |
| `_ControlledBase.__str__` | 546–549 | Explicit FunctionRecord names and Drawer titles; source class-name spelling remains presentation metadata. |
| `_ControlledBase.as_cirq_op` | 551–561 | `ecosystem/cirq` bound export exists. `ecosystem/cirq/_register_control.py` now directly controls an exported child using typed control transport. All four original SDK diagrams and source/imported primitive streams agree. |
| `_ControlledBase._circuit_diagram_info_` | 563–580 | `ecosystem/cirq` bound export exists. `ecosystem/cirq/_register_control.py` now directly controls an exported child using typed control transport. All four original SDK diagrams and source/imported primitive streams agree. |
| `Controlled` | 584–675 | `compiler/registers/component_control.py`, control predicate/selection owners and typed builder. THRU checks, controls-first ordering and stored call propagation are tested. |
| `Controlled.__attrs_post_init__` | 603–610 | `component_predicate.py` typed quantum descriptor validation. Physical Core excludes additional source types. |
| `Controlled.make_ctrl_system` | 613–621 | `compiler/registers/component_control.py`, control predicate/selection owners and typed builder. THRU checks, controls-first ordering and stored call propagation are tested. |
| `Controlled.decompose_bloq` | 623–624 | `compiler/registers/component_control.py`, control predicate/selection owners and typed builder. THRU checks, controls-first ordering and stored call propagation are tested. |
| `Controlled.build_composite_bloq` | 626–653 | `compiler/registers/component_control.py`, control predicate/selection owners and typed builder. THRU checks, controls-first ordering and stored call propagation are tested. |
| `Controlled.build_call_graph` | 655–672 | `compiler/_component_trace` controlled models and program call graphs. `_component_trace/controlled.py` now propagates arbitrary child declarations with specialized factories and the same allocator; complete source graph edges are tested. |
| `Controlled.adjoint` | 674–675 | `compiler/registers/control/component_adjoint_control.py` and exact hierarchy adjoint; branch/work regressions reused. |
| `make_ctrl_system_with_correct_metabloq` | 678–717 | `control/default_component_control.py` selects the existing total, predicate or feedback system and preserves its returned attachment callback. Concrete routing and actual consumers are audited in `DEFAULT_COMPONENT_CONTROL_AUDIT.md`. General classical child graphs and symbolic physical admission remain open. |

## Every original test and helper

The complete unchanged source test file passes 42 parametrized cases, including
its actual Jupyter tutorial. Four original collection/marker warnings remain.
Native regression reuses the tests below without treating source execution
alone as evidence of native capability.

| Original test/helper | Lines | Native evidence or gap |
| --- | --- | --- |
| `test_ctrl_spec` | 43–60 | Control-pattern, predicate and activation tests cover concrete cases. Concrete typed transport and admitted one-bit physical metadata are covered by `test_cirq_control_values.py` and `test_single_bit_control_metadata.py`. Metadata count/identity/lifetime comparisons are covered by `test_control_descriptor_metadata.py`. Symbolic physical execution remains open. |
| `test_ctrl_spec_classical` | 63–74 | `test_classical_control_activation.py`, `test_classical_control_patterns.py`, `test_classical_controlled_values.py`. |
| `test_ctrl_spec_shape` | 77–81 | Control-pattern, predicate and activation tests cover concrete cases. Concrete typed transport and admitted one-bit physical metadata are covered by `test_cirq_control_values.py` and `test_single_bit_control_metadata.py`. Metadata count/identity/lifetime comparisons are covered by `test_control_descriptor_metadata.py`. Symbolic physical execution remains open. |
| `test_ctrl_spec_to_cirq_cv_roundtrip` | 84–102 | SDK integration tests exercise bound exports. Typed single-clause conversion and the four original SDK diagrams are covered by `test_cirq_control_values.py` and `test_cirq_register_control.py`. Metadata auditing is recorded in `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`. Symbolic physical admission remains open. |
| `test_ctrl_spec_single_bit_one` | 108–109 | Control-pattern, predicate and activation tests cover concrete cases. Concrete typed transport and admitted one-bit physical metadata are covered by `test_cirq_control_values.py` and `test_single_bit_control_metadata.py`. Metadata count/identity/lifetime comparisons are covered by `test_control_descriptor_metadata.py`. Symbolic physical execution remains open. |
| `test_ctrl_spec_single_bit_zero` | 115–116 | Control-pattern, predicate and activation tests cover concrete cases. Concrete typed transport and admitted one-bit physical metadata are covered by `test_cirq_control_values.py` and `test_single_bit_control_metadata.py`. Metadata count/identity/lifetime comparisons are covered by `test_control_descriptor_metadata.py`. Symbolic physical execution remains open. |
| `test_ctrl_spec_single_bit_raises` | 120–122 | Control-pattern, predicate and activation tests cover concrete cases. Concrete typed transport and admitted one-bit physical metadata are covered by `test_cirq_control_values.py` and `test_single_bit_control_metadata.py`. Metadata count/identity/lifetime comparisons are covered by `test_control_descriptor_metadata.py`. Symbolic physical execution remains open. |
| `test_ctrl_spec_symbolic_cvs` | 126–131 | Control-pattern, predicate and activation tests cover concrete cases. Concrete typed transport and admitted one-bit physical metadata are covered by `test_cirq_control_values.py` and `test_single_bit_control_metadata.py`. Metadata count/identity/lifetime comparisons are covered by `test_control_descriptor_metadata.py`. Symbolic physical execution remains open. |
| `test_ctrl_spec_symbolic_dtype` | 135–144 | Control-pattern, predicate and activation tests cover concrete cases. Concrete typed transport and admitted one-bit physical metadata are covered by `test_cirq_control_values.py` and `test_single_bit_control_metadata.py`. Metadata count/identity/lifetime comparisons are covered by `test_control_descriptor_metadata.py`. Symbolic physical execution remains open. |
| `test_ctrl_spec_symbolic_wire_symbol` | 147–150 | `test_drawer_control_symbols.py` compares original callbacks. The four original SDK diagrams are compared exactly in `test_cirq_register_control.py`. Symbolic physical admission remains open. |
| `_test_cirq_equivalence` | 153–161 | SDK integration tests exercise bound exports. Typed single-clause conversion and the four original SDK diagrams are covered by `test_cirq_control_values.py` and `test_cirq_register_control.py`. Metadata auditing is recorded in `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`. Symbolic physical admission remains open. |
| `test_ctrl_bloq_as_cirq_op` | 164–203 | SDK integration tests exercise bound exports. Typed single-clause conversion and the four original SDK diagrams are covered by `test_cirq_control_values.py` and `test_cirq_register_control.py`. Metadata auditing is recorded in `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`. Symbolic physical admission remains open. |
| `test_ctrl_spec_activation_1` | 206–210 | `test_classical_control_activation.py`, `test_classical_control_patterns.py`, `test_classical_controlled_values.py`. |
| `test_ctrl_spec_activation_2` | 213–222 | `test_classical_control_activation.py`, `test_classical_control_patterns.py`, `test_classical_controlled_values.py`. |
| `test_ctrl_spec_activation_3` | 225–228 | `test_classical_control_activation.py`, `test_classical_control_patterns.py`, `test_classical_controlled_values.py`. |
| `test_ctrl_spec_activation_4` | 231–234 | `test_classical_control_activation.py`, `test_classical_control_patterns.py`, `test_classical_controlled_values.py`. |
| `test_controlled_serial` | 237–260 | `test_component_structured_control.py`, hierarchy/control/extension tests and tutorial compare stored calls, emitted primitives and histories. |
| `test_controlled_parallel` | 263–298 | `test_component_structured_control.py`, hierarchy/control/extension tests and tutorial compare stored calls, emitted primitives and histories. |
| `test_doubly_controlled` | 301–313 | `test_component_structured_control.py`, hierarchy/control/extension tests and tutorial compare stored calls, emitted primitives and histories. |
| `test_bit_vector_ctrl` | 316–333 | `test_component_structured_control.py`, hierarchy/control/extension tests and tutorial compare stored calls, emitted primitives and histories. |
| `test_classical_sim_simple` | 336–343 | `test_classical_control_activation.py`, `test_classical_control_patterns.py`, `test_classical_controlled_values.py`. |
| `test_classical_sim_array` | 346–357 | `test_classical_control_activation.py`, `test_classical_control_patterns.py`, `test_classical_controlled_values.py`. |
| `test_classical_sim_int` | 360–367 | `test_classical_control_activation.py`, `test_classical_control_patterns.py`, `test_classical_controlled_values.py`. |
| `test_classical_sim_int_arr` | 370–380 | `test_classical_control_activation.py`, `test_classical_control_patterns.py`, `test_classical_controlled_values.py`. |
| `test_classical_sim_int_multi_reg` | 383–391 | `test_classical_control_activation.py`, `test_classical_control_patterns.py`, `test_classical_controlled_values.py`. |
| `test_notebook` | 395–396 | `test_register_control_tutorial.py` executes all native cells and compares full physical cases. |
| `_verify_ctrl_tensor_for_unitary` | 399–404 | `test_controlled_tensor.py`, `test_register_control_tutorial.py` and component tensor integration preserve complete complex states. |
| `test_controlled_tensor_for_unitary` | 417–424 | `test_controlled_tensor.py`, `test_register_control_tutorial.py` and component tensor integration preserve complete complex states. |
| `test_controlled_tensor_without_decompose` | 427–437 | `test_controlled_tensor.py`, `test_register_control_tutorial.py` and component tensor integration preserve complete complex states. |
| `test_controlled_global_phase_tensor` | 440–443 | `test_controlled_tensor.py`, `test_register_control_tutorial.py` and component tensor integration preserve complete complex states. |
| `test_controlled_diagrams` | 446–492 | `test_drawer_control_symbols.py` compares original callbacks. The four original SDK diagrams are compared exactly in `test_cirq_register_control.py`. Symbolic physical admission remains open. |

## Concrete remaining gaps

- Typed single-clause control values now have an Ecosystem to/from Cirq transport
  retaining register metadata and supported numerical encodings. See
  `REGISTER_CONTROL_VALUES_AUDIT.md`. Direct exported wrappers are audited in
  `CONTROLLED_REGISTER_EXPORT_AUDIT.md`.
- Frozen `CtrlSpec.from_cirq_cv` initializes `idx = 0` and never advances it
  between registers. Two scalar QBit clauses `(0, 1)` decode to `(0, 0)` and
  reencode as `(0, 0)`. `g2-controlled-source-roundtrip-boundary.json` records
  the untouched source result. Native decoding advances each slice, verified for all 64 mixed unsigned/signed cases. This repair is explicitly distinguished from parity with the defective source.
- Typed physical one-bit controls now retain QBit, QAny(1), QUInt(1) and
  QInt(1) with scalar/vector/matrix singleton shapes. The source matrix-shape
  decomposition defect and zero-gate native split/join adaptation are recorded
  in `SINGLE_BIT_CONTROL_METADATA_AUDIT.md`.
- One's-complement, bounded and Montgomery descriptors now cross Core typed
  physical admission and preserve their canonical metadata through split/join
  and stored calls. Complete predicate and wrapper evidence is recorded in
  `CONTROLLED_VIA_AND_SOURCE_AUDIT.md` and
  `test_controlled_integer_storage.py`. Unknown binary activation declarations,
  concrete specialization and the source HasLength rule are accepted in the
  same audit. Unresolved symbolic physical widths remain outside execution.
  The default
  selector now preserves the source mixed-storage refusal, including zero-width
  descriptors. It does not claim source mixed-control support.
- Direct exported SDK wrappers and original diagram protocols now have exact
  comparison evidence. Generic declared-child control counts are implemented
  with complete source graph edge comparison. Whole symbolic typed control
  metadata ownership and source lifetime defects are now audited in
  `CONTROL_DESCRIPTOR_METADATA_AUDIT.md`. Symbolic physical execution remains open. Default routing is now covered through existing
  owners and their actual returned callbacks. Stored-child CBit controls now
  compose existing conditions and clean work through native calls, audited in
  `CLASSICAL_COMPONENT_CONTROL_AUDIT.md`. Wider classical-word reads and other
  physical admission remain open.

### Current direct dependency review — 2026-11-20

Q-0156 is accepted after the full nonlinear scalar, global-phase and generic
EigenGate shift dependency closure at 275ea4611. Reuse its published primitive,
complex-amplitude, feedback, work-zero and SDK evidence. The controlled source
parents remain open while their remaining execution dependencies are resolved.

The classical-word intake proves the exact next boundary before any execution
change. Existing Simulation callbacks agree with unchanged source callbacks for
unsigned, signed, ones'-complement, Montgomery, bounded and fixed-point storage.
An identity Program with each wider classical dtype fails current Core port
admission. The source classical wrapper supplies no physical decomposition or
QEC model. A native physical implementation therefore needs its numerical or
tensor reference and the already verified child stream, with source wrapper
model refusal reported independently.

Proposal 108 records the required typed-port and read-result contract extension
in `API_CHANGE_PROPOSAL_108_CLASSICAL_WORD_FEEDBACK.md`. The user approved it
on 2026-11-20, and the bounded integer-word scope is implemented in Core and
Compiler. `TYPED_CLASSICAL_WORD_FEEDBACK_AUDIT.md` supersedes the intake refusal
for that scope and records the source ones'-complement tensor counterexample.
Finite-field classical admission, unresolved symbolic physical values and generic
external-consumer boundaries remain open. Proposal 107 is separately published.
Source-parent acceptance counts remain unchanged.

The source notebook's signed QInt(8) value 255 draws but fails physical encoding.
Its prose also predates the direct tensor and Cirq controlled-operation methods
present in the frozen source. The native tutorial identifies all three facts.
No reference file was modified.

## Batch evidence

All 3344 concentrated cases pass with one existing source deprecation warning.
Strict typing passes seven files including flattened notebook cells. Black,
Ruff, Compiler team scope, architecture and multi-level IR checks pass. Native
control tutorial evidence is detailed in `REGISTER_CONTROL_TUTORIAL_AUDIT.md`.
The original 42-case result and source blob inventory are persistent under
`ftqc-reference-env/g2-controlled-file-*`. Resource observations and model
estimates are separately recorded in `g2-control-tutorial-resources.json`.

The full dependency scan reports 688 existing findings. This batch has zero
changed-path findings after guarding the optional NumPy test dependency.

The default-routing dependency batch is audited in
`DEFAULT_COMPONENT_CONTROL_AUDIT.md`. Its observations supersede stale selection
and SDK-gap wording in earlier intake evidence. Q-0158 and Q-0159 remain open.

The complete metadata dependency review reuses native records without new
production types. See `CONTROL_DESCRIPTOR_METADATA_AUDIT.md` for each source
member's owner, exact array mutation/cache discrepancy and symbolic shape leak.
Parent acceptance remains open for the execution dependencies listed there.

### Binary field dependency review (2026-11-20)

`49792efd6` and `1e74de508` close the quantum single-bit canonical field/poly
control dependency. Wider fields are rejected by original CtrlSpecAnd. The
preliminary wider-field intake was corrected with explicit source witnesses.
Two native zero-gate type views are disclosed in the comparable call traces.

A recursive AST inventory counts 49 definitions, comprising the 47 previously
listed outer occurrences plus nested `CtrlSpec.__cvs_tuple._serialize` and
`_ControlledBase._make_ctrl_system.add_controlled`. The former is covered by
existing identity/cache evidence and the latter by returned callback evidence.
This clarifies inventory counting without adding parallel production owners.

The next concrete execution gap is classical binary field/poly Program
admission. Six source/native callback witnesses agree for active and inactive
values over GF(2), GF(4), GF(8) and degree-one polynomials. Native identity
validation and feedback reads reject those descriptors. Proposal 110 records
this separately protected extension and awaits approval. Original classical
wrapper decomposition/model refusals remain explicit. The evidence is
`g2-classical-field-intake.json`; these findings establish a remaining gap and
supply no new parent acceptance.

Proposal 110 was explicitly approved on 2026-11-20. The canonical classical
binary field/poly execution dependency described in the preceding intake is
implemented through the existing validators and feedback owner. Its accepted
scope and remaining boundaries are in `CLASSICAL_FIELD_FEEDBACK_AUDIT.md`.

## Proposal 111 dependency acceptance

The approved nonbinary classical field/poly dependency now joins the accepted
integer and binary paths. The native builder follows unchanged numerical
callbacks and bit codecs for all 902 small legal activation/input pairs.
`NONBINARY_CLASSICAL_FIELD_FEEDBACK_AUDIT.md` retains the original four polynomial
tensor-index counterexamples, wrapper decomposition/model refusals, scalar and
shaped metadata transport, complete measured-child histories and exact binary
compatibility artifacts. Earlier statements that all nonbinary classical
values are refused are historical. Nonbinary quantum execution and unresolved
symbolic or generic interface dependencies retain their own scopes.

## Current concrete physical dependency after Proposal 112

Proposal 112 closes owned conditional classical updates and Q-0547 at
`3d85c62d8`. Its source-only decomposition/model absence and numerical tensor
counterexamples remain independently recorded. Earlier all-classical-read and
all-classical-write missing-capability statements are superseded by accepted
Proposals 108 through 112.

The next real physical gap is unsigned one-bit QFxp storage. For fractional
width 0 and 1 and both activation bits, unchanged ControlledViaAnd emits CX
or X/CX/X, while the native typed physical validator rejects the descriptor.
For fraction 1, source numerical 0.5 corresponds to native stored bit 1.
`g2-fixed-point-control-intake.json` retains this four-case distinction. Signed
one-bit construction and wider QFxp predicates are original refusals.
Proposal 103 prerequisite 1.20 explicitly protected physical fixed-point
refusal, so Proposal 113 records the narrow required extension before coding.
Proposal 113 was explicitly approved on 2026-11-20 and is implemented in
Core `1c87aac95` and Compiler `f586d6276`. The approved concrete storage
dependency is accepted in `FIXED_POINT_BIT_CONTROL_AUDIT.md`. It supplies
no parent acceptance.

## Negative fractional physical dependency after Proposal 114

The final concrete intake also exercised negative integer fractional widths.
Original QFxp(1,-1/-2/-7) with either stored activation bit decomposes to CX or
X/CX/X. Proposal 113 intentionally admits only fractions 0 and 1, so native
physical admission initially refused these six source-supported cases. The historical
streams are in `g2-fixed-point-negative-fraction-intake.json`. Proposal 114 was
explicitly approved on 2026-11-20 and now admits canonical unsigned one-bit
storage at every built-in integer fractional width at most one. The six exact
source/native streams, complete complex states, measurement histories and
work cleanup pass. See `FIXED_POINT_BIT_CONTROL_AUDIT.md` and
`g2-negative-fraction-control-` receipts. Q-0158/Q-0159 stay open for the other
foreign interfaces and source limitations recorded in this audit.

Unresolved symbolic fractional widths have a different source behavior.
Original CtrlSpec.get_single_ctrl_val raises for symbolic metadata. The
new explicit negative witness verifies both extraction and decomposition
refusals. Its physical rejection supplies no missing source-executable case.
Earlier broad symbolic-physical gap labels need this source distinction.
The complete unchanged controlled test suite now passes 42 cases, including
its actual notebook, with three collection warnings. Native evidence is
recorded separately in the Proposal 113 audit.

## Complete file closure after Proposals 108 through 115

The final review covers all 49 recursive definitions, including the two nested
callbacks, and every original test/helper listed above. Earlier open-dependency
statements describe their intake dates. The following disposition supersedes
those statements for Q-0158 and Q-0159.

| Former dependency | Final owner and executable evidence |
| --- | --- |
| Constructor, identity, count and symbolic metadata | Existing Register/Signature, normalized activation patterns and immutable ControlRegister snapshots. `test_control_descriptor_metadata.py` preserves the array-cache and concrete-shape source defects. |
| Total and predicate child propagation | Existing `component_control.py`, predicate, exact adjoint and specialized control adders. Stored serial/parallel, nested children and actual returned attachment callbacks retain source call/operand/feedback comparisons. |
| Classical stored children and wide reads/writes | Existing classical control/feedback builders and approved Proposals 108, 110, 111 and 112. Scalar/shaped integer and field controls retain complete numerical/tensor comparisons, occupied classical snapshots and measured child histories. |
| Packed quantum fields and fixed-point storage | Approved Proposals 109 and 113 through 115. Single-bit predicate execution retains both polarities, source leaf streams, complex branches and clean work. Wider fixed-point storage supports its approved native arithmetic owners. Wider field/fixed-point predicate refusals match original CtrlSpecAnd. |
| Generic SDK child operation and diagrams | Existing `control_register_operation` preserves the child operation, SDK protocols, port aliases and quantum control clause. The final foreign test exercises every stored value of five unsigned fixed-point formats and all legal values of GF(2), GF(4), GF(3), GF(5), GF(9) and their degree-one polynomials. |
| Symbolic foreign conversion | Original CtrlSpec.to_cirq_cv rejects unresolved widths or activations. Native symbolic metadata/models retain existing owners, with explicit concrete specialization before physical export. The final regression checks the matching refusal. |
| Missing classical wrapper decomposition/model | Original ClassicallyControlled has neither. Native stored feedback is independently specified and checked against the original callbacks/tensors. Conditional measurement/reset/channel graph admission remains its documented execution boundary. |
| Arbitrary foreign child subclasses | SDK operations or explicit native component definitions enter their existing adapters. Native providers implement existing control adders and typed ports. No implicit translation of unknown Python class identity is required by these source methods. Original missing child decomposition failures and native provider-cycle rejection remain explicit. |

The final SDK review found and fixed a real import defect. Eigen-phase
normalization converted a three-control XPow into canonical controlled Rx. The
SDK decomposed Rx into XPow plus phase, which normalization converted back into
the same controlled Rx. Native cycle detection then rejected a valid operation.
For more than two controls, the adapter now retains the original SDK
operation before decomposition. Existing one/two-control symbolic lowering and
true cyclic-provider rejection retain their existing regression evidence.

`test_controlled_foreign_file.py` checks the direct SDK gate/operands through
source operation export, exact diagrams, complete source unitaries, coherent
complex states entangled with a spectator, and normal `circuit.append(op)`.
Three/four controls cover X/Y/Z powers, mixed polarities and three global-shift
values. No work or measurement is introduced by these SDK witnesses. This
protocol has a broader domain than the original typed predicate decomposition,
so its successful wider controls do not change native typed predicate admission.
The original single-bit predicate, exact inverse, all applicable measurement
histories and clean-work certificates remain accepted and are reused.

Actual bound primitive counts and the original wrapper QEC estimates are recorded
separately in `g2-controlled-foreign-evidence.json`. SDK rotation synthesis is
identified as SDK synthesis. It does not establish gate-count equivalence with
the separate predicate recipe or with a declared QEC model. Original cursor,
mutable-array cache, singleton-matrix, signed-tutorial and nonbinary polynomial
tensor defects retain their existing witnesses and remain unchanged upstream.

The affected-path regression passes 1,210 cases with two existing warnings.
Twelve optional Qiskit cases are skipped because Qiskit is absent, explicitly
outside this Cirq batch. All 43 final foreign cases execute. Local construction
and artifact integration passes 17 cases. Strict typing, Black, Ruff, Ecosystem
scope, architecture, multi-level IR and public API checks pass. Six immutable
source blobs, Qualtran 0.7.0 and Kickmix 0.1 Simulator are verified. Nine accepted
owners/tests remain byte-identical to the batch base. The complete original
42-case result, including its notebook and three collection warnings, is reused
after frozen blob verification. The established default docstring-policy failure
is separately retained after verifying its ten files against the batch base.

The Cirq 1.6.1 compatibility lane passes 100 cases, including all 43 final
foreign cases. Its 1,446 SDK/NumPy shape deprecation warnings are retained.
The dependency scan retains 687 existing findings and zero changed-path findings.
Three positive controls emit 5 phase, 15 single-qubit matrix atoms, 12 CX and
2 CCX instructions, with zero workspace and measurements. The separate source
predicate model declares 2 And, 2 Clifford, 1 rotation and 2 measurements.
Every emitted `any` atom in this witness has exactly one operand. Matrix atoms
retain SDK-synthesized one-qubit leaves and have their own count.

The existing native example can be used through the normal append workflow.

```python
from examples.algorithms.controlled_programs import controlled_serial_program
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program

bound = bind_register_program(controlled_serial_program(), ((0,), (1,)))
circuit = Circuit(2)
circuit.append(FieldOperation(2, bound.circuit.instructions))
```

Parent acceptance closes the complete controlled source and test files, with
the source defects and explicit execution boundaries above retained as their
disposition.
