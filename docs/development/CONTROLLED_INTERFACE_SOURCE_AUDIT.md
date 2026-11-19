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
| `CtrlSpec` | 73–312 | Core register metadata plus Compiler activation records. Typed single-clause SDK transport is implemented. Full constructor/equality/hash/count auditing remains open. |
| `CtrlSpec.__attrs_post_init__` | 122–123 | Core register metadata and `simulation/classical_control_activation.py`; construction/shape/width and symbolic guards. Full symbolic typed execution is open. |
| `CtrlSpec.num_ctrl_reg` | 126–127 | Core register metadata and `simulation/classical_control_activation.py`; construction/shape/width and symbolic guards. Full symbolic typed execution is open. |
| `CtrlSpec.shapes` | 130–132 | Core register metadata and `simulation/classical_control_activation.py`; construction/shape/width and symbolic guards. Full symbolic typed execution is open. |
| `CtrlSpec.concrete_shapes` | 135–140 | Core register metadata and `simulation/classical_control_activation.py`; construction/shape/width and symbolic guards. Full symbolic typed execution is open. |
| `CtrlSpec.num_bits` | 143–145 | Core register metadata and `simulation/classical_control_activation.py`; construction/shape/width and symbolic guards. Full symbolic typed execution is open. |
| `CtrlSpec.num_qubits` | 148–152 | Core register metadata and `simulation/classical_control_activation.py`; construction/shape/width and symbolic guards. Full symbolic typed execution is open. |
| `CtrlSpec.num_cbits` | 155–157 | Core register metadata and `simulation/classical_control_activation.py`; construction/shape/width and symbolic guards. Full symbolic typed execution is open. |
| `CtrlSpec.is_symbolic` | 159–160 | Core register metadata and `simulation/classical_control_activation.py`; construction/shape/width and symbolic guards. Full symbolic typed execution is open. |
| `CtrlSpec.activation_function_dtypes` | 162–172 | Core register metadata and `simulation/classical_control_activation.py`; construction/shape/width and symbolic guards. Full symbolic typed execution is open. |
| `CtrlSpec.is_active` | 174–204 | Core register metadata and `simulation/classical_control_activation.py`; construction/shape/width and symbolic guards. Full symbolic typed execution is open. |
| `CtrlSpec.wire_symbol` | 206–220 | `drawer/control_symbols.py`; exact bit, word, signed metadata and unknown-value symbols. |
| `CtrlSpec.__cvs_tuple` | 223–231 | Immutable Core register and Compiler pattern tuples provide shape/type/value identity. Foreign CtrlSpec object equality is kept at the reference boundary. |
| `CtrlSpec.__eq__` | 233–241 | Immutable Core register and Compiler pattern tuples provide shape/type/value identity. Foreign CtrlSpec object equality is kept at the reference boundary. |
| `CtrlSpec.__hash__` | 243–244 | Immutable Core register and Compiler pattern tuples provide shape/type/value identity. Foreign CtrlSpec object equality is kept at the reference boundary. |
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
| `test_ctrl_spec` | 43–60 | Control-pattern, predicate and activation tests cover concrete cases. Concrete typed transport and admitted one-bit physical metadata are covered by `test_cirq_control_values.py` and `test_single_bit_control_metadata.py`. Full constructor/equality/hash/count and symbolic execution coverage remain open. |
| `test_ctrl_spec_classical` | 63–74 | `test_classical_control_activation.py`, `test_classical_control_patterns.py`, `test_classical_controlled_values.py`. |
| `test_ctrl_spec_shape` | 77–81 | Control-pattern, predicate and activation tests cover concrete cases. Concrete typed transport and admitted one-bit physical metadata are covered by `test_cirq_control_values.py` and `test_single_bit_control_metadata.py`. Full constructor/equality/hash/count and symbolic execution coverage remain open. |
| `test_ctrl_spec_to_cirq_cv_roundtrip` | 84–102 | SDK integration tests exercise bound exports. Typed single-clause conversion and the four original SDK diagrams are covered by `test_cirq_control_values.py` and `test_cirq_register_control.py`. Full constructor/metadata auditing remains open. |
| `test_ctrl_spec_single_bit_one` | 108–109 | Control-pattern, predicate and activation tests cover concrete cases. Concrete typed transport and admitted one-bit physical metadata are covered by `test_cirq_control_values.py` and `test_single_bit_control_metadata.py`. Full constructor/equality/hash/count and symbolic execution coverage remain open. |
| `test_ctrl_spec_single_bit_zero` | 115–116 | Control-pattern, predicate and activation tests cover concrete cases. Concrete typed transport and admitted one-bit physical metadata are covered by `test_cirq_control_values.py` and `test_single_bit_control_metadata.py`. Full constructor/equality/hash/count and symbolic execution coverage remain open. |
| `test_ctrl_spec_single_bit_raises` | 120–122 | Control-pattern, predicate and activation tests cover concrete cases. Concrete typed transport and admitted one-bit physical metadata are covered by `test_cirq_control_values.py` and `test_single_bit_control_metadata.py`. Full constructor/equality/hash/count and symbolic execution coverage remain open. |
| `test_ctrl_spec_symbolic_cvs` | 126–131 | Control-pattern, predicate and activation tests cover concrete cases. Concrete typed transport and admitted one-bit physical metadata are covered by `test_cirq_control_values.py` and `test_single_bit_control_metadata.py`. Full constructor/equality/hash/count and symbolic execution coverage remain open. |
| `test_ctrl_spec_symbolic_dtype` | 135–144 | Control-pattern, predicate and activation tests cover concrete cases. Concrete typed transport and admitted one-bit physical metadata are covered by `test_cirq_control_values.py` and `test_single_bit_control_metadata.py`. Full constructor/equality/hash/count and symbolic execution coverage remain open. |
| `test_ctrl_spec_symbolic_wire_symbol` | 147–150 | `test_drawer_control_symbols.py` compares original callbacks. The four original SDK diagrams are compared exactly in `test_cirq_register_control.py`. Symbolic physical admission remains open. |
| `_test_cirq_equivalence` | 153–161 | SDK integration tests exercise bound exports. Typed single-clause conversion and the four original SDK diagrams are covered by `test_cirq_control_values.py` and `test_cirq_register_control.py`. Full constructor/metadata auditing remains open. |
| `test_ctrl_bloq_as_cirq_op` | 164–203 | SDK integration tests exercise bound exports. Typed single-clause conversion and the four original SDK diagrams are covered by `test_cirq_control_values.py` and `test_cirq_register_control.py`. Full constructor/metadata auditing remains open. |
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
- One's-complement, bounded and Montgomery descriptors have flat physical
  emitters but remain outside Core typed physical admission. Generic symbolic
  control execution still requires its physical admission contract. The default
  selector now preserves the source mixed-storage refusal, including zero-width
  descriptors. It does not claim source mixed-control support.
- Direct exported SDK wrappers and original diagram protocols now have exact
  comparison evidence. Generic declared-child control counts are implemented
  with complete source graph edge comparison. Whole symbolic typed control
  metadata requires full audit. Default routing is now covered through existing
  owners and their actual returned callbacks. Generic classical stored-child
  control and additional physical admission remain open.

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
