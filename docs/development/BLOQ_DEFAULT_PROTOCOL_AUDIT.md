# Complete default component interface

This audit covers Q-0151, the complete `qualtran/_infra/bloq.py` from Qualtran
v0.7.0 revision `bcb17bb25360ed68ce3bc084645db3e929799389`, Git blob
`cef1f2f30bdc849048f50186616b4efce9877d8f`. The frozen inventory and the
installed source have the same blob. Copyright 2023 Google LLC, Apache-2.0.
The companion Q-0146 `_infra/__init__.py`, blob
`89a9ee1e20e4051fa8d2e46b3a531167e9e8beb7`, contains the license and a comment
about top-level reexports. It has no executable imports, assignments or
definitions. FlagQuantum keeps the established domain-owned exports.

## Ownership and interface

The source base class delegates to several domains. Its native responsibilities
are distributed among the existing typed Core records, Compiler transforms,
Simulation evaluators, Ecosystem exporters, Drawer and Testing metadata.
Functions retain their explicit stable names. Typed ports retain direction,
encoding and row-major indices. Words remain MSB-first. Builders own linear
handles, while the physical binder owns clean/borrowed workspace placement.
Numerical callbacks receive already-bound values and never transform the graph.

The user-facing execution path remains the existing
`Circuit.append(FieldOperation(width, bound.circuit.instructions))`, exercised
in `test_bloq_interface_file.py` and `test_register_component_infrastructure.py`.
The first test compares complete complex X/CNOT states with a spectator. The
infrastructure tutorial supplies the connected hierarchical program workflow.

## Every source definition

Line numbers refer to the frozen source. Each row names the authoritative
native owner and executable evidence. Test paths below are under `tests/unit/`.

| Source definition and lines | Native owner | Behavioral evidence |
| --- | --- | --- |
| `_decompose_from_build_composite_bloq`, 67–76 | `compiler/registers/component_builder.py` | `test_component_program_builder.py` verifies fixed signatures, named outputs and final ownership. Native callers pass a Mapping directly to finalize, while the source helper requires its Python author callback to return a dict. |
| `signature`, 142–165 | `core/ir/program/signatures.py`, `registers.py` | `test_logical_register_types.py`, `test_register_component_calls.py` verify typed, shaped and directional ports. An empty declaration is admitted separately from an executable body. |
| `build_composite_bloq`, 167–182 | `ComponentProgramBuilder` and stored `FunctionRecord` bodies | `test_component_program_builder.py` and `test_register_component_infrastructure.py` construct real connected bodies and preserve explicit return order. |
| `decompose_bloq`, 184–207 | `compiler/registers/component_expansion.py` | `test_component_expansion.py`, `test_bloq_interface_file.py` verify one-level/full expansion and atomic refusal. Missing expansion and invalid graph diagnostics retain their native types. |
| `as_composite_bloq`, 209–218 | `compiler/registers/component_tensor_expansion.py` | `test_bloq_interface_file.py`, `test_tensor_root_entry.py` verify one-call root wrapping, already-composite retention and actual bound execution. |
| `adjoint`, 220–254 | `compiler/registers/component_adjoint.py`, `component_hierarchy_adjoint.py`, `_component_trace/adjoint.py` | `test_declared_adjoint.py`, `test_component_adjoint.py`, `test_component_hierarchy_adjoint.py` verify default identities, specialized inverses, complex amplitudes and all supported Add histories. |
| `on_classical_vals`, 256–278 | `simulation/classical_action.py`, `component_classical_state.py` | `test_classical_action.py` compares the unchanged source dispatch with dict/NotImplemented/malformed results and callback errors. |
| `basis_state_phase`, 280–300 | Same Simulation owners and `classical_measurement.py` | `test_component_classical_state.py` verifies value-before-phase order, complex phases, measurement branches and phased full matrices. |
| `call_classically`, 302–324 | `simulation/classical_typed_values.py`, `component_classical_state.py` | `test_classical_tensor_crosscheck.py` compares actual source calls and declaration-ordered results for X/CNOT/Toffoli/Add. |
| `tensor_contract`, 326–360 | `simulation/tensor_network/register_tensor_binding.py`, `register_superoperator_binding.py` | `test_tensor_root_entry.py`, `test_component_tensor_expansion.py` compare full complex pure tensors, open-system tensors and appended physical execution. |
| `my_tensors`, 362–394 | `simulation/tensor_network/component_tensor_providers.py`, `component_tensor_nodes.py` | `test_component_tensor_providers.py`, `test_component_tensor_expansion.py` verify custom suppliers, missing providers, explicit replacement and default flatten precedence. |
| `build_call_graph`, 396–413 | `compiler/_component_trace/program_call_graph.py`, `call_graph.py` | `test_program_component_call_graph.py`, `test_call_graph_protocol_file.py` verify stored decomposition counts, explicit symbolic declarations and the shared allocator. |
| `my_static_costs`, 415–427 | `compiler/_component_trace/cost_query.py`, `qec_gate_cost.py` | `test_component_cost_query.py`, `test_qec_gate_cost.py` verify seed/static/recursive priority and independent metrics. |
| `call_graph`, 429–459 | `compiler/_component_trace/call_graph.py` | `test_call_graph_protocol_file.py` verifies full directed weights, generalizer sequences, leaf policies and depth bounds against real source graphs. |
| `bloq_counts`, 461–481 | `component_callee_counts` in the same owner | Same test verifies one-level set/dict accumulation, omissions and errors. `test_declared_adjoint.py` verifies specialized inverse merging. |
| `get_ctrl_system`, 483–519 | `compiler/registers/control/component_control_selection.py`, `specialized_component_control.py`, `component_predicate.py` | `test_register_controlled_component.py`, `test_register_control_predicate.py` verify predicate wiring, controls-first ordering, emitted operations, feedback and models. |
| `controlled`, 521–543 | The same control owners and `controlled_component.py` | `test_controlled_tensor.py` verifies positive/negative and multi-register control, complete tensors and all small measurement histories. |
| `t_complexity`, 545–553 | `compiler/_component_trace/qec_gate_cost.py`, `resource_generalization.py` | `test_qec_gate_cost.py` compares recursive costs and retained legacy shims with the source model. Model declarations remain distinct from emitted instructions. |
| `as_cirq_op`, 555–582 | `ecosystem/cirq/_register_gate.py` and register export | `tests/test_cirq_register_export.py` verifies typed ports, output leases, workspace, exact inverse and classical feedback. |
| `as_pl_op`, 584–601 | `ecosystem/pennylane` export/import | `test_bloq_interface_file.py` compares the full source FromBloq CNOT matrix with native export and exact import. The native exporter supplies the native PennyLane operation. |
| `on`, 603–635 | Cirq bound register gate and explicit wire layout | `tests/test_cirq_register_export.py` verifies flat placement, wrong-width refusal and full complex matrices. |
| `on_registers`, 637–657 | Cirq register layout/export | The same test verifies shaped names, control/target ordering and terminal measurement wiring. |
| `wire_symbol`, 659–694 | `drawer/register_score_data.py` | `test_bloq_default_protocols.py` directly invokes the unchanged default method for all three sides, scalar/vector/matrix indices, title lengths 4/10/11 and Unicode. Native score labels and symbol records agree. Existing score tests verify actual graph traversal. |
| `__str__`, 696–697 | Explicit FunctionRecord names and presentation labels | `test_bloq_interface_file.py`, `test_declared_adjoint.py` verify retained identity and labels. Native definitions use stable declared names, while source default labels use the Python class name. |
| `_class_name_in_pkg_`, 700–708 | `testing/component_examples.py`, `ComponentDocumentation` | `test_bloq_default_protocols.py` compares the unchanged source parent-package rule with inferred native documentation imports. Runtime identities remain with their typed records. |

The two source exception classes have no executable bodies beyond their
docstrings. Compiler keeps its existing `NoComponentExpansionError`,
`CapabilityError` and graph-validation diagnostics. The audit maps their atomic,
unavailable and invalid-input meanings without adding source exception aliases.

## Execution and resource evidence

Existing numerical witnesses are reused through their actual consumers. The
concentrated batch reruns the atomic root, complete infrastructure notebook,
adjoint, controls, tensor selection, phased classical state and SDK workflows.
This includes exact operands and feedback, full complex states with spectators,
all small measured Add/control histories, and promised-clean work returns.
The source's 70-cell infrastructure notebook and four original interface tests
were already executed unchanged for Q-0144/Q-0154. Their frozen blobs remain
unchanged. See `BLOQ_INTERFACE_TEST_AUDIT.md` and the recorded `g2-bloq-*` logs.

| Quantity | Emitted primitives | Declared model |
| --- | --- | --- |
| Default interface dispatch, metadata and score formatting | 0 | No physical model requested |
| Atomic TestAtom X witness | 1 X, 0 T, no work or measurement | 100 synthetic T in the original fixture |
| Atomic TestTwoBitOp CNOT witness | 1 CNOT, no work or measurement | Evaluated separately by the cost owner |
| Adjoint H/T and measured Add/control consumers | Actual instruction streams and branch checks retained in their tests | Independent QECGateCost and direct-call declarations |

## Recorded boundaries

- Q-0156 still owns nonlinear SDK parameter expressions and symbolic global
  shifts that cannot enter the protected scalar vocabulary. This audit verifies
  the base delegation paths through the admitted register adapters. Full
  arbitrary SDK object expressiveness remains an open integration requirement.
- Source empty-array output omission, raw CompositeBloq default-adjoint
  decomposition refusal and TestAtom inverse static-cost refusal retain their
  previous evidence in `REGISTER_ADJOINT_PROTOCOL_AUDIT.md`.
- Native callbacks and exporters have explicit function/provider boundaries.
  Source inheritance and Python method discovery are distributed to these
  owners. Source Python wrapper type identity is an explicit architectural
  adaptation, including the native PennyLane CNOT export.
- Existing default-suite failure in the public docstring-example registry and
  whole-repository dependency-policy findings remain recorded. Batch checks
  establish their own measured scope.

The complete file's default responsibilities are accounted for above. Q-0156,
the broader composite-program implementation, and the numerical resource
dashboard parents retain their separate open acceptance decisions.

## Local acceptance

The concentrated regression passes 1,834 cases with no skips and one existing
PennyLane deprecation warning. The SDK integration regression passes 96 cases
with two existing warnings. The new test file passes strict typing, Black and
Ruff. Team scope, architecture and internal IR contract checks pass. Dependency
policy records 689 global findings and zero on the changed test. Both source
blobs match the frozen inventory. Evidence logs use `g2-bloq-defaults-*`.

The initial direct IR-tool invocation imported the installed package instead
of this worktree and reported a vocabulary mismatch. Running through the
established reference wrapper selected this worktree and passed all 44
diagnostic codes, 15 reader rules and 44 refusal sites. Production code and
contracts have no changes in this batch.
