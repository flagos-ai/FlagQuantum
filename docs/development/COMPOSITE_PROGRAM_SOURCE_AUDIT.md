# Composite Program source-file audit

This inventory covers frozen Qualtran v0.7.0
`qualtran/_infra/composite_bloq.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Q-0156 remains open. Having a native
owner below does not establish full source-file acceptance. The outstanding
boundaries after each group identify the next verification or implementation.

The immutable Core `ProgramRecord`, `FunctionRecord`, `Operation` and `ValueRef`
are the graph authority. Compiler owns construction and transformations,
Simulation owns numerical traversal, Ecosystem owns SDK conversion, and Drawer
owns presentation. Source Python objects are confined to reference tests.

## Complete source definition inventory

| Source definitions | Native owner and evidence | Remaining boundary |
| --- | --- | --- |
| `_to_tuple`, `_to_set`, `CompositeBloq` | Core immutable tuples and typed Program validation replace attrs converter storage. | Symbolic and non-executable typed graph admission is still Q-0149. Source object equality is represented through native records, not source Python classes. |
| `CompositeBloq._default_bloq_instances`, `all_soquets`, `_binst_graph` | Existing operations/results/arguments and `component_graph` / `component_schedule` derive nodes, values and dependencies. `test_component_graph.py` checks metadata, array shape, ownership and copy. | A portable ordering for source frozenset-only isolated nodes is unavailable. Native stored order is explicit. |
| `CompositeBloq.as_cirq_op`, `to_cirq_circuit_and_quregs`, `to_cirq_circuit` | `_register_program_export.export_bound_register_cirq` composes Compiler-bound Program traversal with external manager lifetime replay and returns final shaped Qids. Source/native tests cover clean/borrowed reuse, LEFT release and RIGHT transfer, reordered ports, primitive order and every repeated measured-child branch. See CIRQ_REGISTER_PROGRAM_EXPORT.md. | Finite executable Programs use consecutive input binding explicitly. Arbitrary SDK-backed leaf wrappers and symbolic graphs remain open. The gate wrapper and direct returned-output conversion have separate ownership semantics. |
| `CompositeBloq.from_cirq_circuit` | `_register_import.import_cirq_register_optree` builds native typed Programs. `test_cirq_register_import.py` checks default sorted qubits, directional/shaped ports, phase, resets, custom definitions and six-wire full unitary. | Source permits arbitrary Cirq wrappers; native admission is finite and explicit. Audit unsupported operation classes before full conversion closure. |
| `CompositeBloq.on_classical_vals` | `component_classical_state.ComponentClassicalState` and pure/phased component traversal compose registered numerical actions. Tests cover shaped values, phase, RNG, failures and all small measurement branches. | Symbolic typed graph admission and arbitrary source Python callbacks remain outside finite Program construction. |
| `CompositeBloq.as_composite_bloq`, `decompose_bloq` | A native stored Program already is the composite. Root wrapping and decomposition use `component_tensor_expansion` and `component_expansion`; no redundant wrapper is required for an existing composite. | Source refusal to decompose an already-composite value must stay distinct from selective child expansion. Existing root tensor tests document the distinction. |
| `CompositeBloq.build_call_graph` | `_component_trace.program_call_graph.program_component_call_graph`, backed by the shared symbolic call graph. `test_program_component_call_graph.py` covers hierarchy, direct counts, specialized adjoints, depth and keep policies. | Stored component multiplicities and source models remain separate from measured emitted instructions. |
| `CompositeBloq.iter_bloqnections`, `iter_bloqsoqs`, `final_soqs` | `component_graph.iter_component_calls`, SSA endpoints and the final return tuple. Graph and construction notebook tests cover every immediate connection and final permutation. | Native has no public connection-object class, intentionally using Core SSA. Empty-shaped source final output omission remains a recorded source defect. |
| `CompositeBloq.copy` | `component_graph.copy_component_program`; graph tests retain definitions, metadata, arrays and visitation ordering. | Native retains declared empty outputs, where source copy fails. |
| `CompositeBloq.flatten_once`, `flatten` | `component_expansion.expand_component_calls_once` and `expand_component_calls`; selective expansion and notebook tests cover callback decisions, depth, retained/gapped IDs, atomic refusal and empty bodies. | Arbitrary Python decomposition callbacks are not admitted as typed Program bodies. |
| `CompositeBloq.adjoint` | `component_adjoint` and specialized hierarchy adjoints, with physical and all-branch reference tests. | Adjoint selection for a source leaf must resolve to a declared native provider. Reversing measurement instruction order is not a valid substitute. |
| `CompositeBloq._debug_binst`, `debug_text` | `drawer.component_debug_text.format_component_debug_text`, commit `3dde10db5`. Complete text parity covers adjacency grouping, generations, multidimensional ports, empty/zero-width graphs and expansion labels. | Source names and angle-bracket instance notation map to native names and `name#instance`. Source isolated-node ties use native stored order. |
| `CompositeBloq.__str__` | The native record representation and explicit Drawer diagnostic serve separate storage and display uses. | Exact source class-name summary formatting has not been introduced. Decide the native summary entry together with full diagnostic acceptance; do not equate dataclass repr with source summary. |
| `_create_binst_graph`, `_binst_to_cxns` | `component_schedule._function_connections`, existing SSA traversal, Drawer score generations and adjacency text. | Native graph consumers derive edges; no cached mutable NetworkX graph enters Core. |
| `_get_soquet`, `_cxns_to_soq_dict`, `_cxns_to_cxn_dict`, `_get_dangling_soquets`, `_flatten_soquet_collection`, `_get_flat_dangling_soqs` | `component_graph._collections`, `_defined_bindings`, `_elements`, `create_register_bindings`, SSA argument/result/return identities. Existing graph/bindings/notebook parity establishes row-major and directional endpoints. | Non-executable type/shape admission follows Q-0149; no separate graph identity representation is planned. |
| `BloqError`, `DidNotFlattenAnythingError` | `RegisterBindingError` and `NoComponentExpansionError`, both native value errors with existing failure tests. | Native exception names are explicit, source classes do not leak through the public boundary. |
| `_IgnoreAvailable`, `_IgnoreAvailable.add` | Existing binding constructor can omit availability tracking, while live builder owns its availability set. | A source private helper class is replaced by the native callable boundary. |
| `_reg_to_soq`, `_process_soquets` | `component_bindings.create_register_bindings`, `process_register_bindings`; source-backed contract tests cover shape, order, prefix mutations and width compatibility. | Wider descriptor admission remains tracked separately. |
| `_map_soqs`, `_map_flat_soqs`, `_update_flat_soq_map` | `component_binding_map.map_register_bindings`, copy/expansion SSA replacement maps. `test_component_inline_calls.py` covers scalar/shaped replacement, duplicate precedence, failures and inlining. | The public collection operation is covered; private flat-map mutation is expressed with native SSA dictionaries and must be assessed through copy/expansion consumers. |
| `BloqBuilder`, `__init__`, `add_register_from_dtype`, all `add_register` overloads, `from_signature` | `ComponentProgramBuilder` accepts a Signature and supports opt-in `allow_register_addition`; `component_signature.infer_output_registers`. Commit `b6a6705b5`, mutable-signature tests and audit. | Negative width source metadata is rejected at the native finite execution boundary. Symbolic and fixed-point typed Programs remain open. |
| `BloqBuilder.map_soqs` | `map_register_bindings` is independently owned and callable, shared by inlining. | The operation is a module function; no builder convenience duplicate is needed. |
| `BloqBuilder._new_binst_i`, `_add_cxn`, `_add_binst` | Builder operation positions, Core SSA operands/results, `_new_value`, `_consume` and ordinary call insertion own identity and consumption. | Source private counters are not native contracts. Preserved display labels remain separate metadata returned by expansion. |
| `BloqBuilder.add_t`, `add_d`, `add` | Native `call_tuple`, `call_dict`, `call` for known definitions and `call_program` for imported composite bodies. Builder tests verify ordered outputs, linear consumption and failure prefixes. | Executable source leaves require native definitions/providers, not a live foreign object in the Program. |
| `BloqBuilder.add_and_partition`, `add_from` | Native `call_partitioned` delegates to `layout.component_auto_partition`; `inline_calls` uses shared import preparation. Commit `f362fb62d`, complete Program-import/inline regressions. | Declaration-only entries are importable but physical execution rejects a missing body. |
| `BloqBuilder.finalize`, `_finalize_strict` | Native `finalize`, signature inference and Core verifier. Default strict outputs and opt-in inferred RIGHT outputs have source failure-prefix tests. | Source malformed descriptor admission is not a reason to weaken native execution validation. |
| `BloqBuilder.allocate`, `free`, `split`, `join` | Existing shared workspace and rebinding component owners plus physical binder. Tests cover clean/borrowed/retired work, partial release, output ownership, logical zero widths and native source gate parity. | Source external qubit-manager composition belongs to the SDK facade audit above. |

## Next complete workflow

The direct finite Program export is now implemented and covered by the connected
source/native workflow described in CIRQ_REGISTER_PROGRAM_EXPORT.md. Reuse that
evidence when evaluating the three Cirq methods. The source `as_cirq_op` includes
a wrapper fallback for source components whose direct Cirq conversion refuses;
its arbitrary SDK-backed leaves and symbolic declaration semantics still need
explicit disposition. The source summary method also needs a native presentation
owner and original-source behavioral comparison.

The frozen source file's own tests currently pass 39 cases and its original
notebook test passes separately. Those runs confirm the reference environment.
They do not close untested native replacements. Q-0157 and Q-0149 remain open
independently, as do the dashboard numerical and PennyLane declaration gaps.
