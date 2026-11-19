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
| `CompositeBloq.__str__` | The native record representation and explicit Drawer diagnostic serve separate storage and display uses. | Drawer format_component_summary reports the immediate call-instance count with native ComponentProgram/components spelling. Nested shared definitions remain single instances at the selected layer; source text is compared after only this naming substitution. |
| `_create_binst_graph`, `_binst_to_cxns` | `component_schedule._function_connections`, existing SSA traversal, Drawer score generations and adjacency text. | Native graph consumers derive edges; no cached mutable NetworkX graph enters Core. |
| `_get_soquet`, `_cxns_to_soq_dict`, `_cxns_to_cxn_dict`, `_get_dangling_soquets`, `_flatten_soquet_collection`, `_get_flat_dangling_soqs` | `component_graph._collections`, `_defined_bindings`, `_elements`, `create_register_bindings`, SSA argument/result/return identities. Existing graph/bindings/notebook parity establishes row-major and directional endpoints. | Non-executable type/shape admission follows Q-0149; no separate graph identity representation is planned. |
| `BloqError`, `DidNotFlattenAnythingError` | `RegisterBindingError` and `NoComponentExpansionError`, both native value errors with existing failure tests. | Native exception names are explicit, source classes do not leak through the public boundary. |
| `_IgnoreAvailable`, `_IgnoreAvailable.add` | Existing binding constructor can omit availability tracking, while live builder owns its availability set. | A source private helper class is replaced by the native callable boundary. |
| `_reg_to_soq`, `_process_soquets` | `component_bindings.create_register_bindings`, `process_register_bindings`; source-backed contract tests cover shape, order, prefix mutations and width compatibility. | Wider descriptor admission remains tracked separately. |
| `_map_soqs`, `_map_flat_soqs`, `_update_flat_soq_map` | `component_binding_map.map_register_bindings`, copy/expansion SSA replacement maps. `test_component_inline_calls.py` covers scalar/shaped replacement, duplicate precedence, failures and inlining. | The public collection operation is covered; private flat-map mutation is expressed with native SSA dictionaries and must be assessed through copy/expansion consumers. |
| `BloqBuilder`, `__init__`, `add_register_from_dtype`, all `add_register` overloads, `from_signature` | `ComponentProgramBuilder` accepts a Signature and supports opt-in `allow_register_addition`; `component_signature.infer_output_registers`. Commit `b6a6705b5`, mutable-signature tests and audit. | Negative width source metadata is rejected at the native finite execution boundary. Fixed-point, bounded integer and symbolic bit-width graphs are admitted by the logical register contract. Symbolic element shapes and broader descriptor families remain open. |
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
explicit disposition. The source summary method is owned by Drawer and compared directly with native
immediate-layer counts, including nested shared definitions.

The frozen source file's own tests currently pass 39 cases and its original
notebook test passes separately. Those runs confirm the reference environment.
They do not close untested native replacements. Q-0157 is accepted through its complete test inventory. Q-0149 remains open
for broader descriptors and symbolic shapes, as do the dashboard numerical and PennyLane declaration gaps.

## Summary presentation follow-up

Drawer `format_component_summary` counts immediate verified call instances,
including repeated calls to a shared nested definition. It reports
`ComponentProgram([N components...])`, preserving the original summary shape
with native domain names. Empty graphs report zero. Invalid entries use the
existing Compiler diagnostics. This projection retains the stored graph.
The unchanged source parallel fixture and twice-wrapped fixture match after
the explicit domain-name substitution. The 25-file concentrated regression
passes 915 cases without skips. Strict typing of both changed Python files,
Black, Ruff, Compiler scope, architecture and internal IR checks pass.
The broader SDK wrapper conversion remains open.

## Recursive SDK decomposition follow-up

The Ecosystem register importer now resolves SDK decomposition protocols before
creating owned typed functions. Explicit native definitions take precedence.
Every sibling operand is reserved during recursive allocation. The shared
interop manager prevents collisions with caller inputs, including a deliberately
colliding allocator name. Dirty borrowing is refused before clean workspace is
assumed. Cycles and nesting beyond 256 frames produce an explicit error.

The frozen `SwapTest` and `BloqAsCirqGate` fixtures for widths one through three
match every emitted SWAP operand and the complete complex operator, including
reversed SDK operand order. Native gate export and reimport retain the same
primitives. A two-level clean-work decomposition emits four CNOTs and one T,
restores both work bits, and matches full complex amplitudes through the existing
`Circuit.append` workflow. Primitive resource counts are asserted directly.
This fixture has no independent source cost model, so no model estimate is
inferred from the primitive counts. The previous measured Program export tests
are reused for all-branch evidence, while recursive measured SDK leaves remain
open. The concentrated graph and Cirq regression passes 997 cases without skips.

At this batch boundary, the arbitrary SDK leaf gap was concrete. The physical
CircuitIR supported explicit matrices, but typed Program operations could not yet
carry an `any` matrix operand. Both control polarities of an original
`CirqGateAsBloq` matrix wrapper are verified as source evidence and rejected by
the native register importer with a specific definition requirement. This needs
a shared Core contract before implementation. Tensor-only source component
fallback, arbitrary channels, and recursive measurement-key transfer also remain
open. Q-0156 is therefore still unaccepted as a complete source file.

Strict typing of both changed Python files, Black, Ruff, Ecosystem scope,
architecture and internal IR checks pass. Dependency inspection records 709
whole-repository findings and zero changed-path findings. The default smoke/unit
tier stops after 434 passes and 51 skips at the pre-existing docstring-example
registry failure. Its test and every scanned example-bearing module are byte
identical to the baseline. The only changed production module contains no
docstring examples in either revision. The persistent evidence is
`g2-sdk-leaf-default-baseline.json`. Whole-repository acceptance remains open.

## Explicit unitary leaf closure

Proposal 103 prerequisite 1.21 now carries explicit unitary leaves through
Core, Compiler and Ecosystem. Core stores immutable built-in complex rows,
checks finite square unitary data, preserves old hashes when absent, and rejects
missing payloads or payloads on other opcodes. Compiler creates independent
complex128 physical storage while preserving wire order and classical clauses.
Ecosystem reuses its explicit matrix recognizer and emits one owned `any` atom.
The register-gate decomposition exports that atom as a copied SDK MatrixGate.

The unchanged source wrapper witnesses cover one- and two-qubit leaves, both
control polarities, reversed operands, full global phase and an entangled
spectator. Native Program binding and export/reimport match complete operators.
A graph-copy and `Circuit.append(op)` workflow applies the exact forward and
inverse matrices to work, releases the clean wire, and retains the data phase.
Both measurement outcomes and control polarities retain full complex branch
amplitudes after binding and after export/reimport, with the reset wire zero.
The measured circuit has one matrix atom, one measurement and one reset. The
clean-work workflow emits three matrix atoms and two lifetime events. These
are primitive counts at the explicit-matrix boundary. These fixtures provide
no independent Clifford/T synthesis model.

Validation includes 1,016 concentrated cases across 33 files without skips,
90 local integration and public-IR cases, strict typing of 12 files, Black,
Ruff, architecture, classified scope and the internal IR contract. The latter
checks 44 diagnostic codes and 15 rules. Dependency scanning records 706
repository findings and zero findings in this batch. Optional source-test
imports have explicit missing-dependency guards. The default smoke/unit run
again stops at the existing docstring registry failure after 434 passes and
51 skips. Logs use the `g2-matrix-leaf-` prefix in the reference environment.

The complete Q-0156 source file remains open. Recursive measurement-key
transfer and arbitrary nonunitary channels still need their own call-chain
treatment. Tensor-only component admission is covered below. Existing public static export admission
retains its established gate vocabulary. The matrix export here belongs to
the register-gate decomposition entry used by the source wrapper workflow.


## Tensor-only SDK fallback closure

The complete tensor-only wrapper path follows the unchanged source
`BloqAsCirqGate._decompose_with_context_` refusal, then `_unitary_` and
`bloq.tensor_contract()`. Ecosystem converts the resulting finite unitary to
an owned matrix leaf using the existing Core/Compiler contract. An available
SDK decomposition takes precedence, including an empty body. Existing native
Simulation tensor contraction owners remain authoritative. This adapter does
not introduce another tensor provider or retain a foreign component in Program.

Original `SwapTestWithOnlyTensorData` fixtures at widths 1 through 4 match full
complex operators with reversed input order, entangled spectators and native
`Circuit.append(op)`. Export and reimport retain the operator and typed binding.
Each imported tensor-only wrapper emits one `any` primitive, with no workspace
leases. Its original decomposable swap body has a different primitive expansion.
The tensor-only fixture supplies no independent Clifford/T resource model.
Previously accepted matrix feedback tests retain all measurement outcomes,
phase and reset-work evidence. Recursive measured wrapper import remains open.

The frozen implementation and test blobs are respectively
`d49ab63f9f74104dc5f2856a9e1c997635e21e5b` and
`6b2ab24b21b3bbb926a9228ac56a6b14a735104c`, matching the pinned inventory.
All 16 unchanged source tests, including the original notebook, pass. The
reference runner passes its frozen dependency path to the Jupyter subprocess.
The 33-file concentrated regression passes 1,021 cases without skips. The
three-file Cirq integration/conformance regression passes 96 cases. Strict
typing of both changed Python files, Black, Ruff, scope, architecture and
internal IR checks pass. Dependency policy records 706 repository findings
and zero changed-path findings. Default smoke/unit repeats the existing
example-registry failure after 434 passes and 51 skips. Persistent evidence
uses the `g2-tensor-leaf-` prefix. Implementation commit is `c00ab7cea`.


## Recursive SDK measurement-history closure

Commit `95bf52007` extends recursive SDK import through the existing Ecosystem
feedback translator. Expanded leaves share one measurement history before they
enter a typed bit-gate function. Register binding owns the clean work lease and
returns it after the complete body. External measurement keys terminate at the
adapter and native classical slots preserve old and latest records. Classical
controls expand with their original predicates. Tagged operations are rejected
before removing controls because the SDK otherwise discards the tags. Shared
primitive lowering now handles ISWAP and its adjoint in measured and static bodies.

The unchanged source CompositeBloq-to-Cirq call chain runs two And/And-adjoint
cycles for all four control-value combinations. Primitive instructions, operands
and feedback match exactly. There are 34, 42 or 50 measured primitive instructions
as zero-valued controls increase. The source QEC model reports two ANDs, two
Cliffords and two measurements for every combination. Its negative-control X
omission remains recorded separately from measured primitive counts. One clean
work bit has one enclosing allocation/release pair and is zero in every branch.
All four histories match unnormalized full complex amplitudes with an entangled
spectator. Export/reimport and normal dynamic `circuit.append(op)` execution pass.
Additional cases retain inverted, old and latest repeated records, and verify
ISWAP/adjoint phases with reversed operands in both measurement branches.

The 33-file concentrated regression passes 1,032 cases without skips. Cirq
integration/conformance passes 96 cases. Strict typing of three changed files,
Black, Ruff, scope, architecture and internal IR checks pass. Dependency policy
records 706 global findings and zero batch findings. The default smoke/unit
suite retains its pre-existing example-registry failure after 434 passes and
51 skips. Logs use `g2-recursive-feedback-`. Prior original-source whole-file
execution and pinned source hashes are reused from the tensor-leaf batch.

The complete Q-0156 parent remains open. A measured SDK body containing an
explicit native FunctionRecord is currently rejected with a precise error. Its
native-definition precedence must be retained when composing classical history.
Arbitrary channels also remain outside this adapter's admitted scope.


## Explicit native definitions inside SDK feedback

Compiler commit `5a0020d76` introduces private local component binding through
existing Program validation, physical binding and workspace ownership. The sum
of finite call-tree allocations provides a conservative pool capacity. Final
bindings compact unused wires, including the state shape. This capacity is an
internal construction bound, not a resource-model estimate. Actual allocation
and release events remain authoritative. Clean and borrowed pools, repeated
work reuse, consumed LEFT inputs and transferred RIGHT outputs are covered.
Invalid forward references retain Core rejection. Empty components retain a
scalar carrier without emitting a work lease.

Ecosystem commit `87aaf8bad` translates explicitly selected native definitions
through that Compiler owner and the existing register-gate exporter. Definitions
retain precedence over SDK decomposition. Local measurement names avoid both
external measurement names and external control references, so generated names
cannot accidentally satisfy an undefined external key. Repeated native calls
have separate local histories while sharing the outer SDK history.

The second expansion replays the original SDK tree in one manager context.
Re-expanding a flattened tree would reserve old SDK workspace for the entire
body and prevent subsequent native calls from reusing it. The mixed SDK/native
controlled fixture matches the independent original decomposition instruction
for instruction, including operands and conditions: nine actual primitives and
one clean work bit, returned to zero in both measurement branches. Export and
reimport retain full complex amplitudes; dynamic append execution also passes.
Two repeated measured native calls with a deliberately colliding external key
produce twelve actual primitives and three measurements. All eight branches
match full unnormalized source amplitudes and restore private work. These SDK
fixtures provide no independent resource model; no model is inferred from
measured primitive totals.

The 34-file concentrated regression passes 1,043 cases without skips. Cirq
integration/conformance passes 96 cases, and five changed Python files pass
strict typing, Black, Ruff, classified scope, architecture and internal IR
checks. Dependency policy records 706 whole-repository findings and zero batch
findings. Default smoke/unit retains the existing example-registry failure after
434 passes and 51 skips. Persistent evidence uses `g2-native-feedback-`.

Q-0156 remains open for arbitrary SDK nonunitary channels. The existing admission
rules for dirty SDK borrowing, conditional measurement, tagged semantics,
nonfinite or symbolic physical definitions and external CBit ports remain in
force. This batch changes private lowering without a public API or schema change.
