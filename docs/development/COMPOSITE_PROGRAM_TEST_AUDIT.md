# Composite Program original-test inventory

This audit covers every top-level test in frozen Qualtran v0.7.0
`qualtran/_infra/composite_bloq_test.py`, Git blob
`83dfa234bc95c8310f1d3d8526fbb2aa9b538a7f`. There are 38 test functions. The two
parametrized functions add two collected cases, giving 40 cases including the
notebook. The connected fixed-point Program path in `test_type_error` now has
a native logical-graph replacement. Physical execution remains subject to its
separate descriptor admission. See `LOGICAL_REGISTER_PROGRAM_AUDIT.md`.

The inventory is checked against the pinned source file. Existing evidence is
reused by complete behavior group. Native files in this table are under
`tests/unit/`. These references describe evidence coverage and explicit gaps;
they are not an assertion that every source Python helper has a native class.

| Original test functions | Native evidence and owner | Disposition |
| --- | --- | --- |
| `test_create_binst_graph`, `test_iter_bloqnections`, `test_iter_bloqsoqs` | `test_component_graph.py`, immutable SSA traversal and dependency generations | Native connection/instance representation replaces mutable source graph. Every predecessor, output and shape is compared. Isolated source frozenset ties retain an explicit ordering boundary. |
| `test_composite_bloq` | `test_component_debug_text.py` | Generation separators, incoming/outgoing connection grouping and indexed names match after documented native spelling conversion. |
| `test_map_soqs` | `test_component_inline_calls.py`, mapping duplicates/unmapped/empty arrays and crossed CNOT inlining | Source private instance counter is represented by explicit occurrence labels. |
| `test_to_from_cirq_circuit` | `test_component_source_export_examples.py`, `test_cirq_register_program_export.py` | The actual source TestTwoCNOT is compared to native construction, including diagram, output permutation, primitive sequence and full operator. |
| `test_bloq_builder` | `test_component_program_builder.py`, `test_register_component_calls.py` | Live calls create distinct results and preserve source consumption order. |
| `test_wrong_soquet`, `test_double_use_1`, `test_double_use_2`, `test_missing_args`, `test_too_many_args` | `test_component_program_builder.py`, `test_register_bindings.py`, `test_register_component_calls.py` | Foreign/stale handles, aliasing, missing and excess names retain tested failure prefixes. Source Python exception spelling is replaced by native diagnostics. |
| `test_finalize_wrong_soquet`, `test_finalize_double_use_1`, `test_finalize_double_use_2`, `test_finalize_missing_args`, `test_finalize_strict_too_many_args`, `test_finalize_bad_args` | `test_component_program_builder.py`, `test_component_mutable_signature.py` | Strict and inferred returns have source-backed ownership/error-timing tests. Source can retain an incomplete signature-only failed instance; native verified bodies reject it. |
| `test_finalize_alloc` | `test_component_mutable_signature.py`, `test_cirq_register_program_export.py` | Inferred RIGHT metadata and transferred clean/borrowed output leases have independent coverage. |
| `test_get_soquets` | `test_register_binding_creation.py`, `test_component_rebindings.py` | Scalar and indexed left/right bindings use immutable native values. |
| `test_complicated_target_register` | `test_component_source_export_examples.py` | The actual source TestMultiCNOT and its 2×3 bit array are compared in row-major order, with exact diagram, all six primitive CNOTs and full 128×128 operator. |
| `test_util_convenience_methods`, `test_join_list`, `test_util_convenience_methods_errors` | `test_component_rebindings.py`, `test_component_workspace.py` | Actual split/join/allocate/free Programs and malformed shape/value cases replace source convenience calls. |
| `test_test_serial_combo_decomp`, `test_test_parallel_combo_decomp`, `test_copy` | `test_component_construction_notebook.py`, `test_component_graph.py` | Full original notebook, serial/parallel graph construction, independent copies, display labels and physical execution have published evidence. |
| `test_add_from` | `test_component_inline_calls.py`, `test_component_construction_notebook.py` | Inlining a Program or its declared root retains boundary/shape mapping and exact source connection text. |
| `test_final_soqs`, `test_add_from_left_bloq` | `test_component_inline_calls.py`, `test_component_program_builder.py`, `test_cirq_register_program_export.py` | Left-only consumption has empty returned mappings and verified releases. |
| `test_add_duplicate_register` | `test_component_mutable_signature.py` | Native rejects duplicate registers before finalization. The stricter admission time is explicit. |
| `test_flatten` | `test_component_expansion.py`, `test_component_construction_notebook.py` | One-level and recursive expansion retain exact immediate graph structure and primitive behavior. |
| `test_type_error` | `test_logical_register_program.py`, `test_logical_register_contract.py`, existing descriptor consistency | Actual frozen TestMultiTypedRegister/TestQFxp calls and native typed graphs preserve producer types, successful QFxp/QUInt connections, both failure consumption prefixes and graph copies. The original source test executes unchanged. |
| `test_t_complexity` | `test_component_construction_notebook.py`, `test_component_resource_model.py`, published notebook evidence | Synthetic 100/300 T source models remain separate from 1/3 emitted X gates. |
| `test_add_and_partition` | `test_component_auto_partition.py`, `test_component_program_import.py` | Controlled Swap with reordered packed registers and Unused slices has existing primitive/operator coverage. Builder import delegates to the same partition implementation. |
| `test_decompose_symbolic_register_shape_raises` | `test_component_program_builder.py`, native finite descriptor admission | Finite execution refuses unresolved symbolic shape. Wider non-executable symbolic graph admission remains Q-0149. |
| `test_get_soquet` | `test_component_graph.py`, `test_register_binding_creation.py`, directional call tests | Directional endpoints are derived from typed operands/results. Missing endpoints raise native value errors. |
| `test_notebook` | `test_component_construction_notebook.py`, `COMPONENT_CONSTRUCTION_TUTORIAL_AUDIT.md` | All 42 source cells and native replacements previously executed, with every immediate connection stage and complex tensor compared. |

## New exact source example evidence

The added test imports the unchanged original test fixtures. It constructs their
native counterparts using the ordinary ComponentProgramBuilder and CNOT provider,
binds physical inputs through Compiler and exports through the accepted Cirq
adapter. Both source circuits have zero workspace and no measurement branch.
The two-gate example and six-gate 2×3 example compare exact primitive order,
output arrays, diagram text, full complex unitary matrices and source QEC Clifford
counts. This comparison specifically covers the source examples, in addition to
the broader preexisting shape/lifetime tests.

## Full-file disposition

All 38 original test functions have mapped native evidence, including the
connected fixed-point construction path. This closes the original test-file
scope. Broader source APIs in Q-0156 and symbolic descriptor families in Q-0149
remain independently tracked. The new graph contract admits fixed-point and
bounded integers without claiming their numerical execution or inventing
bodies for declaration-only source fixtures. Symbolic array shapes remain
outside the concrete-element graph profile.

## Prior source-example audit results

The concentrated native regression covers 17 relevant files and passes 705 cases
without skips in 20.64 seconds. This includes the full new source examples,
builder/return errors, inlining, graph copy, expansion, workspace, descriptor
compatibility, controlled partitions and the complete native construction
notebook. The unchanged source-test evidence is reused: 39 non-notebook cases and
one notebook case pass separately. Those original cases confirm the reference;
the logical-register follow-up supplies the native fixed-point Program evidence.

The new test module passes strict typing, Black and Ruff. Scope and architecture
checks pass. The dependency scan retains 706 existing findings with zero on the
new path. That source-example batch changed no production behavior. The subsequent logical-register batch follows proposal 103 prerequisite 1.20 and internal IR contract 0.9.
