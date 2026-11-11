# Tensor protocol source audit

Reference: Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The reference checkout and its original
tests are unchanged. This audit covers the complete `_flattening_test.py` and
`_tensor_from_classical_test.py` files, both Simulation namespace assets, and the
existing native implementation boundaries used by those tests.

## Ownership and interfaces

Compiler owns typed graph transformation. Simulation owns numerical tensors,
explicit leaf suppliers and classical value codecs. The interface between them
remains the existing function name predicate and selected Program graph;
numerical callbacks are not executed to choose leaves. There is no new IR,
provider registry, foreign-object dispatcher or stable API export.

The five direct users of `register_tensor_label_groups` now import its
`register_axis_groups` owner. The preceding move left imports in classical action,
classical parity, controlled tensors, identity tensors and zero-state tensors
pointing through `register_tensor_binding`; strict mypy rejected that implicit
reexport. This batch fixes those imports and the binding test's import. An AST
comparison with `ca5f7774` confirms that all five production modules are unchanged
apart from imports. The existing binder still uses the shared owner itself.

| Source implementation surface | Existing native target and remaining boundary |
|---|---|
| `bloq_has_custom_tensors` | `ComponentTensorProviders.has_custom_tensor`: exact, explicit function-to-supplier declarations. Source Python `my_tensors.__qualname__` inference is not transplanted into Core metadata. |
| `flatten_for_tensor_contraction` | Compiler `expand_components_for_tensor` and `expand_component_root_for_tensor`, reusing `expand_component_calls`. Full expansion bypasses classification; partial expansion stops at declared suppliers. Foreign lazy decomposer exception behavior remains outside the audited native boundary. Q-0889 remains open. |
| `_bits_to_classical_reg_data` | Existing native scalar/array bit and integer codecs supplied to the classical action owner. Input array shape uses register shape followed by the source quantum-bit width. |
| `_bloq_to_dense_via_classical_action` | `_classical_action_tensor_data`: enumerate input bits, split registers, decode, call the supplied classical callback, encode output bits and set output-before-input entries. |
| `bloq_to_dense_via_classical_action` | `classical_action_tensor`: source ValueError context/cause and scalar/vector/matrix grouping. |
| `my_tensors_from_classical_action` | `classical_action_tensor_node`: use the raw enumerator and existing binary-axis binder. Native connections are validated before enumeration, and classical ports are explicitly refused. Source counts all bits for data but quantum bits for supplier indices; its empty-array and phase-free limitations remain recorded. Q-0895 remains open. |

Registers retain row-major element order and MSB-first word bits. No physical
work is allocated by declaration lookup or classical table generation. Callers
still own the action and codecs; a numerical table does not infer an unknown
phase or automatically supply a missing physical decomposition.

## Complete original test replacement

All seven original cases run unchanged. The same original test functions then
run with their specific numerical or classification boundary replaced:

| Original file and test | Native replacement |
|---|---|
| `_flattening_test.py::test_bloq_has_custom_tensors` | Explicit provider declarations mark CNOT as supplied and MultiAnd as decomposable. An assertion guard confirms that classification never executes a numerical supplier. |
| `_tensor_from_classical_test.py::test_tensor_consistent_with_classical` (all five parameters) | Native classical table generation and codecs for X, controlled swap, Toffoli, **3-bit** Add and **3-bit** Xor, compared with the original tensor path by the unchanged original assertions. |
| `_tensor_from_classical_test.py::test_my_tensors_from_classical_action` | Native node/table generation. The test-only adapter transports existing source connection labels into integers and transports the resulting node back to the original test's Quimb boundary. |

The adapter retains the supplied source classical callback as an input. It does
not call the source table generator. The original builder, original tensor
reference and original assertions remain in the test fixture. This proves those
specific numerical and selection replacements; it is not a claim that every
foreign builder, dtype or Python decomposition exception has been replaced.
Existing native graph/root/provider regressions cover their own production paths.
Q-0890 and Q-0896 close as complete original test-file mappings.

## Namespace asset disposition

Q-0880 (`qualtran/simulation/__init__.py`) contains a docstring only: no imports,
functions, classes, constants, data or executable initialization. Its Simulation
ownership description is represented by the native Simulation README. No
upstream package identity is copied.

Q-0886 (`qualtran/simulation/tensor/__init__.py`) contains one docstring and 15
explicit reexports. AST inspection verifies the entire import set and absence
of additional initialization. Each name has an existing native owner:

| Source reexport | Native owner |
|---|---|
| `bloq_to_dense` | `component_dense.contract_component_tensors` / `contract_component_superoperator` |
| `quimb_to_dense` | `register_tensor_binding.contract_register_tensors` / `register_superoperator_binding.contract_register_superoperator` |
| `bloq_has_custom_tensors` | `component_tensor_providers.ComponentTensorProviders.has_custom_tensor` |
| `flatten_for_tensor_contraction` | Compiler `component_tensor_expansion` composite/root entries |
| `cbloq_to_quimb` | `component_tensor_nodes.component_tensor_nodes` |
| `cbloq_to_superquimb` | `component_superoperator_nodes.component_superoperator_nodes` |
| `DiscardInd` | Existing integer trace requests in mixed suppliers and `trace_indices.trace_conjugate_indices` |
| `initialize_from_zero` | Compiler `component_zero_initialization.initialize_component_from_zero` |
| `active_space_for_ctrl_spec` | `register_tensor_data.register_active_indices` |
| `eye_tensor_for_signature` | `register_tensor_data.register_identity_data` |
| `tensor_data_from_unitary_and_signature` | `register_tensor_data.project_register_tensor` |
| `tensor_out_inp_shape_from_signature` | `register_tensor_data.register_word_shapes` |
| `tensor_shape_from_signature` | `register_tensor_data.register_word_shape` |
| `bloq_to_dense_via_classical_action` | `classical_action_tensor.classical_action_tensor` |
| `my_tensors_from_classical_action` | `classical_action_tensor.classical_action_tensor_node` |

These are responsibility mappings, not new convenience exports. Native
readmes and internal module imports remain authoritative. Closing the two
namespace assets does not close their referenced implementation files or imply
support for their inherited defects. No package initializer or protected API
snapshot is changed.

## Verification and resources

Implementation/import repair: `b3352d37`. Original-test replacements: `f93505f5`.

- Nine audit cases pass: seven original boundary replacements and two complete
  namespace AST inspections. All seven original reference cases also pass.
- Concentrated regression: 506 passed in 9.08 s, with 42 reference Autoray
  matmul warnings. Existing tests cover ordered gates, complete complex
  amplitudes, all small measured-Add history pairs, clean work, gradients,
  supplier replacement and normal `circuit.append(op)` execution.
- CPU/representation/tensor-network regression: 84 passed in 6.56 s, with two
  known optional optimizer/deprecation warnings.
- Three relevant production owners cover 96 statements and 20 branches at 100%:
  Compiler tensor expansion, Simulation explicit providers and classical action.
- Strict Python 3.12 mypy passes nine files. Black, Ruff, Simulation scope,
  architecture, multi-level IR and protected API checks pass.
- Seven source Git blobs match the immutable inventory, including both
  namespaces, both implementations, both original test files and dtype codecs.
  Qualtran 0.7.0 and native Kickmix `Simulator` are checked before reference runs.

Actual new primitives: **zero**. New physical work bits and feedback slots:
**zero**. Source gate cost model: **not applicable** for lookup declarations,
classical numerical enumeration or namespace assets. Existing measured component
gate counts remain separate from source QEC estimates. Import repair is not
reported as a new quantum decomposition or a performance improvement.

Default smoke/unit reports 6 failed, 21,868 passed, 300 skipped and 1,037
warnings in 124.66 s. All six failures belong to the independent seven-case
baseline; one baseline failure is not observed before xdist stops at its failure
threshold. There are no new observed failures, but this is not a full-green or
complete default-suite result.

Default-suite comparison and publication evidence use prefix
`g2-tensor-protocol-source-audit-` in the local reference evidence directory.
G2 remains in progress; G3 is not entered and the PR is not merged.

```bash
python -m pytest tests/unit/test_tensor_protocol_source_audit.py \
  tests/unit/test_component_tensor_expansion.py \
  tests/unit/test_classical_action_tensor.py -q
```

Run the source-backed checks in the locked reference environment. For a normal
production change, edit the existing owner rather than adding a second copy of
the protocol under an upstream package name.
