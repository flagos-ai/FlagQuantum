# Dense tensor axis source audit

Reference: Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`qualtran/simulation/tensor/_dense.py` and `_dense_test.py`.
The reference checkout is unchanged. Apache-2.0 attribution remains in the
adapted numerical owners. This is a local single-device tensor contraction path.

## Ownership and source disposition

| Source surface | FlagQuantum owner and disposition |
|---|---|
| `_order_incoming_outgoing_indices` | `register_axis_groups.register_tensor_label_groups` owns declared right/left, row-major word and bit order. Existing validated port checks remain. `register_tensor_binding` imports this owner rather than keeping another loop. |
| `_group_outer_inds` | `register_axis_groups.group_register_outer_indices` groups actual external edge descriptors using the source's left-before-right lookup traversal and right-before-left output order. Pure groups are `r,l`; mixed groups are `rf,rb,lf,lb`. |
| `quimb_to_dense` | Existing pure/mixed register contraction entries now count node edge occurrences and select the actual exposed descriptors before calling the shared grouping owner and existing PyTorch greedy executor. Scalar, vector, matrix and channel results keep their existing shape conventions. |
| `bloq_to_dense` | Existing `component_dense` entries compose Compiler-owned flatten selection, component tensor providers, node wiring and the register numerical owners. This batch adds no graph, executor, registry, discovery heuristic or public API. |
| Source logging and foreign graph types | Remain reference evidence; production uses native IR and tensors. The test-only adapter translates source network tensors to existing integer-labeled nodes at the numerical boundary. No production import of Qualtran or Quimb is added. |
| Width-based simplification | **Open:** source calls `full_simplify(inplace=True)` only when `contraction_width() > 12`. Native greedy contraction has no equivalent audited simplification policy. Q-0887 remains open. |

The grouping input maps `(register_name, element_index, bit_index, group)` to an
existing integer edge. Indices use the existing diagnostic side table; they are
not a second graph or an allocation contract. Registers retain their existing
dtype widths, including classical bits. Extra descriptors are ignored, missing
live descriptors raise `KeyError`, and empty groups are omitted. Port validation
and duplicate endpoint refusals remain owned by the existing validated boundary.
The raw source group rule does not invent those validations.

At least one numerical node is required, including a scalar node for an empty
signature. Missing declared edges no longer pass directly to einsum; both pure
and mixed contraction identify the missing external descriptor first. PyTorch
retains dtype, device and autograd. Storage is still exponential in the number
of exposed bits; this work makes no new memory-capacity or performance claim.

## Original tests and replacement

All five original tests run unchanged against the pinned checkout. The same
five functions are also invoked with only source `quimb_to_dense` replaced by
native integer-node contraction:

| Original test | Preserved behavior |
|---|---|
| `test_bloq_to_dense` | Shaped/custom tensor ports; the only nonzero entry is output 4, input 13. |
| `test_nest` | Single nested component and decomposed graph both give X. |
| `test_double_nest` | Two nested levels and the decomposed graph both give X. |
| `test_bloq_with_non_trivial_inds` | X/CNOT/Z internal connections preserve the complete matrix and phase. |
| `test_bloq_stop_flattening` | Full and partial source flattening preserve decomposition/supplier flags and equal numerical results. |

This replacement retains the source graph construction and source flattening
when executing the original tests. It proves replacement of the numerical
boundary, not native replacement of every foreign builder or exception path.
Existing native component/provider/expansion regressions separately cover the
FlagQuantum call chain. Q-0888's complete test-file mapping is accepted;
Q-0887 and Q-0889 remain open for their independent implementation gaps.

The 65 audit cases cover all side combinations with scalar, shaped and empty
registers, pure/mixed groups, shuffled actual indices, ignored extras, each
missing live edge, input-first failures, scalar gradients and the exact source
simplification threshold. Existing regression exercises complex amplitudes,
measured Add history pairs, clean-work checks, provider replacement and normal
`circuit.append(op)` consumption. No redundant quantum scenario family is added.

## Verification and resources

- 765 concentrated tests pass, including 65 audit cases. Three known reference
  Autoray matmul warnings remain in the measured-Add graph test.
- Five unchanged original tests and all five native numerical replacements pass.
- 84 CPU vertical-path, representation and tensor-network tests pass (two known
  optional-path/deprecation warnings).
- Four changed numerical owners cover 103 statements and 38 branches at 100%.
- Strict Python 3.12 mypy, Black, Ruff, Simulation scope, architecture,
  multi-level IR and protected API snapshot checks pass.
- Five source Git blobs match the frozen source inventory, including dense,
  dense tests, graph conversion, flattening and register metadata.
- Qualtran 0.7.0 and the native Kickmix `Simulator` are checked before reference
  regression; no reference tests are silently skipped.

Actual emitted primitives: **zero** for axis grouping. Quantum work bits and
classical feedback slots: **zero**. Source gate-cost model: **not applicable**
for a numerical grouping helper. Existing executable component gate counts and
source QEC estimates remain separate in their own evidence; contraction work is
not reported as quantum gate count. No synthetic gate-count equality is claimed.

Implementation: `837b69c3`. The default smoke/unit run reports 7 failed,
21,890 passed and 300 skipped in 157.09 s (1,037 warnings). All seven failure IDs
match the recorded baseline exactly; there are no new observed failures. This
bounded xdist run stops after the failure threshold and is not a full-green
suite claim.

Detailed logs and source hash evidence are in the local reference evidence
folder with prefix `g2-dense-axis-source-audit-`. Default-suite results and
publication SHAs are recorded in the synchronized integration ledger.

## Contributor check

```bash
python -m pytest tests/unit/test_dense_axis_source_audit.py \
  tests/unit/test_register_tensor_binding.py \
  tests/unit/test_register_superoperator_binding.py \
  tests/unit/test_register_tensor_indices.py -q
```

Use the locked reference environment for source tests. Production grouping only
consumes native signatures and existing edge descriptors. Keep optional foreign
network simplification outside this shared axis-order owner until its numerical,
failure and resource behavior is independently audited.

## Follow-up: 2026-11-12

The historical open-width row above is superseded by `TENSOR_WIDTH_POLICY.md`
and `DENSE_ENTRY_DISPOSITION.md`: the explicit reported-width/ADCR policy and
fresh-label dense composition are implemented. The five original test bodies
also pass with explicit tolerance. Native default training semantics, source
defects and foreign flattening boundaries remain explicit; Q-0887 stays open.
