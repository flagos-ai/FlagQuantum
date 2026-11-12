# Explicit antidiagonal gauge and diagonal reduction

```bash
python examples/algorithms/tensor_diagonal_reduction.py
# A/D/C/R: 3 -> 1 tensors; 5 -> 1 axis occurrences
# Output: exp(i * phase) * [17, 22]; fixed-support gradient preserved.
```

This extends Q-0887.03 with explicit A/D dependencies of the pinned dense tensor
policy. Squeeze/C/R already exist. The fixed-point loop and complete width > 12
trigger remain open. No default training or register simplification is enabled.

## Ownership and execution

Simulation owns numerical preprocessing of existing integer-labeled nodes:

| File | Responsibility |
| --- | --- |
| `tensor_structure.py` | Shared finite support mask and lexicographic diagonal/antidiagonal pair detector |
| `tensor_columns.py` | Existing C detector reuses that support mask |
| `diagonal_reduction.py` | A edge gauge and D edge merging in source traversal order |
| `register_tensor_indices.py` | Capture actual external register groups before preprocessing and validate their reuse |
| Existing pure/superoperator binders | Pass captured groups to the same greedy contraction consumer |

Capture `register_contraction_groups(nodes, signature, incoming, outgoing)` before
simplification, protect its flattened labels in each pass, then supply its tuple
as `output_groups` to the existing register consumer. Use `superoperator=True`
for doubled nodes. D may merge an internal edge into a protected output, creating
an output hyperedge. Re-inferring outputs by single occurrences would then lose
that axis. Qualtran likewise captures groups before `full_simplify` and passes
those groups to dense contraction afterward. Cached groups must match the current
signature, bindings and present labels. Default missing-axis failures remain;
there is no permissive flag to treat arbitrary internal edges as register ports.

No new graph, executor, public export, physical gate emitter or compiler policy
is introduced. Inputs are immutable; names and metadata survive transformations.
Torch numerical operations retain dtype, device and retained autograd paths.

## Locked source and exact comparison scope

Qualtran v0.7.0 is pinned at
`bcb17bb25360ed68ce3bc084645db3e929799389`; `_dense.py` has blob
`13aa81f50654ccaaa4293e8a63991686723022f5`. Qualtran does not pin Quimb, so these
comparisons explicitly lock restored **Quimb 1.15.0**, Apache-2.0, unchanged.
Tests hash the original implementations and execute them directly.

| Quimb original | SHA256 |
| --- | --- |
| `TensorNetwork.antidiag_gauge` | `407328b6216b3ec7855139c8d73f4756482e6d80b0cdbaef171035293615cc10` |
| `TensorNetwork.diagonal_reduce` | `2f8688060fe3b9d8889cd6776b8c70211e5cd0cdcf1cf7cb902faf1ed7157f7d` |
| `Tensor.flip` | `403fa01a5cfeddf739dbeaa4ffd58aa4b73aea49f3ff3bb45aa6f093c863e429` |
| `array_ops.find_antidiag_axes` | `af67cbcc85833c1418f6c683560bd90e27dcdd13c7bad3c55140ca552170682f` |
| `array_ops.find_diag_axes` | `300d744353cb525adcf294957b4c913a0088209168e81cc93d7b59457c4b1366` |
| `_numba_find_antidiag_axes` | `db17dc20e7d9ffe9f143da414655ed5f0546838a3f981ffbf2182a5f1236ac9d` |
| `_numba_find_diag_axes` | `2e648627499e3ca84c4e1d9e900a8b49b196019a428de853df2c908482467751` |

Tests instrument every flip/reindex selector and compare every resulting leaf's
labels and data. Both passes use the reverse occurrence queue, first candidate,
output protection and triggering-node requeue. A flips each selected edge at most
once. D globally reindexes and collapses repeats only on the triggering tensor.
The pair detector follows equal-size axis pair order and strict `abs(x) > atol`
activity, including empty/all-zero tensors. It avoids rank-sized coordinate grids.
Data-dependent support detection synchronizes device execution; no GPU performance
or distributed-capacity claim is made.

## Explicit upstream defect and numerical boundaries

Quimb `Tensor.flip(ind)` selects only the first occurrence of a repeated index.
On a single tensor `[[0,1],[2,0]]` with labels `(0,0)`, the initial trace is zero.
Original A flips one axis and changes that trace to three. Native A flips every
occurrence on every node, retaining trace zero. The reference remains untouched;
this is a separately asserted defect, not per-leaf parity on repeated-edge A.

Flipping all occurrences of an internal edge is an exact gauge, even when its
candidate detection uses tolerance. It preserves the full contraction and ambient
gradient. These flips are numerical index reversals, not physical X gates.

D preserves values at zero tolerance, up to contraction roundoff, and gradients
for fixed sparse parameterizations. It does not preserve arbitrary ambient
Jacobians: a diagonal leaf `[1,2]` against `[[3,4],[5,6]]` has unsimplified gradient
`[[3,4],[5,6]]`, while source and native D retain only `[[3,0],[0,6]]`. At tolerance
1e-12, an off-diagonal value 0.5e-12 multiplied by an environment entry 1e12
changes 15.5 to 15. Local tolerance supplies no global contraction-error bound.
Tests retain both counterexamples; this policy is not silently enabled in training.

Support detection accepts finite float32/64 and complex64/128 tensors and finite
nonnegative real tolerance. It refuses meta, nonfinite and unsupported precision
inputs. Shape-only squeeze still supports meta. Existing C failure/gradient
boundaries remain in [the C evidence](TENSOR_COLUMN_REDUCTION.md).

## Validation and resources

Prepared and unitary Bell workflows retain the same native consumer and agree in
complete complex outputs and phase gradients across explicit squeeze/A/D/C/R.
Pure and doubled register tests cover protected output hyperedges, gradients,
invalid captures and missing outputs. No measurements or work wires are added;
physical append/all-branch/workspace tests remain in concentrated regression.

The example has three input tensors and five axis occurrences, reduced to one
tensor and one axis. Two complex128 output entries occupy 32 bytes; this is output
storage, not measured process RAM. Measured added physical gates, measurements,
work bits and feedback slots are all zero. A quantum gate-cost model does not
apply to these numerical helpers. Numerical structure reduction is not a gate
count reduction claim.

Local acceptance: **146 new scenarios**, **2,292 related tests in 55.29 s**,
**84 CPU/TN tests in 6.96 s**; strict mypy on eight changed Python files,
Black/Ruff/scope/architecture/IR/protected API and executable example pass.
The two new numerical owners plus example cover **122 statements and 40 branches
at 100%**; this is not whole-binder/contraction coverage. Implementation: `051cde78`.
The
prior default-suite timeout and source numerical warnings remain open; focused
success is not full-repository certification. Q-0887 stays open for fixed-point
composition and complete width-trigger integration, with no duplicate capability
ID and no G3 or merge expansion.
