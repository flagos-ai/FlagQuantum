# Explicit tensor squeezing and column reduction

Run the executable numerical example:

```bash
python examples/algorithms/tensor_column_reduction.py
# Squeeze/C/R: 4 -> 1 axis occurrences
# Output: exp(i * phase) * [3, 4]; fixed-support gradient preserved.
```

This batch adds the squeeze/C dependencies of Qualtran v0.7.0's wide-network
simplification policy. R was delivered previously. A/D and complete policy
integration remain open; these operations are explicit preprocessing and are
not automatically enabled in register contraction or training.

## Responsibilities and interface

| Owner | Responsibility |
| --- | --- |
| `simplification_axes.py` | Shared output protection/axis validation and shape-only `squeeze_tensor_axes` |
| `tensor_columns.py` | Single-tensor `find_tensor_column` support detection at explicit tolerance |
| `column_reduction.py` | Source-ordered `reduce_tensor_columns` traversal over existing nodes |
| Existing `contraction._slice_nodes` | Actual selection of every occurrence of an integer edge |
| Existing rank/binding/contraction | R reduction, register grouping and numerical execution |

R now reuses the same output validation instead of repeating the axis census.
The existing slicing helper selects every occurrence of a repeated label;
previously it selected only the first axis. Unique-label behavior is unchanged.
No foreign graph, parallel executor, registry, new public export or protected API
is introduced. Untouched data and metadata retain identity; reduced nodes retain
name/metadata and use native view/slice operations. Input nodes/data are immutable.

Outputs default to labels occurring once in the original network. Explicit
outputs can protect a hyperedge. Squeeze removes only **size-one unprotected
axes**, preserving output extents, zero-sized dimensions and scalar nodes. This
is different from the existing slicing-plan canonicalizer, which normalizes
output unit extents and absorbs scalars. Squeeze is shape-only, supports meta
tensors and preserves the complete autograd map.

## Locked reference and traversal

Qualtran v0.7.0's pinned `_dense.py` blob is
`13aa81f50654ccaaa4293e8a63991686723022f5`; its `quimb_to_dense` delegates to
`full_simplify` when width > 12. Qualtran itself leaves Quimb unpinned. The
comparison explicitly uses unchanged **Quimb 1.15.0**, Apache-2.0:

| Original method | SHA256 |
| --- | --- |
| `TensorNetwork.column_reduce` | `cd9f0952b0704c01baf3db84a10aa76b59e693d4a5ec8c228d18bb5ce132018e` |
| `Tensor.squeeze` | `f39812d153c60d87fe408a8afad8fa1f265a12ffc3a39b04dd5cb83e573ce33d` |
| `array_ops.find_columns` | `6c4ef75c51594c4f6f1a43c52f12b2fbce96c1d1d76776f11d74ac35c6bb59c4` |
| `_numba_find_columns` | `dc5fe0346981af7bf93a10c35546be7adcb0a8ddfaaaf6640ce4d79d2c2143e2` |

Tests assert these hashes and execute the original methods without rewriting
reference numerical code. Boundary instrumentation records each network cut;
native/reference selectors and every resulting tensor's axes/data agree.

C processes occurrences in reverse order. The first eligible `(axis, column)`
is lexicographic, following the reference NumPy/Numba path. An entry is active
only when `abs(value) > atol`; exact equality is inactive. All-zero data selects
the first nonempty axis and column zero. If that first candidate is an output,
the tensor is left alone; native code does not search a later internal axis.
Every matching axis on every connected tensor is sliced, including repeated
indices and hyperedges. Only the triggering occurrence is requeued; already
processed neighbors are not revisited in the same pass. No external persistent
Quimb cache interface is introduced.

Detection uses one support mask and per-axis reductions instead of a full
coordinate grid. It performs no NumPy copy or device transfer, but reading the
data-dependent decision synchronizes device execution. This is not a shape-only
plan or a GPU-performance claim. Tests cover CPU float32/64 and complex64/128.

## Tolerance, gradient and failure boundaries

C is a support-dependent transformation. With atol > 0 it can discard small
contributions: `[1, 0.5e-12]` contracted with `[2, 1e12]` is 2.5, while source
and native C at atol=1e-12 both return 2. There is **no global contraction-error
bound** supplied by the local support tolerance. Calling with atol=0 retains
all nonzero contributions exactly, up to normal contraction roundoff.

Gradients through retained slices are preserved. A phase multiplying a fixed
sparse ket gives the same complete complex output and parameter gradient before
and after squeeze/C/R. Prepared and unitary Bell graphs replace only preprocessing
before the existing consumer and pass the same checks. This does not establish
the ambient Jacobian for arbitrary trainable entries: at a leaf value `[1,0]`
against `[2,3]`, unsimplified gradients are `[2,3]`, but both original and native
C produce `[2,0]`. This counterexample is asserted, not hidden or repaired in the
reference. Changing sparse topology during optimization needs a separate policy.

Detector admission requires finite float32/64 or complex64/128 data and a finite,
nonnegative real tolerance. Meta/data-free, integer/bool/other precision and
nonfinite data are explicitly refused. Squeeze remains available on meta data.
Source NumPy/Numba can ignore a NaN when invalidating columns while its Torch
backend rejects the candidate: `[NaN,1]` demonstrates this difference. Native C
refuses nonfinite inputs rather than choosing one backend's behavior.

Source squeezing `(0,1)` axes with no protection attempts a scalar reshape and
raises; native squeeze removes the size-one axis while preserving the empty
axis. This mathematically explicit native boundary is documented and tested;
it is not claimed to reproduce the source's empty-shape failure. Empty networks,
missing/duplicate outputs and inconsistent dimensions remain validation errors.

## Resources and remaining scope

Squeeze/C emit **zero physical gates, measurements, work bits or classical
feedback slots**. The quantum gate-cost model is not applicable to these
numerical helpers. The example reduces four axis occurrences to one, with two
complex128 output entries (32 bytes of output storage, not process RAM).
Existing all-measurement-branch/workspace/append tests remain in regression;
this batch does not change the physical emitters or quantum resource models.

Q-0887.03 dependency evidence is extended without a duplicate capability ID.
Q-0887 stays open for A/D, fixed-point composition and the complete wide-network
trigger. G3 and integration-PR merge remain outside this batch.

Local validation: **130 new scenarios**, **2,146 related tests in 89.72 s**, and
**84 CPU/TN tests in 11.79 s** pass. The three new numerical owners plus example
cover **96 statements and 28 branches at 100%**. This coverage percentage does
not describe the entire existing contraction module. Strict mypy covers all
seven changed Python files, including the shared slicer and refactored R pass.
Black, Ruff, scope, architecture, internal IR, protected API and executable example
checks pass. Reference numerical warnings and the prior default-suite timeout
remain recorded; this is not a full-repository green claim. Implementation:
`680ace82`.
