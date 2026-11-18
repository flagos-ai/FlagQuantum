# In-place sorting source-file responsibility

Q-0204 tracks the complete locked `lists/sort_in_place.py` blob
`d4da3727ad3ce11294c140870f2dbf4e1cb0f9fd`. Its single class owns two
fields, signature, index dtype and a direct call graph. The documentation
specification has no examples. The source has no root decomposition or
classical implementation.

## Native ownership and interface

`sorting/in_place_sort_declaration.py` owns the full original metadata and
resource query. `in_place_sort_signature(length, dtype)` retains both `xs` and
`pi` as THRU arrays and a bounded unsigned index of ceil(log2(length)) bits,
exclusive bound length. This follows the actual signature, despite the source
prose describing a RIGHT ancilla. Native dtype metadata preserves unsigned,
signed and opaque words, including symbolic widths and list lengths.

`source_in_place_sort_cost(length, width)` composes the original Comparator,
Comparator adjoint and Xor calls, each repeated 3 length squared. Comparator
reuses the existing GreaterThan dependency and declares width CSwap calls.
QubitCount's declaration-only fallback takes the maximum root/child width.
`source_in_place_sort_resource_formula` composes the existing symbolic comparison
fallback, whose counts differ from concretely bound widths in the original.
Both models have no primitive stream or measurement execution.

The existing `in_place_sort.py` owns actual distinct unsigned-word sorting and
its exact inverse. Callers bind big-endian words, clean rank words, clean work
and reusable feedback slots. Its existing append scenario verifies coherent
amplitudes with a spectator, forward permutation output, backward recovery and
work cleanup. The dependency gate comparisons retain exact operands and feedback.
No emitter or reference implementation was changed in this batch.

| Three words of two bits | Independent account |
| --- | --- |
| Existing native emitter, per direction | 684 T/TDG, 1608 CX, 108 measurements, 32 clean work |
| Original declaration | 108 CSwap, 162 logical And, 1998 Clifford, 162 measurements, 12 model peak wires |

The source root supplies no gate-level stream against which to assert whole-root
primitive equality. Dependency comparisons remain the applicable evidence.

## Verification and remaining boundaries

The new audit checks the exact source blob, empty DocSpec, all three direct
callees and multiplicities, native signature/type/shape/side metadata, concrete
QEC and peak queries across positive lengths and widths, and symbolic formulas
for separately symbolic length/width and signed/opaque metadata. Existing complete
permutation, sort, branch, coherent append and inverse suites are reused in the
concentrated regression.

The declared model accepts every positive length, independently of the physical
unsigned distinct-word domain. It does not certify signed executable sorting,
occupied rank words, duplicates, generic foreign construction, serialization or
protected static/symbolic execution admission. Zero-length source QubitCount
raises a wrapped math-domain error. These unadmitted execution and diagnostic
boundaries remain open under Q-0204, even after the full declaration is covered.

Batch verification passed 718 concentrated native/reference cases with no skips or warnings.
Changed Python passed strict mypy, Black, Ruff, Compiler scope, architecture,
multilevel IR and protected API checks. Supplemental compare/exchange and
shared-control results are recorded in `g2-lists-file-control-dependencies.log`.
