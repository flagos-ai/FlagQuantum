# Contiguous triangular-index whole-file audit

Q-0190 and Q-0191 cover the complete
`arithmetic/conversions/contiguous_index.py` and its original test file at
Qualtran v0.7.0 `bcb17bb25360ed68ce3bc084645db3e929799389`.
Both frozen Git blobs are checked. The source defines one class with widths,
cached signature, scalar callback, wire symbols and direct cost graph, and one
factory with its documentation specification.

## Complete source responsibility and ownership

The actual factory declares four-bit mu and nu and eight-bit s, all through
unsigned words. Its scalar callback returns nu*(nu+1)//2+mu and preserves both
input objects. It has no s input argument despite the through-s signature.
Compiler `registers/sorting/contiguous_index.py` owns finite-word execution and
the independent source resource declaration. Simulation
`classical_contiguous_index.py` retains the exact original scalar/array
operation order, NumPy promotion and overflow, and symbolic values without
imposing finite-word masking.

The native square/add implementation and Appendix F column-compression
implementation both XOR the finite result into an occupied target and restore
all caller-owned work. Their semantics, complete complex states, small
measurement branches and dependency streams reuse the existing
`test_contiguous_index.py` evidence. Neither root has a frozen source gate
sequence to compare, because ToContiguousIndex has no decomposition. Direct
shared primitives retain their own original-source comparison evidence.

Drawer `chemistry_symbols.contiguous_index_wire_symbol` retains the original
mu/nu labels and the exact historical output label with nu-1. The scalar
callback and executed arithmetic use nu+1. This discrepancy is tested at
mu=1, nu=2, where the actual result is four and the printed expression yields
two. The drawer preserves source presentation and does not determine execution.
Its existing typed score callback compares the complete factory score.

The factory, DocSpec membership, all three dtypes/sides, complete source model,
missing decomposition and through-s callback error are checked directly.
The generic physical importer reports that this opaque source primitive cannot
be decomposed. Native computation remains available through ordinary
`circuit.append(op)` with bound finite-word operands.

## Direct dependencies and consumers

Square/add, exact adjoints, weighted column compression, XOR and temporary And
remain in their existing Compiler owners. Raw scalar evaluation remains in
Simulation. The source resource root is consumed by the existing single-
factorization inner PREPARE and THC chemistry PREPARE models. A replacement
test changes the former's index declaration only and confirms the same consumer
uses it at width two and output width four. Existing full PREPARE resource
tests are included in the concentrated regression.

The original test file contains two functions. The factory autotester runs
construction/resource checks and skips decomposition, count comparison and
dtype checking requiring a decomposition. Its five-bit BloqBuilder test retains
4*29 T-equivalent cost. The untouched original file reports three passes and
three skips. No frozen reference is edited.

## Distinct resource accounts and retained gaps

The four-bit factory declares nineteen Toffoli over sixteen logical ports.
Existing five-bit square/add execution records 304 T/TDG, 772 CX, 76
measurements and thirty work wires. The independent Appendix F path records
116 T/TDG, 342 CX, 29 measurements and 33 work wires with one reusable classical
slot. The source five-bit model declares 29 Toffoli. These are separate accounts.

Q-0190 and Q-0191 retain missing original decomposition, the through-s classical
callback mismatch, original drawing-sign discrepancy, generic physical capture
and source skipped modes. Existing scalar versus occupied-target XOR contracts
remain documented rather than treated as identical functions. The new drawer
function adds presentation only and does not change arithmetic or resource
semantics. Batch logs are `ftqc-reference-env/g2-contiguous-index-file-*.log`.

The concentrated batch passed 209 cases without skips. Twenty-four existing
NumPy overflow warnings occur in paired original/native raw-scalar diagnostics.
Both changed Python files passed strict typing, formatting and lint. Compiler
scope, architecture, multilevel IR and protected API checks passed. Existing
THC and single-factorization model consumers and chemistry/grid symbol tests
were included.
