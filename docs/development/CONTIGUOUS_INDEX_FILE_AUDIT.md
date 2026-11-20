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

## Native stored dependency closure, November 20, 2026

The complete frozen declaration and original test file were revisited with
their direct square/add dependencies. `contiguous_index_component.py` now
retains the seven executable native calls in an existing Program. The order is
square, Add(nu), Add(mu), XOR into s, exact Add(mu) adjoint, exact Add(nu)
adjoint and square erasure. MSB-first views divide the even numerator by two
and restore its original layout before erasure. All three entry ports remain
THRU. Low-result truncation and an untouched wider prefix preserve the existing
native occupied-target XOR contract.

`arithmetic/integer_square_component.py` owns the square's THRU input and
RIGHT result, with LEFT consumption for its inverse. The result's physical
storage is allocated once across the index calculation. The reusable column
owner stores the existing paper-defined compression sequence as shared And,
measured And erasure and CX calls. It allocates a single work block and frees
it after reversing products and carries. This retains fixed work-wire bindings
across independent product scheduling. Source Square contributes a numeric,
sided-output and model contract, while its frozen root has no decomposition.

`bits/bit_compression.py` supplies the shared logical sequence to both the
existing flat emitters and the typed graph. Exact pre-batch comparison covers
184 square/sum-of-squares, integer-product and column-index outputs without
normalization. Widths reach 12 for squares, 8 for products and column indexing,
with up to four square inputs and both flat measurement slots 0 and 9.

Stored/expanded typed columns, square and index preserve the flat primitive
multiset and each wire's dependency order, operands and feedback at the tested
widths. Independent gates can have a different global schedule. Fresh typed
measurement SSA slots are renamed to the flat reusable slot only after checking
that every conditional CZ uses the immediately corresponding measured slot.
The original And and adjoint leaves are also compared directly against the
unchanged frozen source. Root-level upstream primitive parity is unavailable
for ToContiguousIndex and Square because neither declares a decomposition.

Square checks include all forward/inverse history pairs through width two and
both homogeneous measurement histories at width three, with every complex input
amplitude, entangled spectator, clean scratch and consumed result restored.
Index checks include complete coherent input words at widths one through three,
occupied results, spectator coherence and zero work. All 16 measurement histories
at width one are checked independently for both directions. Larger index
histories reuse the existing flat column-channel evidence and matching primitive
dependency order. A normal append round trip replaces both flat root emitters
with raising stubs. Source through-s callback and printed-label differences,
foreign capture and original skipped modes remain open under Q-0190/Q-0191.

At width five the square/add index emits 304 T/TDG and 76 measurements with
30 peak clean work wires, including its 10-bit numerator. The native typed path
uses 76 fresh measurement slots. The source root model declares 29 Toffoli.
Its independent Appendix F flat path remains at 116 T/TDG, 29 measurements and
33 work wires. At width eight Square emits 224 T/TDG, 56 measurements and 56
scratch wires, plus its 16-bit logical output. The source model is 56 Toffoli.
These primitive, storage and model accounts remain separate.

No parent acceptance or capability count is added. This closes the concrete
native stored dependency; generic foreign source construction and the original
missing decompositions retain their centralized ledger entries. Receipts use
`g2-contiguous-component-*`.

Batch acceptance includes 1364 concentrated passes without skips and one
existing IPython-directory warning. The 17 local integration checks pass.
The unchanged original index tests report three passes and three source skips.
All seven Python files pass strict typing, Black and Ruff. Integration and
Compiler scopes, architecture, IR and protected API checks pass. Five frozen
source blobs, Qualtran 0.7.0 and Kickmix 0.1 Simulator are verified. Dependency
policy retains 687 whole-repository findings and zero batch findings. The prior
default-suite docstring failure is reused separately, with its nine implicated
modules verified byte-identical to the recorded baseline. That evidence is not
a fresh whole-suite pass.
