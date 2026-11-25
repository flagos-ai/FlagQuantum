# Complete Kikuchi block-encoding source mapping

The complete Qualtran v0.7.0 source
`optimization/k_xor_sat/kikuchi_block_encoding.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389` maps to
`flagquantum.algorithms.kikuchi_block_encoding`. Frozen source and test blobs
are checked. The two original BlackBoxKikuchi wrappers map to the native
KikuchiEntryOracle and KikuchiRowColumnOracle. The Hamiltonian class, both
factories, defaults, properties and composite call sites are covered.

## Interfaces and dependency ownership

The entry wrapper renames q/i/j to matrix q/S/T while retaining source result
ordering. The column wrapper declares a packed bounded unsigned ordinal l and
an unsigned index i. Both have ell*ceil(log2(n)) bits. Its child neighbor query
has two ell-element arrays of unsigned words. The source attempts to pass the
packed scalar bindings directly into those arrays.

The Hamiltonian retains system, ancilla and omitted zero-width resource ports.
System width is ell*ceil(log2(n)), ancilla width is system+2, normalization is
sparsity and declared precision is zero. Default sparsity is the number of
unique constraint scopes. Default angle precision is three bits when max_rhs
is symbolic or two. Other maxima require an explicit precision. Source
internal O_H/O_F properties map to matrix_entry/neighbor_query. The phase
precision property maps to phase_gradient_bitsize. Identity signal preparation
reuses PackedPrepareDeclaration and IdentityPrepare, including symbolic widths.

Typed entry-wrapper and Hamiltonian programs accept real caller-supplied child
FunctionRecords and use the shared ComponentProgramBuilder. Tests compare
untouched composite wiring, native SSA operands, output order, verifier results
and rejected incompatible signatures. Child identity substitution proves wiring
and ownership admission. It does not prove the underlying query algorithm.

The resource expansion reuses existing Compiler uniform-preparation and
Algorithms query owners. The existing exact controlled-add/subtract model now
also supplies inverse RHS and matrix-entry costs. The inverse has two fewer
Cliffords than the forward model. Forward/adjoint declarations are priced
separately. Source Always wrappers leave diffusion and both query pairs
uncontrolled. Only the system and flag swaps change under a positive control.

All seven QEC categories match untouched concrete/symbolic source queries.
The original concrete Hamiltonian model has 112 Toffoli, 13312 CSwap, 28268
And, 310479 Clifford, six rotations and 28268 measurements. Positive control
changes CSwap to 13345 and Clifford to 310446. These are source model estimates.
Emitted primitive counts and physical peak-work requirements remain open.

## Source failures and physical admission

The untouched packed column-wrapper decomposition raises IndexError when it
indexes a zero-dimensional scalar as an array. Native literal composition now
passes bounded unsigned storage admission and raises the same zero-dimensional
IndexError at the scalar-to-array binding. The earlier storage rejection is
obsolete; the actual packing failure is retained.
A physical packing contract and admitted query bodies are still needed.
The root and entry wrapper have valid one-child source composite graphs, but
their descendants retain the query/RHS physical gaps in the preceding audits.
Symbolic sparse decomposition also retains its original rejection.

The generic controlled-adjoint matrix-entry cost query reaches atomic C[And]
and fails in the source. Native model admission reports that unsupported path.
The Hamiltonian's positive-control expansion bypasses control on its entry
pair, matching the source, and its resource query passes.

Q-0608 remains active. Full complex-amplitude, measurement-branch and
work-zero acceptance awaits actual physical query bodies, correct source root
return/free semantics and packed-array admission. Existing executable shared
components retain their circuit.append(op) entry points. No new physical root
execution claim is made. Q-0607 separately tracks the complete notebook.

Q-0609 covers the original test file rather than a nonexistent physical query
body. Its complete assertion and raw protocol mapping is recorded in
[the four-file acceptance audit](KIKUCHI_KXOR_SOURCE_TEST_ACCEPTANCE.md).
The native source_sparse_calls inventory retains both swap widths, forward and
inverse query identities, and the original Always control bypass. Resource
pricing consumes that inventory. Concrete and symbolic controlled comparisons
now check all eight calls before folding; equal one-bit swaps combine with
multiplicity two. No primitive execution is inferred from this model graph.

## Verification

The complete unchanged original non-notebook suite records nine passes, two
original count-only skips, two original serialization xfails and one notebook
deselection. The native batch includes direct query, preparation, sparse
encoding and typed-builder dependencies, alongside strict typing, formatting,
lint, scope, architecture, IR and protected API checks. Final results are
recorded in the batch logs and synchronized ledger after publication.

The expanded dependency regression exposed an existing reference-harness wire
mapping error for padded Hermitian circuits. Its uniform-preparation helper
imported initial Hadamards with a zero-based compact wire order even when called
on offset physical wires. The helper now retains physical wire IDs. The source
circuits are unchanged and all 57 focused Hermitian/uniform tests pass, including
all seven cases that previously failed before comparison. This is a test-harness
correction with no production lowering change.

Strict checking of the touched existing uniform-reference test initially exposed
60 pre-existing diagnostics. The complete file now has typed fixtures and tests,
validated source operation/line-qubit types, explicit dynamic-circuit ownership
and a documented malformed-input boundary. Strict checking now passes for that
complete file without ignores or casts. Source operation trees retain their
order and the complete phase/measurement assertions remain intact.

The final integrated batch passed 848 tests with three warnings. Strict mypy
passed all six changed Python files across the dependency and root checks.
Black, Ruff, Algorithms scope, architecture, multi-level IR and protected API
checks passed. The earlier dependency-only run passed 618 tests.
