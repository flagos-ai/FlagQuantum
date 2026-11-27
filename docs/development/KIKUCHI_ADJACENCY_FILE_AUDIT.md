# Complete Kikuchi adjacency-list source mapping

The complete Qualtran v0.7.0 source
`optimization/k_xor_sat/kikuchi_adjacency_list.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389` maps to
`flagquantum.algorithms.kikuchi_adjacency`. Tests bind the untouched source and
original test blobs. All three classes, four factories, signatures, index types,
self-adjoint query declarations and direct dependencies are covered.

## Interfaces and ownership

S, k and T each contain ell big-endian unsigned index words. Their width is
ceil(log2(n)). Queries additionally carry one flag bit. The root owns ell clean
T words and a clean flag, calls the forward and reverse queries in signature
order, and records their release through the shared ComponentProgramBuilder.
The caller supplies typed FunctionRecord child bodies. Generic release records
ownership and does not establish numerical zero restoration.

The frozen queries expose no physical decomposition. Their declared multiset
includes controlled XOR, bit XOR, symmetric difference and its adjoint,
controlled increment and its adjoint, equality and its adjoint, conjunction,
measurement-assisted conjunction erasure and CNOT. Native models reuse the
Compiler arithmetic, comparison and symmetric-difference owners. Symbolic and
concrete operands, multiplicities and all seven QEC categories agree.

The original example query model has 28 Toffoli, 3072 CSwap, 6414 And, 71335
Clifford and 6414 measurements. The root model is twice this query cost.
These values are source QEC estimates. Emitted primitive counts, complex
amplitudes, measurement branches and work-zero claims require admitted physical
query bodies and remain open. Existing executable components retain their
`circuit.append(op)` bindings.

## Literal source wiring and remaining boundaries

The untouched root assigns `(S, T, k, flag)` from the reverse query's
`(S, k, T, flag)` result tuple. Its actual composite graph consequently returns
reverse k and frees reverse T. Both original composite edges and native typed
SSA results are checked explicitly. This differs from the documented conceptual
role of the reverse query, which should erase the old neighbor index. The
mapping preserves the frozen implementation. Correctness of the intended
neighbor algorithm remains unresolved.

The s parameter remains metadata, matching the source. The symbolic array shape
cannot enter either builder. The native root rejects symbolic composition and
mismatched child signatures. Identity child substitution verifies typed wiring
and ownership only. Numerical erasure and root Circuit lowering remain open.

The complete unchanged source suite records seven passes, nine original skips,
five original xfails and one separately tracked notebook deselection. Missing
query decompositions, symbolic array construction, model-only count checks and
dtype checks retain their original skips. Root count/decomposition and four
KXorInstance serialization failures retain their original xfails. Q-0602 and
Q-0603 remain active for these physical and source acceptance gaps. Q-0601 owns
the complete notebook separately.

## Validation

The concentrated regression passed 560 tests with four warnings. Strict mypy
for both Python files, Black, Ruff, Algorithms scope, architecture, multi-level
IR and protected API checks passed. The warnings remain visible in the batch
log. The separate unchanged source-suite outcome is recorded above.

## Complete-source closure review

Q-0602 is eligible for complete-file acceptance under the explicitly approved
[faithful-migration rule](G2_FAITHFUL_MIGRATION_CLOSURE_PROPOSAL.md). All three
classes, four example factories, DocSpec, signatures, self-adjoint query
identities, every direct resource call and the defined typed root have native
owners. The native root preserves the original positional assignment: return
reverse k and release reverse T. Sparsity remains metadata because the original
root does not consult it. Neither behavior is silently corrected.

The current dependency regression reuses complete source-call/operand/output
comparisons and every QEC category for both query classes, concrete/symbolic
instances and both graph directions. The actual neighbor root succeeds in both
implementations at Python ell zero, one and two. Exact SymPy Integer(2) and a
positive symbolic ell both fail original decomposition because their array
shapes are symbolic; native finite composition retains its refusal. No new
symbolic-array physical contract is inferred from Proposal 130.

The two query leaves have no original physical body. Their supposed no-junk
classical algorithms are prose, not an executable source decomposition. A child
identity used to inspect root wiring is not a quantum implementation or a
zero-work certificate. The original return/free alias defect, absent leaf
bodies and instance serialization failure remain separate source follow-ups.
They no longer impose an invented implementation prerequisite on faithful
migration. Existing real primitive owners retain circuit.append(op).
