# Approved raw quantum field storage and bookkeeping closure

Proposal 125 was explicitly approved on 2026-11-25. The preceding approved
Proposal 116 admits signed positive-width and unsigned empty fixed-point words.
This batch completes their actual bookkeeping dependency paths and adds raw
nonbinary quantum field/polynomial storage through the existing Core validator.
No public entry, instruction, dtype encoding or serialized vocabulary is added.
Users continue to append existing operations with circuit.append(op).

## Source and ownership

The reference remains unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0. Complete
Split, Join, Partition and AutoPartition sources were read with their direct
typed rebinding and storage dependencies. Native Core preserves canonical
characteristic, degree, order, polynomial identity, packed extent and MSB widths.
The same structural field validator already used for nonbinary classical words
now validates raw quantum words. Its obsolete private binary-only flag is
removed; classical semantics and validation are unchanged.

Compiler retains the existing Split/Join, coefficient views, Partition,
AutoPartition and physical binding owners. Simulation retains numerical codecs.
Before this change, some binary algorithms implicitly depended on Core's old
storage restriction. Explicit binary arithmetic checks now remain at addition,
Frobenius and the common field-word owner used by multiplication and inverse.
The binary numerical resolver and structured-control admission independently
retain their previous rejection, including empty shaped field-control groups.
This is storage admission, not nonbinary numerical arithmetic or controls.

## Exact raw storage and source failures

Twelve source/native directions cover GF(3), GF(5), GF(9) and degree-one packed
polynomials over each field. All complete source operators are exact identity
matrices, including unused binary code points. Native aliases match literal SDK
reshapes on reversed caller wires. Full complex vectors include every raw word
and a coherent spectator. Captured CircuitIR JSON executes while Qualtran, Cirq,
galois and Kickmix imports are blocked. There are no primitive gates, private
work, measurements or feedback events; the independent source QEC model is zero.

Packed polynomial views retain descending coefficient order and nominal dtype
while exposing the existing two coefficient words. Joining restores the original
packed alias. Scalar numeric decoding still rejects raw codes at or above field
order, as does the original decoder. Malformed, reordered, boolean, symbolic,
mixed quantum/classical and classical-only physical descriptors remain rejected.

Original QGF/QGFPoly foreign serialization still raises TypeError. A test first
tried to compare signatures through that failed original codec; it now compares
the actual source interface and field parameters directly, preserving the codec
failure separately. Source GFPoly tensor-index failures remain outside this raw
Split/Join identity proof. No reference codec, tensor or gate expectation is
repaired. Existing binary Program/JSON/hash compatibility fixtures remain exact.
Proposal 111's historical binary quantum limit is superseded only by the newly
approved raw-storage contract; its classical contract and fixtures are retained.

## Fixed-point and packed dependent calls

Actual source libraries at depths zero/one/two exercise signed fixed-point words
at positive and negative fractions and unsigned empty words at zero/negative
fractions. Typed Partition covers scalar, row-major multidimensional and empty
arrays. All metadata, aliases, full complex spectator amplitudes and zero-gate
view resources match. The old signed-target rejection assertions are replaced
by approved execution checks. The complete touched Partition test module is
strictly typed without skips or ignores.

Empty Split -> Join executes in a real H/T spectator context. Its full complex
operator matches at 2e-15 and retains the empty logical output. An initial fixture
serialized the composite root at depth zero, omitting its actual call graph;
depth one supplies that graph and resolves correctly. This was a fixture-depth
error, not a missing-empty-connection production bug. Standalone zero-wire source
tensor failures remain separate. The context emits only its caller's H and T,
with independent model one Clifford and one T.

Whole-array AutoPartition compares original ordered calls, operands, logical
aliases and full complex states in ordinary and left-only modes. Targets include
signed/empty fixed point and all three nonbinary fields. Views add no gates.
Nonempty targets use one actual CH per word; the independent source model has
four Clifford and two rotations per CH. Empty targets and zero-extent arrays
have no child CH. The real control wire remains live.

Signed-target two-control execution preserves both raw erasure histories against
the literal source and an independent compute/CH/uncompute operator. Absolute
unnormalized complex amplitudes, spectator coherence and clean work match.
Its source model and primitive scan are kept separate. Signed targets do not
admit signed controls. Pending Proposal 124's signed zero-width storage remains
refused, with the original zero-width SDK success tested separately.

## Scope and verification

Proposal 125's contract is
`contracts/nonbinary-quantum-field-storage-contract.toml`. Prior contracts,
schemas, enum values and binary artifact fixtures are unchanged. All changed
production/test modules receive strict typing, formatting and ownership checks;
architecture, IR, protected API, language and local links are checked. Final
concentrated regression, source suite, Cirq 1.6 and local integration outcomes
are recorded in the synchronized progress ledger after publication.

Four source-file parents retain unapproved or unresolved physical profiles,
source outer-LEFT finalization, generic/classical views and existing codec
boundaries where applicable. This batch closes the approved 116/125 storage
dependency, not unrestricted whole-file acceptance. Receipts use
`g2-approved-storage-bookkeeping-*20261125` and
`g2-nonbinary-quantum-storage-*20261125`.

The final batch passes 1,242 concentrated cases, 481 independent Cirq 1.6 cases
and 145 local integration cases. The final complete 25-case new storage owner
also passes after its direct polynomial SDK alias assertions are added. All
four unchanged source suites record 32 passes, sixteen original skips, one
original xfail and three notebook deselections; prior notebook evidence is
reused. Eight complete source/test blobs and Kickmix 0.1's native Simulator
are reverified. Strict mypy passes all fourteen changed Python files, alongside
Black, Ruff, approved Integration/Compiler/Simulation scopes, language, local
links, architecture, IR and protected API checks. The dependency scan retains
exactly the prior 720 findings, not a globally clean result. No baseline,
compatibility fixture or reference algorithm was weakened or rewritten.
