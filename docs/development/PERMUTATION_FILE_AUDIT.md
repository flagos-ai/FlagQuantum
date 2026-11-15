# Basis-state permutation whole-file audit

Q-0213, Q-0214 and Q-0215 cover the arithmetic permutation notebook,
implementation and tests at Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The original three Git blobs
match the frozen inventory. The new example catalogue and notebook reuse
the existing native emitters and resource owners.

## Ownership and contracts

Compiler owns big-endian register binding and ordered cycle emission.
The target x is a THRU bounded unsigned register. Each concrete cycle
compares its current entry into a marker and uses the marker to control
the XOR delta to the successor. Its final equality cleans the marker.
One marker, n-1 carry wires and n-1 reusable feedback slots are caller
owned. Every quantum work wire is zero at the end of each cycle. The
feedback slots retain the latest outcomes. Outside-cycle words are
unchanged. Overlapping cycles retain their declared order. The inverse
reverses cycle order and reverses each cycle.

The dense and partial cycle discovery owners preserve the locked order.
For the insertion-ordered partial map {0: 1, 2: 0}, the locked traversal
returns only (0, 1), losing the 2-to-0 edge. Reversing insertion order
produces (2, 0, 1). The original witness and broader partial-map checks
remain in `test_permutation_source_audit.py`. The checked native concrete
entry requires a positive register width, a fitting bound and a nonempty
sequence of legal cycles, with distinct values within each cycle.

| Original definitions | Native disposition |
|---|---|
| Two cycle converters, both cached signatures/default widths and symbolic detection | The example catalogue retains concrete tuples and the dimension of shape-only declarations, bounded x and the exact bit-length expression. Foreign constructors, general converter admission and foreign serialization remain open. |
| PermutationCycle executable body and graph | Existing `permutation_cycle_ir` retains marker allocation semantics, all equality and controlled XOR dependencies, singleton work and final cleanup. Graph evidence now compares the concrete constants, shape-unknown value, marker declarations, polarity and multiplicities for all three factories. |
| Permutation factories, build and graph | Dense/partial factories reuse ordered traversal. Example graphs retain ordered concrete cycles, the unknown-count worst-case length N and four indexed symbolic lengths. Generic from-cycle-lengths filtering and generic symbolic executable admission remain open. |
| Eight example factories and two documentation specifications | All original factory signatures, exact cycle data/shape, direct edges and leaf counts at depth one compare against the frozen source. Display uses native names, graph keys and signature tables. |

## Original tests and native evidence

All seven original functions were reviewed. Their modes cover eight
factory autotesters, six direct decomposition cases, concrete cycle and
multi-cycle tensor matrices and graph counts, sparse classical simulation,
and both symbolic count assertions. The unchanged source reports
37 passed, 8 skipped and 6 xfailed. Skips involve symbolic decomposition,
counts available through declarations only, dtype checks requiring
decomposition, and bounded-to-plain unsigned edges entering equality.
The six xfails concern source example serialization.

The native tests reuse the original exact concrete gate, operand,
measurement feedback and resource evidence recorded for Q-0214.01–.04 in the per-item ledger. Existing permutation
tests compare full complex states with an entangled spectator, exercise
each small measurement history, verify zero work and restore the full
initial state. Existing independent shape resource tests retain logical
cost and peak-width formulas. No executable gate recipe is changed by
this batch.

| Accepted scenario | Actual emitted subset | Independent model |
|---|---|---|
| Four-entry eight-bit cycle | 233 CX, 140 T/TDG, 35 measurements, 8 clean work wires, 7 reused slots | 35 logical And, 35 measurements |
| Ordered overlapping three-bit cycles | 101 CX, 56 T/TDG, 14 measurements, 3 clean work wires, 2 reused slots | 14 logical And, 14 measurements |

Primitive CX and source Clifford categories have separate accounting.

## Teaching asset and closure

The original 24-cell notebook and the native 26-cell notebook execute in
independent kernels. The native kernel denies reference imports. It
retains all eight examples, original teaching topics, both signature
groups and both displayed one-level call graphs. The final section runs
the existing coherent `circuit.append(op)` forward/inverse workflow.

Q-0213 closes. Q-0214 and Q-0215 retain the order-sensitive partial-map
witness, generic shape and foreign admission, strict source dtype checks
and original serialization boundaries. Symbolic declarations and costs
are available separately from concrete gate lowering.
