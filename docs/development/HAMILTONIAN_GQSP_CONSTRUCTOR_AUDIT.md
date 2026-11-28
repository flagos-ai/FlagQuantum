# Hamiltonian GQSP constructor ownership

Reference: complete unchanged Qualtran v0.7.0
`hamiltonian_simulation/hamiltonian_simulation_by_gqsp.py`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

## Interface recorded before implementation

The outer signature is the complete GQSP signature, including caller-owned
`signal`. PREPARE selection ports are a named subset of that signature.
PREPARE junk ports are separate, privately allocated zero words. Root order is
allocate junk, PREPARE, GQSP, exact inverse PREPARE, free junk. No signal or
selection output is discarded. Packed words remain MSB-first and shaped words
remain row-major. Every supplied Program retains its helper and workspace
ownership; an inverse is explicit, never inferred by reversing primitives.

The Compiler constructor consumes the actual resolved walk, time and precision.
An injected owner supplies the complete derived GQSP Program, PREPARE pair and
explicit junk names. Numerical polynomial, normalization and phase work remains
outside Compiler in the existing Simulation owners. This is the same explicit
context boundary as the walk constructor; shallow walk ports cannot identify
PREPARE or infer its inverse. Missing context fails closed.

Finite composition requires matching quantum THRU PREPARE ports and their exact
inverse, a THRU GQSP, disjoint private junk, and equal shared descriptors. Native
private allocations use those existing typed descriptors. This does not assert
that arbitrary PREPARE junk is clean after GQSP: source nonzero-junk Free
counterexamples remain open and must not be hidden by tracing out work.

## Closed finite constructor and dependencies

The Compiler component composes complete Programs, imports their exact helpers,
and records all private lifetimes using the existing builder. The format factory
validates the source walk/time/precision fields, preserves its exact walk ports
under the leading signal bit, and requires explicit derivation context. The
Ecosystem adapter reuses the existing HamiltonianGQSP numerical model: its degree,
Jacobi-Anger sign, finite-grid normalization, Issue860 margin, verified complement
and GQSP precision split remain unchanged. It never uses flat emitters to replace
calls or imports SDKs during resolution. Model time and precision must match the
actual constructor arguments. Association of a PREPARE normalization and exact
walk implementations remains the explicit binding owner's responsibility.

Original serializer depths 0/1/2 retain the actual custom small-walk fields.
All root operands and predecessor identities compare directly to the unchanged
stored source graph, including allocation and final free. Positive and negative
time, degrees one and seven, complete complex operators, reversed wire placement,
a coherent spectator, all applicable measurement histories, zero private work,
source-free construction and SDK-free append/JSON execution pass. The native
numerical injection independently reproduces the original P/Q and precision;
a replacement binding owner gives identical Programs. Invalid metadata,
signatures, duplicate/unknown junk and incomplete children fail closed.

A separate unchanged random matrix walk has no PREPARE junk. Its source object
array serialization fails with allow_pickle=False; live typed signature capture
and finite execution are separate capabilities. That instance's measured
reflection path compares every small history and runs through DynamicCircuit
trajectories. No matrix or gate synthesis is inferred from constructor ports;
the supplied exact child definitions are captured by the existing SDK boundary.

## Physical inventory and declared estimates

Three independently recorded rows are in
`g2-hamiltonian-constructor-resources-20261126.json`. Original five-child declared
models match hamiltonian_gqsp_source_graph. Existing native walk, prepared
reflection and GQSP cost owners reproduce all seven source QEC categories.
These calculations do not count or price the physical instruction stream.

| Time | Precision | Degree | Actual instructions | Actual measurements | Actual private peak | Declared seven-category estimate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.15 | 0.05 | 1 | 45 | 0 | 2 | 22 Clifford, 9 rotations, 2 Toffoli; other categories zero |
| -0.15 | 0.05 | 1 | 45 | 0 | 2 | 22 Clifford, 9 rotations, 2 Toffoli; other categories zero |
| 0.75 | 1e-5 | 7 | 249 | 0 | 2 | 130 Clifford, 45 rotations, 14 Toffoli; other categories zero |

All leases return. Declared native peak is five logical qubits, while actual
private peak two is measured separately with three caller-owned quantum inputs.
Reserved pool capacity is not reported as live workspace. Scalar carrier gates
and their source zero-cost logical phases retain different accounting bases.

## Retained boundaries

The complete source Hamiltonian module and direct source/test dependencies were
read and five frozen blob identities reverified. Existing full original
42-case test disposition and random-projection receipts are reused. Original
symbolic Hubbard declarations retain model evidence, not numeric execution or
serialization claims; this finite context adapter does not implement a symbolic
numerical model. Large Hubbard matrices were not constructed.

The original Hubbard outer temp nonzero probability 0.016501741620346923 remains
a counterexample to unconditional clean Free. This constructor is accepted only
with exact child ownership and a demonstrated clean domain. It does not replace
the existing caller-owned-junk operation, repair that original defect, or trace
out nonzero work. Q-0540 and whole serializer parents Q-0866/Q-0876 remain open.
Original QSP grid/root defects are not waived. Registration count is now203/303,
not203 accepted constructor modules. Pending118/119/124/126/127 remain unchanged.

## Batch validation

145 concentrated cases pass without skips in each actual Cirq1.7.0/Qiskit2.5.0
and Cirq1.6.1/Qiskit2.0.0 environment. All145 local integration cases pass.
Four files pass strict typing, Black and Ruff; both scopes, English/links,
architecture, internal IR and protected API checks pass using this worktree.
Qualtran0.7.0 and actual Kickmix0.1 Simulator are asserted. Publication receipts
are recorded in the external G2 ledger. No CI wait, dependency installation, ReadTheDocs, G3 change
or PR merge is part of this batch.

## Literal wrapper and symbolic composition closure, 2026-11-28

The shared root now retains the actual `Always(PREPARE)` and
`Always(PREPARE.adjoint())` wrappers, using the existing bookkeeping component.
Allocation, signal/selection ownership and final Free remain source ordered.
The former global rejection of nested declarations is removed; physical binding
still refuses an actually missing reachable body. Explicit inverse ownership and
THRU signatures remain mandatory. The wrapper alone is not a certificate that an
arbitrary generic control transform preserves Always semantics.

The actual unchanged `_symbolic_hamsim_by_gqsp` example now passes the five-call
root comparison and both complete 35-call Hubbard PREPARE directions. Only its
symbolic GQSP, which the original explicitly refuses to decompose, is a bodyless
leaf. The actual serialized root resolves with an explicit context; its reachable
Program equals direct composition up to function names. Transport references are
logical declarations, not substitutes for available executed PREPARE bodies.
Numerical emission is correctly refused at symbolic GQSP. Symbolic assumptions
and original codec quirks retain their separate recorded boundaries.

The existing finite small-walk full complex operators, all applicable measured
branches, private-zero checks and three separate physical/model rows are reused
and rerun against the wrapper change. Q-0540 remains open for complete actual
finite Hubbard context and source-selected controlled composition, including
unconditional Always ownership. The original outer temp residual probability
0.016501741620346923 remains an explicit dirty-Free counterexample; no release
of dirty work is accepted as clean execution. No source patch or new public API
is introduced by this batch.
