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

## Complete actual Hubbard dependency adapter, 2026-11-28

`ecosystem/_hubbard_walk_component_library.py` now captures the actual square
Hubbard walk into existing native PREPARE/SELECT, select-block, reflection,
control-view and exact-adjoint owners. It adds no numerical kernel, Core type,
export, synthesis algorithm or private replacement registry. Source signatures
and named control views are retained, including the original set-derived order.
Geometry and ordinary-source ownership are checked before conversion.

Both original finite and symbolic Hamiltonian examples compare all four ordinary,
adjoint, positive-control and positive-control-adjoint walk graphs, 118 reachable
source/native node pairs per graph. Available children all have bodies. The
existing ModAddK and scalar carrier acceptance boundaries remain explicit.
Original specialized SELECT adjoint serialization refuses SpecializeOnCtrlBit.ONE;
its live exact inverse is supplied by the existing native owner, without changing
the codec or reference. Finite source angles retain the accepted coefficient owner.

The actual finite factory has degree127. Its native composed Hamiltonian has
2921 definitions and matches all563 reachable source nodes above the accepted
atomic boundaries, including every one of255 exact SU2 definitions. This is an
interoperability case: source coefficient payloads are captured at the SDK
boundary and native Simulation computes phases. Independent native Hamiltonian
coefficient numerics retain their prior acceptance receipts. This does not imply
a complete serialized symbolic coefficient codec or a whole physical operator
certificate for the defective private-Free source domain.

Separate emitted inventory/model accounts for the finite original walk:

| Direction | Emitted instructions | Emitted measurements | Declared source model | Source/native logical peak |
|---|---:|---:|---|---:|
| Ordinary and adjoint | 963 each | 28 each | 16 CSwap,26 And,87 Clifford,8 rotation,26 measurement | 26/26 |
| Positive control and its adjoint | 1008 each | 31 each | 16 CSwap,29 And,90 Clifford,8 rotation,29 measurement | 26/26 |

These inventories include existing native scalar carriers and primitive lowering.
They are not the original model categories or proof that released reflection temp
was zero. The 0.016501741620346923 dirty-Free counterexample and the independently
validated caller-owned-junk mathematical operation remain separate.

This completes the actual finite Hubbard dependency/context gap recorded above.
Q-0540 stays open for the remaining literal controlled-Hamiltonian composition:
its Always-wrapped preparations must bypass outer controls while the actual GQSP
receives them, with explicit source-selected owners. Missing controlled execution
is not relabeled as an upstream defect. No G3 or automatic PR merge follows.

## Source double-positive walk control composition, 2026-11-28

`doubly_controlled_select_walk_pair` in the existing Compiler preparation owner
retains source nested positive controls as separate per-child reductions. The
SELECT-block root calls its packed multi-controlled SELECT once. The walk then
calls the packed multi-controlled reflection. Reverse source order is reflection
then the SELECT-block adjoint, with its exact supplied SELECT inverse. A single
whole-walk predicate would have a different available source graph and is not
substituted here. Actual outer/inner scalar names and packed array order agree.
The existing source declaration treats reflection as self-adjoint; it remains
the same identity in both recipes, with no inferred arbitrary-junk cleanup proof.

Actual Hubbard forward and inverse each match124reachable source/native nodes,
including all available control packing, predicate and erasure dependencies.
For a clean independent small SELECT/reflection, the complete complex operator,
every small raw measurement history and zero private work agree in both
orientations. The existing FlagQuantum small test oracle now reports that its
controlled specialization already has current_ctrl_bit=1 when controlled again.
That repair is to this repository's test asset, not to the frozen Qualtran
reference; the upstream generic/specialization helpers remain unchanged.

Both actual Hubbard directions emit1042instructions and33measurements with18
caller logical qubits. Source models separately declare16CSwap,31And,92Clifford,
8rotation and31measurement. Source and native logical peaks agree at28. Native
lowering and source category accounts retain their different measurement counts;
these are not interchangeable inventories. Whole Hubbard dirty-Free behavior
remains the original counterexample. The clean small-oracle proof does not
replace that source domain or certify its released temp.

The next remaining Q-0540 responsibility is the controlled GQSP root schedule:
combine the accepted controlled SU2 calls, nested positive inverse walk and the
source mixed-polarity forward-walk predicate, then compose the Always-aware
Hamiltonian root. The source mixed predicate is CtrlSpecAnd, not the packed
specialization used for the double-positive inverse branch. No parent checkbox
is changed by the present dependency delivery.

## Complete source closure and controlled execution, 2026-11-28

The remaining controlled composition is implemented. The ordinary root is
unchanged. `controlled_gqsp_component_program` preserves the outer control,
controlled SU2 owners, double-positive inverse walk and mixed-polarity forward
walk. The mixed branch retains CtrlSpecAnd and the existing specialized positive
walk; the inverse branch retains separate packed SELECT/reflection reductions.
The scoped Hamiltonian constructor binds these explicit contexts to the actual
resolved walk, time and precision. Both Always preparations bypass the outer
control, including the zero-polarity protocol wrapper.

| Complete source responsibility | Native owner and acceptance |
|---|---|
| Normalization presence, alpha, time/precision and signature | Existing constructor/context validation and original source constructor receipts |
| Degree and finite approx_cos | Simulation Jacobi-Anger and new shared Hamiltonian coefficient owner; exact source arrays, both time signs, zero time, cached-degree reuse |
| Symbolic approx_cos | Same owner returns the original shape expression; no fabricated coefficients or executable empty body |
| Verified complement and phase recursion | Existing Simulation root and phase owners; native P/Q/phases derive independently of reference values |
| GQSP and half synthesis precision | Existing scoped GQSP factory and controlled root with explicit directional children |
| Private junk and Always preparation pair | Existing Hamiltonian component; allocation, two uncontrolled Always calls and final Free remain source ordered |
| Positive and negative default controls | Scoped Hamiltonian constructor and existing control predicate owner; full available source graph and clean execution |
| Concrete and symbolic original examples | Degree127 concrete graph:9901definitions/3120compared source nodes; symbolic declarations preserve the source refusal to execute |
| Declared costs | Existing source-model owners and original model receipts; emitted primitive inventories below remain separate |

192 concentrated cases pass independently with Cirq1.7 and Cirq1.6, including
all changed owners, existing hierarchy controls, ordinary Hamiltonian numerical
and projection paths, source constructors, measured GQSP and SU2 controls. The
clean independent SELECT fixture verifies complete complex operators, every
small raw measurement history, zero private work, SDK-free append/JSON and
DynamicCircuit execution for both polarities. It supplies a generic composition
certificate, not a replacement numerical claim about the actual Hubbard model.

The large comparison checks the complete Program once, then retains immediate
child bodies and deeper signatures per local traversal. Immediate Allocate/Free
bodies are necessary for source scheduling; dropping them was detected by the
large graph comparison and corrected. Full-module validation remains mandatory.
Physical binding still requires bodies for every included definition, including
unused declarations. Unused owner inputs that are never imported are distinct.

| Fixture | Control | Emitted primitives | Emitted measurements | Declared source model | Source logical peak |
|---|---:|---:|---:|---|---:|
| Clean small SELECT | 1 |149|5|2Toffoli,5And,43Clifford,20rotation,5measurement|7|
| Clean small SELECT | 0 |151|5|2Toffoli,5And,45Clifford,20rotation,5measurement|7|
| Actual Hubbard,t=.002,precision=.01 |1|2441|71|42CSwap,65And,217Clifford,42rotation,65measurement|30|
| Actual Hubbard,t=.002,precision=.01 |0|2443|71|42CSwap,65And,219Clifford,42rotation,65measurement|30|

Under the approved faithful-migration rule, Q-0540 is ready for whole-file
acceptance after the local strict/architecture gates and verified publication.
This supersedes the earlier staged open notes. Frozen source/test hashes and
original source-test receipts remain unchanged. Retained upstream boundaries
are the Issue860 finite-grid bound, root-conditioning failures, the actual
Hubbard private-Free residual0.016501741620346923, symbolic-assumption transport
loss and specialized SELECT adjoint enum serialization refusal. Controlled Rx
is an original atomic leaf: no deeper source primitive stream is claimed.
No whole actual Hubbard complex-amplitude or zero-work certificate is asserted.
