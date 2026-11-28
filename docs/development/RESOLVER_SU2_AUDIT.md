# Complete SU(2) constructor dependency closure

## Source and interfaces

The complete basic_gates/su2_rotation.py file and original tests are locked at
Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389.
Compiler formats/legacy_su2_factories.py owns the constructor boundary only.
The existing phase/su2_rotation.py owner supplies source angle formulas.
Existing scalar and Pauli rotation constructors own exact materialization and
symbolic translation. The original source remains unchanged.

The signature is one QBit THRU port named q. The source construction adds
GlobalPhase, Rz, Rx and Rz. It gives the scalar its default precision 1e-11 and
each rotation eps/3. The native builder retains all four typed calls, including
the isolated zero-port scalar. Dependency scheduling may move that scalar
relative to connected rotations, as it commutes with them. Ordered connected
rotations, operands, angles and scalar phase remain exact. Matrix conversion,
ZXZ conversion, adjoint parameters and original examples reuse the accepted
complete geometry/source-binding evidence. Symbolic angles use the approved
parameter-expression path and require concrete values at execution.

## Original composite serialization defect

The unchanged source serializer stores composite occurrences through wire
connections. Its zero-port GlobalPhase has no such connection. The serialized
composite includes its resource-count dependency but loses its occurrence.
The unchanged source deserializer reconstructs that graph without the scalar,
which changes the complex tensor. A regression records both original objects
and their different complex tensors, without editing the source or message.

Native composite resolution now rejects a nonzero counted zero-port dependency
that has no connected occurrence. Counts do not specify execution order, so the
resolver does not synthesize an occurrence from analytical resource counts.
A shallow wire message without counts cannot expose this defect. That format
limitation remains explicit. Named SU2 construction retains the complete source
constructor and correctly executes the scalar. The guard is an independent
compiler dependency commit and does not change the wire schema.

## Verification and resources

Tests compare the native typed predecessor/output graph with the original graph,
the source traversal's gate stream including scalar materialization, and the
original full complex matrix. Both directions, coherent spectators, original
standard examples, matrix/ZXZ constructors, complex64/128, nonlinear symbolic
angles and zero private work are checked. The normal entry is circuit.append(op).
No measurement is emitted, so there are no measurement histories to enumerate.

| Account | Observation |
|---|---|
| Actual nonzero-scalar primitive stream | Two Rz, one Rx, two X and two phase instructions |
| Actual private work | One scalar carrier, restored to zero and freed |
| Original analytical graph | One zero-cost GlobalPhase and three rotations, each with eps/3 |
| Original logical QubitCount | One logical data qubit, separately reported from the native carrier |

Zero scalar elimination follows the existing scalar owner. The source scalar is
atomic without a primitive decomposition. Actual scalar recipe counts and
analytical rotation-synthesis estimates remain separate. Original symbolic
rotation_matrix refusal remains preserved even though its symbolic decomposition
can bind and execute through the existing native parameter path.

## Remaining boundaries

This closes one complete constructor identifier, taking built-in coverage to
66/303 with 237 remaining. Generic ControlledViaAnd child dispatch,
Controlled/Adjoint/Power, arbitrary-target Pauli, partial OnEach, permissive
non-signed sign conversions, pending Proposals 116-119 and whole-repository/GPU/
performance acceptance remain open. Batch receipts use
ftqc-reference-env/g2-su2-constructor-*.

## Batch acceptance

The concentrated dependency regression passes 2909 cases. Local integration
passes 145 cases. The focused constructor and scalar-loss checks pass 15 cases
with full statement and branch coverage of the SU2 constructor owner. Unchanged
original tests pass ten cases with one original skip. Four files pass strict
typing, Black and Ruff. Compiler scope, language, architecture, IR and public
API checks pass. Dependency policy retains the identical 720-line recorded
baseline. Six frozen blobs and the real Kickmix 0.1 Simulator are verified.

## Source-controlled dependency closure, 2026-11-28

The opt-in `controlled_su2_factories()` tracks actual SU2 constructor records in
one compilation context. Its positive owner retains the literal source four-call
order: CRz(left), Controlled(Rx), CRz(right), then ZPow on the actual control.
Negative protocol controls reuse the original X/positive/X predicate wrapper.
Unknown same-signature children are refused. Ordinary `su2_factories()` remains
the existing mapping; shared parameter validation does not change its Program.

The source CRz children each expose two ZPow and two CNOT calls, all compared
recursively. Source Controlled(Rx) explicitly refuses Bloq decomposition because
Rx is atomic; its native crx atom supplies an independently verified numerical
action. The source has no deeper primitive stream to compare for that atom.
The scalar phase now acts on the existing caller control, without allocating a
carrier or expanding the pending scalar-phase proposals. Original per-rotation
precision and source GlobalPhase default precision retain their prior meaning.

Actual named constructors at depths0/1, both control polarities and both SU2
adjoint directions pass complete graph/operand comparisons and full complex
operators with reversed operands and a coherent spectator. SDK-free reconstruction,
normal append/JSON and coherent-state execution pass; nonlinear symbolic angles
bind through existing owned expressions. Invalid controls and unknown children
fail explicitly. Source and test blobs are checked against91283ba1822f1cc3c1b1cedc37a4dfa74ec21085
and61b4d09dbc41dac7590d4655a6b5ecb6c188228f, with Qualtran0.7.0 and realKickmix0.1.

For the generic non-Clifford row (.3,.4,.2,.1), positive emitted inventory is
5 phase,4 CX,1 crx:10 instructions. Negative adds2 X:12 instructions. Neither
uses work, measurement or feedback. Independent original model categories are
4 Clifford/6 rotation for positive and6 Clifford/6 rotation for negative; both
source/native logical peaks are2. These are separate inventory/model accounts;
no decomposition of the source-atomic Controlled(Rx) is claimed. The catalogue's
literal Controlled entry handles one positive scalar bit; source protocol zero
controls use ControlledViaAnd. Arbitrary explicit generic control specifications
are not inferred from signatures.

This closes the controlled-SU2 dependency of Hamiltonian GQSP. Q-0540 remains
open for the controlled GQSP call schedule and its source-selected double-walk
control recipes, followed by the five-call Always-aware Hamiltonian root. The
original Hubbard dirty-Free domain and independent mathematical operation remain
separate. No parent checkbox is changed by this dependency alone.
