# Sparse chemistry SELECT constructor and dependency closure

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0.
Source/test blobs are `1c7fd5595aa87a6c647070cffb43d941a2d7cc9f` and
`4afa771527bfa51d37cc7a85f64b15fd7471db4c`. Fixtures verify both and require
Qualtran 0.7.0 plus the real Kickmix 0.1 Simulator.

## Source file and authoritative owners

The complete `chemistry/sparse/select_bloq.py` constructor is bound to a native
typed graph: S(flag_1b), flagged Y(p,alpha), flagged X(q,alpha), unconditional
Y(r,beta), unconditional X(s,beta). All four children reuse the accepted
Majorana owner and its exact directional programs. Named sys storage maps to
each child target. Inputs are scalar MSB-first bounded orbital/spin/flag words
and opaque system storage, all caller-owned. Children own and clear only their
private accumulators and measured predicates.

Compiler now owns the signature and ordered Majorana stage list. The existing
Algorithms signature entry and appendable source execution reuse those
authorities, removing their duplicate metadata and stage list. The separate
existing mathematical controlled operation keeps its behavior. No public
signature, serialization schema or Core operation changes.

Formats resolves the concrete constructor and exact generic adjoint with
scoped native recipes. Ecosystem assembly ends Majorana gate payloads inside
its Cirq directory; SDK objects do not enter Compiler or native execution.
The original source example at four spin orbitals and actual Walk selector
at eight are covered, along with the non-power-of-two six-orbital case.

## Verified scope

Forward/inverse, absent control and both source control polarities match
recursive calls, dependency direction and named operands. Depth-zero and
depth-one original libraries resolve, bind and execute through
`circuit.append(op)` with both source SDK modules unavailable after capture.
Representative superpositions cover orbital, spin/flag, target and outer
control sectors, comparing full complex amplitudes and private work zero.
Bounded addresses are valid input domains; padded illegal bounded-register
encodings are not asserted to have a clean source action.

For the four-orbital uncontrolled case, both directions enumerate all 1024
raw measurement histories on the complex witness, in addition to the shared
all-history erasure certificate. Larger cases use zero/one representative
histories and the certificate. Accepted Majorana full-operator and local
And feedback evidence is reused. This is not a newly evaluated dense operator
over every eleven- or nineteen-bit root input.

Mixed grouped controls 10 and 010 retain the original predicate call,
source-specialized child and exact erasure. Generic inverse after grouping
uses the inverse of that actual specialized child; it is not replaced by the
different source recipe for controlling an already-adjoint root.

The concentrated batch has 128 passes, plus a final source zero-width boundary
case. Local integration has 145 passes. The unchanged original suite has five
passes and its existing counts-check xfail. Strict typing covers six files;
format, scope, language, architecture, IR/API and the unchanged 720 dependency
findings are checked separately. The final factory rerun also verifies the
Cirq assembly relocation required by architecture policy.

## Explicit source defects and limits

For control_val=0 or 1, the original external control connects directly from
LeftDangle to RightDangle. It never enables the five source body calls.
Native source binding preserves this defect; it does not silently implement a
different controlled unitary. Existing separately named mathematical control
execution remains distinct.

Controlling a generic source adjoint follows a different recipe from generic
adjoint after grouping. The former reaches an undecomposable Controlled(And)
in the unchanged source. Direct source physical decomposition reproduces that
failure, and native binding explicitly refuses the unsupported recipe.
The latter source route is executable and has independent call and numerical
evidence. The source control/adjoint protocols must not be treated as freely
interchangeable here. Further control of an already-controlled constructor
requires its explicit shared helper context; this factory does not guess one.

At two spin orbitals the source root declares five calls but recursive
Majorana decomposition asserts on a zero-width spatial axis. Its resource
models also fail. Native metadata remains available through the unchanged
signature entry, while this typed executable recipe refuses. The existing
Algorithms mathematical single-orbital extension is unchanged and is not
reported as an upstream decomposition.

## Resources and next dependency

Eighteen directional/control inventories in
`g2-sparse-select-resources-20261125.json` distinguish emitted primitives,
private/live peaks and original QEC/peak estimates. Four orbitals emit
40 T-family, 96 CX, 30 H and ten measurements, with three private/fourteen
live wires. Source models separately report ten And, 77 Clifford, ten
measurements and peak fourteen. Eight orbitals emit 104 T-family, 232 CX,
78 H and 26 measurements, with four private/23 live wires; the source model
reports 26 And, 157 Clifford, 26 measurements and peak 23. Model gates are
not emitted primitive counts. Lease release and numerical cleanup proofs
are independent.

The frozen constructor inventory includes SelectSparse, so this source-supported
concrete constructor closes one identifier: 118/303, 185 remaining. Original
source failures above remain recorded. Complete sparse chemistry and THC Walk
constructors remain open, especially their RIGHT QROAM PREPARE/isometric
dependencies. G2 continues in the same integration PR; no G3 or merge follows.


## Complete source-file closure, 2026-11-28

Q-0410 now closes the complete file: constructor fields and control validation,
selection/control/target/signature properties, five ordered Majorana calls,
call graph, specialized control protocol, example and documentation declaration.
The accepted constructor, exact Majorana dependencies and source resource rows
above are reused. Source control remains an actual bystander and generic
controlled-adjoint decomposition remains unavailable; neither is repaired in the
reference or relabeled as a corrected controlled unitary. The distinct native
mathematical control remains separate. Current concentrated acceptance includes
this whole owner and both supported source control polarities, with original
small all-history evidence reused. The source zero-width failure stays recorded.
Sparse PREPARE and retained-bank Walk closure is now documented in their audits;
THC dependencies are a separate next batch.
