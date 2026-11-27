# Stored quantum variable rotation constructor audit

The [2026-11-28 complete-file review](QVR_FILE_AUDIT.md#current-complete-file-disposition-2026-11-28)
supersedes the historical declaration-only decision for symbolic scalar inputs
with finite source rotation indices. It preserves the literal source type
dispatch and ordinary typed phase owner. Unknown indices, symbolic storage
Split failures and original SDK/codec failures retain explicit limitations.

## Scope and ownership

The complete frozen `rotations/quantum_variable_rotation.py` constructor family
contains `QvrZPow` and `QvrPhaseGradient`. `QvrInterface` is abstract. Both
constructors retain the caller's `cost_reg`, `gamma=1.0` and `eps=1e-9`.
The MSB-first fixed-point cost is THRU and caller-owned. Gradient QVR adds a
caller-owned fractional THRU `phase_grad`. Existing phase sizing determines its
width. The format adapter owns serialized arguments and constructor dispatch.
Shared phase owners retain exact graph construction, scheduling and clean
workspace lifetimes. Runtime binding and ordinary `circuit.append(op)` execute
the resulting Program. Classification is `single_device_fast_path`.

Frozen Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389` stays unchanged. The numerical
sizing search and fixed-point coefficient encoding reuse accepted native owners.

## Complete direct dependency disposition

Direct QVR retains Split, original weighted Z powers and Join. Gradient QVR
retains its original single scaled-add call. The shared root helper accepts
both executable and declared children, retaining caller names and signatures.
Symbolic gamma uses the accepted symbolic sizing model and retains the
original successful one-call root with a declared scaled child.

Negative fractional widths already satisfy the unsigned fixed-point storage
contract. Scheduling and sizing now admit that domain, with exact binary
scaling for the original truncation search. A scaled schedule can include a
zero-width Add input. The component retains this identity call and its gradient
port, while physical lowering emits an empty stream and needs zero workspace.

Original one-bit equivalent descriptors use the existing compatibility checker.
For larger incompatible full-word descriptors, the successful QVR root retains
a declared scaled child. Original negative-width slices similarly retain their
child refusal. These cases fail original depth-two serialization with the
original dtype or shape exception. Depth-zero and depth-one successful roots
are compared independently.

## Explicit boundaries

Signed direct QVR with positive rotation count raises the original indexing
error. A negative rotation count can produce a logical root. Approved Proposal
116 now admits its positive-width signed physical storage, as verified below.
Symbolic direct coefficient or
precision also reaches an original indexing refusal. Such constructors retain
typed declarations. Zero rotation count retains the original division refusal.
Gradient signed input rejects at construction. Concrete gamma with symbolic
precision cannot form the original fixed-point template. Symbolic shapes and
non-THRU costs remain declarations with concrete execution unresolved.

Previously recorded default-precision error witnesses and aggregate rotation
model discrepancies remain open upstream observations. They are preserved by
existing source-file tests. Full repository, GPU and performance acceptance are
outside this local batch.

## Verification

The concentrated batch compares named signatures, original root operands and
serialized depths zero, one and applicable two. Accepted physical cases retain
identical primitives, events and outputs under reversed wire bindings. Complex
tensors are compared against unchanged original contraction. New one-bit
negative-fraction cases exhaust every measurement history, compare unnormalized
complex amplitudes and prove all work amplitudes outside zero vanish. Existing
QVR and scaled-add evidence covers spectators, inverses and ordinary append.
Actual primitive inventories and original resource models are recorded in
separate receipt fields after regression.

## Acceptance receipt, 2026-11-24

The concentrated regression passes 303 cases, including two additional direct
negative-fraction source probes performed before restoring their existing test
file. The retained constructor suite covers the same direct formats. Affected
local integration passes 145 cases. Unchanged original QVR tests pass 85 cases
with three expected failures and one dense 4096-dimensional test deselected.
The original file fixtures verify frozen source blobs and actual Kickmix 0.1
Simulator availability. Strict mypy checks ten touched Python files. Black,
Ruff, team scope, language, architecture, multi-level IR and API checks pass.
Dependency analysis matches the same 720 existing findings exactly.

| One-bit cost, gamma 0.375, eps 0.2 | Actual primitives and wires | Independent original model |
| --- | --- | --- |
| Direct, fraction -1 | 1 phase, 1 wire | 1 rotation, 1 logical wire |
| Direct, fraction -2 | 1 phase, 1 wire | 1 Clifford, 1 logical wire |
| Gradient, fraction -1 | 9 CX, 3 H, 2 T, 2 Tdag, 1 S, 1 measurement/CZ/reset, 4 wires | zero gates, 3 logical wires |
| Gradient, fraction -2 | 1 CX, 2 wires | negative one Toffoli, 4 logical wires |

The last source estimate retains the original negative small-width formula.
It is an upstream model boundary. Neither its gate count nor its logical-wire
estimate is substituted for the executed primitive inventory. Resource receipt
`g2-qvr-constructor-resources-20261124.json` retains both columns separately.
Both constructor identifiers are resolved within their original successful
logical domains. The historical signed-storage refusal is superseded by the
approved Proposal 116 follow-up below.

## Approved signed-storage dependency closure (2026-11-25)

The old signed-QVR test still expected physical binding to fail after Proposal
116 was implemented. The original tiny-coefficient path succeeds, and the same
native constructor now executes; that stale negative assertion failed before
this correction. Production QVR algorithms and frozen source remain unchanged.

Actual source QFxp profiles `(width,fraction)` of `(2,0)`, `(2,1)`, `(3,2)` and
`(4,-2)` with gamma `1e-5` and epsilon `0.2` all emit Split, one sign Z and Join.
Source libraries at depths zero, one and two resolve identically. Typed nominal
metadata, exact root call order, reversed physical aliases and source tensor
operators match. Full complex states retain an entangled spectator, ordinary
`circuit.append(op)` and exact adjoint restoration. No private work, measurement,
feedback, release or binding swap is needed.

| Signed format | Actual body | Original analytical QEC declaration | Source logical qubits |
| --- | --- | --- | --- |
| 2 bits, fraction 0 | 1 phase(pi) | -8 rotations, 0 Clifford | 2 |
| 2 bits, fraction 1 | 1 phase(pi) | -9 rotations, 0 Clifford | 2 |
| 3 bits, fraction 2 | 1 phase(pi) | -9 rotations, 0 Clifford | 3 |
| 4 bits, fraction -2 | 1 phase(pi) | -4 rotations, 0 Clifford | 4 |

Negative original synthesis tolerances and counts remain in reference metadata.
They are upstream model defects, not negative measured gate inventories. The
sign operation preserves the original phase exactly; no ideal mathematical QVR
is substituted. The gamma `0.375` positive-count paths still raise the source
IndexError and retain nonexecutable native declarations. Signed gradient QVR
still rejects at construction. Original signed one-bit QFxp also rejects before
QVR construction. Symbolic and malformed cases retain their earlier scope.

The existing implementation parent stays open for its broader precision/model
and source boundaries. No constructor or parent denominator changes here.

The concentrated constructor/rotation/gradient/decomposition/storage/cost-phase
regression passes 173 cases; the independent Cirq 1.6 constructor and direct
rotation selection passes 58. Strict typing of the complete changed test file,
Black, Ruff, scope, language, links, architecture, IR and API checks pass.
Receipts use `g2-signed-qvr-storage-closure-*20261125`; the earlier stale-test
failure and source one-bit constructor refusal remain diagnostic history.
