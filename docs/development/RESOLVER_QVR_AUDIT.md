# Stored quantum variable rotation constructor audit

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
error. A negative rotation count can produce a logical root, while physical
signed storage remains pending Proposal 116. Symbolic direct coefficient or
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
logical domains. Signed physical storage remains pending Proposal 116.
