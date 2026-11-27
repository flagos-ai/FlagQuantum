# Proposal 130: known-size quantum storage with symbolic interpretation metadata

Status: explicitly approved by the user on 2026-11-27. This contract precedes implementation; approval alone closes no source owner.

## Verified source-success/native-failure gap

Locked Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389` admits these actual raw-storage
operations while native binding refuses the descriptors:

| Descriptor | Unchanged source | Current native gap |
|---|---|---|
| QMontgomeryUInt(3, p) | Allocate SDK, Partition SDK and Split constructor succeed | The physical validator rejects symbolic modulus metadata. |
| QFxp(3, f) | Allocate and Partition SDK hooks succeed | The physical validator rejects symbolic fractional metadata. Original Split/Join constructors separately refuse this symbolic dtype. |
| BQUInt(3, p) | Allocate and Partition SDK hooks succeed | The physical validator rejects symbolic iteration bounds. Original Split/Join constructors separately refuse this symbolic dtype. |

The exact before-state is recorded in
`ftqc-reference-env/g2-word-symbolic-metadata-intake-20261127.json`.
The three-bit physical size is known; these are not unresolved wire dimensions.
The current guard requires every metadata value to be a Python integer even
when the operation only stores or rebinds raw quantum bits.

Original transport remains a separate observation. The source drops the
Montgomery modulus during serialization; symbolic assumptions can also be lost.
The original codec is not patched and a successful encode/decode alone is not
reported as a faithful object round trip. See
`g2-word-symbolic-metadata-roundtrip-20261127.json` in the reference evidence.

## Proposed contract

Admit a canonical logical quantum descriptor for raw physical storage when its
bit width is exactly known, finite, integral and nonnegative. Existing quantum
bit minimums, fixed-point sign constraints, field constraints and descriptor
parameter ordering remain authoritative. Zero-width words still require a real
caller wire for a physical circuit.

* Keep built-in integer widths on their existing fast path. Also resolve exact
  integral symbolic constants without free symbols, without floating-point
  approximation or rounding. Do not accept a genuinely unresolved width.
* Preserve original descriptor objects, symbolic expressions and parameter order
  in logical records. Use a concrete validation view and a resolved count only
  for arity/storage checks; do not rewrite the caller's metadata.
* Permit unresolved interpretation metadata, such as Montgomery modulus,
  bounded-integer bound and fixed-point fractional width, when it does not
  prevent exact physical sizing and existing logical validation succeeds.
  Canonical field descriptors may use this same rule only if their existing
  exact order/width relation proves a concrete storage size.
* Revalidate resolvable constants through existing descriptor constraints.
  Known-invalid widths, signed one-bit fixed point, inconsistent field orders,
  invalid concrete bounds and malformed metadata remain refused.
* Keep array extent admission unchanged. Symbolic array dimensions do not
  become physically executable through this proposal.
* Raw storage, views, ownership allocation/release and existing bit-level
  consumers remain the only newly admitted responsibility. Do not infer
  numerical modular/fixed-point/field arithmetic or external value admission
  from a storage success. Unknown interpretation parameters remain unresolved
  requirements for numerical consumers.
* Do not expand classical storage, quantum/classical reinterpretation,
  measurement, snapshots or classical writes. Proposal 118's mixed-domain
  physical refusal remains unchanged.

This supersedes the interpretation-parameter concreteness restriction of the
existing raw quantum storage profile. It does not supersede a requirement for
known physical dimensions or an actual arithmetic algorithm/body.

## Source constructor and control boundaries

Preserve original Split/Join constructor refusal when the source dtype is
symbolic. For example, raw storage for QFxp(3,f) may be available, while an
original Split(QFxp(3,f)) remains a source-refused constructor. Compiler-generated
word views may lower a storage operation without claiming that an unavailable
source Split decomposition exists.

The existing Cast factory must distinguish unresolved wire size from symbolic
interpretation metadata. A known-size symbolic-metadata Cast can use its owned
raw view; genuinely unresolved dimensions retain a declaration.

Preserve source-refused symbolic control specifications and the current
fixed-point, signed and field control restrictions. In particular, admitting raw
storage must not accidentally enable a control predicate previously rejected
because its source interpretation metadata is symbolic. Source-valid raw-bit
Montgomery controls with a concrete width may reuse their existing integer
predicate path, with both polarities and complete erasure evidence. This is not
numerical modular control or arithmetic.

## Ownership and implementation plan

Core owns logical descriptor validation and exact concrete size projection.
Reuse the current register/type and width helpers; no new public type, opcode,
schema field, registry, runtime or backend is introduced. Keep the concrete
classical validator independent from this quantum extension.

Compiler owns source-constructor qualification, symbolic-control refusals,
partition construction and finite binding. Audit callers of
`executable_register_width` and `concrete_logical_register_width` so the storage
extension cannot silently change numerical interpretation or control contracts.
Simulation and Runtime retain their existing primitive execution and codecs.
The consumer still uses `circuit.append(op)`.

Land the approved contract first, followed by domain-owned Core and Compiler
commits. The affected source owners include Cast, Allocate/Free, Split/Join,
Partition, AutoPartition and their direct control/OnEach dependencies. Source
parents are accepted only after their complete responsibilities pass; approval
alone closes none.

## Acceptance

1. Compare actual untouched source storage/SDK calls, ordered leases and aliases
   for all newly admitted profiles, including clean/borrowed and empty cases.
2. Verify complete complex operators/states with a phased, entangled spectator,
   zero clean work and restored borrowed state through ordinary append.
3. Preserve original symbolic Split/Join and control constructor failures,
   genuinely unresolved dimensions, malformed/known-invalid metadata and
   classical/mixed-domain refusals. Record source transport defects separately.
4. For comparable control circuits compare every primitive, operand and feedback;
   exhaust applicable small measurement branches. Keep primitive inventories
   separate from source analytical models and logical width estimates.
5. Verify existing artifact/alias/hash behavior, source-independent native
   execution, strict typing, formatting, scope, architecture, IR and API checks.
   No new direct logical-word serialization promise is made; bound primitive
   CircuitIR continues using its existing format.
6. Reconcile each affected complete-file audit, parent checkbox, stage overview,
   top status and PR evidence. Restore a reopened parent only after acceptance.

## Why explicit approval is required

Root AGENTS rule 8 prohibits changing protected accepted inputs and failure
behavior without an approved proposal. Core/AGENTS.md states: "Changes to
protected artifacts, IR, public schemas, or Stable Core APIs require an
integration contract/ADR change before implementation."

Proposals 118, 124 and 126 did not authorize this physical storage extension.
Proposal 126 explicitly retained physical refusal for unresolved fields; it
approved logical declarations only. This proposal changes physical admission
for known-size storage and therefore needs its own explicit decision.
