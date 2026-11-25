# Proposal 116 — Signed and empty fixed-point quantum storage

## Review status

Explicitly approved by the user on 2026-11-25. This proposal supersedes Proposal
115's signed and zero-width fixed-point storage rejection in the narrow profiles
below. Implementation and local acceptance are documented in
`SIGNED_EMPTY_FIXED_POINT_STORAGE.md`. Signed zero-width remains separate.

## Requirement and frozen evidence

The unchanged Qualtran v0.7.0 revision is
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0.
OnEach(3, XGate(), QFxp(3, 1, signed=True)) and the negative-fraction variant
successfully decompose into Split, three X calls and Join. Zero-width unsigned
QFxp targets at fractional widths zero and minus two successfully decompose into
Split and Join with no child calls. Native logical construction succeeds, but
physical binding refuses these descriptors under the approved Proposal 115.
The exact source steps and native errors are recorded in
`ftqc-reference-env/g2-on-each-fixed-storage-intake.json`.

## Contract and ownership

Extend the existing canonical quantum fixed-point storage admission to signed
positive-width words and unsigned zero-width words. Preserve every nominal type,
fractional width, sign flag and MSB-first raw bit. Reuse the existing descriptor,
Split/Join, typed call, physical binding and circuit.append(op) entry points.

- Quantum width remains a built-in integer, at least zero.
- Classical width remains the built-in integer zero.
- Parameters remain fractional_width followed by signed, with a built-in integer
  fractional width and an exact boolean sign flag.
- Logical fixed-point validation continues to determine the fractional-width
  bounds and minimum signed width.
- Symbolic, classical, malformed and noncanonical storage remains rejected.
- An empty register retains a logical binding and emits no primitive gates.
  A physical circuit continues to require a real wire, such as a spectator.

Core owns executable admission. Compiler retains component composition and
ownership. Simulation retains numerical decoding and arithmetic approximation.
This changes storage admission only. Control predicates continue to admit only
the already approved unsigned one-bit fixed-point controls from Proposals 113
and 114. Wider fixed-point controls and signed fixed-point controls retain their
current refusal. No operation, public entry, serialization vocabulary or
numerical arithmetic contract changes.

## Acceptance and compatibility

Compare original signed and empty unsigned OnEach targets with native signatures,
logical Split/Join bodies and exact primitive order. Verify complete complex
amplitudes with a spectator, both phase directions, actual adjoint restoration,
caller ownership and zero private work or feedback. Cover direct physical binding
and actual foreign-library resolution. Compare original resource models with
emitted primitives separately.

Exercise positive and negative fractional widths, shaped storage, identity,
creation and consumption where supported. Retain source constructor and codec
failures, malformed/symbolic/classical descriptor rejections and the approved
control refusal boundaries. Run focused and concentrated tests, local integration,
strict typing, format, scope, architecture, IR and public API checks.

Core/AGENTS.md and PUBLIC_API_PROTECTION.md require explicit approval before
implementing this accepted-input extension. Approval is recorded above; the
implementation reuses the existing shared storage validator.
