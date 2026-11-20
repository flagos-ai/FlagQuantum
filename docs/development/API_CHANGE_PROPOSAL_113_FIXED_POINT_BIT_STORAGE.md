# Proposal 113 — Executable single-bit unsigned fixed-point storage

## Review status

Approved explicitly by the user on 2026-11-20 with the instruction
“批准，按 Proposal 113 实施”. Proposal 103 logical
register prerequisite 1.20 deliberately preserves fixed-point physical refusal.
This proposal extends that admission only for the concrete source-supported
single-bit storage dependency identified in the complete controlled-file review.

## Unchanged source requirement

The reference is Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`.
`ControlledViaAnd(XGate(), CtrlSpec(QFxp(1, fraction, False), activation))`
decomposes for fraction 0 and 1, both activation bits and scalar controls.
The call chain extracts the encoded activation bit, invokes the existing
single-controlled child and surrounds negative activation with X gates.
`ftqc-reference-env/g2-fixed-point-control-intake.json` records all four successful
source streams and the current native executable descriptor refusals.

For fraction 1, the upper stored bit represents the numerical value 0.5.
Native activation patterns use the stored integer bit 1. Simulation retains
numerical encoding ownership. This distinction must be explicit in tests.
The original QFxp constructor rejects signed width one. Original CtrlSpecAnd
rejects wider fixed-point controls. Those refusals are retained independently.

## Contract and ownership

Core `executable_register_width` admits existing canonical quantum fixed-point
RegisterType records exactly when quantum width is the built-in integer 1,
classical width is the built-in integer 0, parameters are the canonical
`fractional_width` followed by `signed`, fraction is the built-in integer 0 or 1,
and signed is exactly False. Reuse the existing constructor/logical validator.
Descriptor identity and serialized spelling retain their existing representation.
Malformed and symbolic metadata follows existing rejection semantics.

Compiler maps this admitted control word to the existing unsigned stored-bit
ControlRegister pattern. The actual bit is MSB-first. Existing typed Split/Join,
calls, binder and exact adjoints preserve the nominal descriptor. They emit no
extra gates for the type views. The existing single-bit specialization and
negative-control conjugations own the emitted primitive stream.

Classical fixed-point reads and writes retain their independently approved
contracts. Wider quantum fixed-point descriptors retain logical graph admission
and existing physical refusal. Their CtrlSpecAnd source refusal remains a
negative witness. There is no new numerical cast, method, opcode, field,
serialization vocabulary or dependency. The entry remains `circuit.append(op)`.

## Acceptance and compatibility

Compare original decomposition calls, exact gates, operands and activation
encodings for fraction 0 and 1 and both polarities. Verify scalar and singleton
array control paths and separately preserve original matrix-shape defects.
Verify complete complex operators, an entangled spectator, exact inverse,
output aliases, zero work and resource counts. Positive activation emits one CX,
negative activation emits X/CX/X, matching source Clifford counts 1 and 3.
Run through the real typed append path and replace the previous physical helper
to confirm execution uses the shared owner.

Retain invalid signed-width-one constructor witnesses, wider/symbolic physical
refusals and wider-source CtrlSpecAnd refusals. Reuse exact existing integer,
CBit and field artifacts. Existing logical fixed-point tests must retain the
wider descriptor refusal they assert. Strict typing, formatting, scopes,
architecture, IR/API checks and concentrated regression precede publication.
This proposal alone supplies no complete controlled-file acceptance.
