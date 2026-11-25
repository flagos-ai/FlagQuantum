# Proposal 125 — Nonbinary quantum field word storage

Status: explicitly approved by the user on 2026-11-25 with the instruction
"批准，按 Proposal 125 实施". Proposals 109 and 111 cover binary quantum polynomials and nonbinary
classical field words respectively; quantum field storage remains binary.

## Verified complete-file requirement

Frozen Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389` admits Split and Join on QGF(3,1),
QGF(5,1), QGF(3,2), and degree-one packed polynomials over those fields.
The original constructors, direct Cirq callbacks and full tensor operators are
unchanged. All twelve forward/inverse cases have exact identity operators,
preserve every raw wire and declare zero gates. Scalar widths are two, three
and four bits; polynomial widths are four, six and eight bits.

The existing neutral descriptors preserve the corresponding characteristic,
degree, order and polynomial metadata. Native physical binding refuses all
twelve cases because executable_register_width requires binary fields. Thus
closing approved Proposal 116's fixed-point profiles does not close the complete
Split/Join/Partition/AutoPartition dtype responsibility.

Evidence: `ftqc-reference-env/g2-nonbinary-quantum-storage-intake-20261125.json`
and its bounded execution log. This is an actual source-success/native-refusal
boundary, not a hypothetical field arithmetic requirement. Original QGF/QGFPoly
foreign serialization failures remain separate and will not be repaired here.

## Proposed narrow contract

Admit the existing canonical concrete finite_field and finite_field_polynomial
quantum descriptors as raw packed storage, including nonbinary characteristic.
Reuse the current structural validator already shared with nonbinary classical
words. Preserve positive built-in integer quantum width, zero classical width,
parameter order, characteristic/degree/order consistency, coefficient width,
packed polynomial extent and nominal identity. Existing logical descriptor
validation remains authoritative. Malformed, symbolic and noncanonical
physical descriptors remain refused.

Split, Join, Cast, typed calls, lifetime operations and packed views retain raw
MSB-first bits. Coefficient storage remains in the existing declared order.
Unused binary encodings in a non-power-of-two field may travel through a raw
quantum storage view, as in the unchanged original identity operator. External
numeric value creation and decoding retain their existing range checks and
errors. No numeric field algorithm, arithmetic, value codec, control predicate,
classical-word semantics or approximation contract is expanded.

No new dtype encoding, opcode, public entry point, field or serialized
vocabulary is introduced. Core owns executable admission. Compiler retains
typed composition, views and binding. Simulation retains numerical codecs;
foreign SDK objects remain at the existing Ecosystem boundary. Ordinary use
continues through circuit.append(op).

## Acceptance after approval

* Compare every literal direct SDK alias and full complex operator for both
  directions of all six source dtype profiles, including every unused raw code.
* Verify typed descriptor identity, shaped and packed views, round trips,
  coherent spectators, real ownership, zero private work and measurements.
* Exercise actual source/library paths where source serialization succeeds;
  preserve original QGF/QGFPoly codec failures where it does not.
* Verify independent source model estimates and primitive scans separately.
* Retain symbolic/malformed descriptor, classical-storage, numeric-range,
  binary-only arithmetic and unapproved field-control rejections.
* Run concentrated field/bookkeeping regression, strict typing, format, scope,
  architecture, IR and protected API checks; synchronize all progress ledgers.

Root AGENTS.md rule 8 protects admitted public behavior. Core/AGENTS.md states:
"Changes to protected artifacts, IR, public schemas, or Stable Core APIs require
an integration contract/ADR change before implementation." PUBLIC_API_PROTECTION.md
also protects accepted public types and failure behavior. Proposal 111 explicitly
keeps quantum admission binary; its approval does not authorize this extension.
