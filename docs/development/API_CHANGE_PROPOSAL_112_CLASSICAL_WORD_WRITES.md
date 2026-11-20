# Proposal 112 — Owned classical word writes and conditional feedback

## Review status

Approved explicitly by the user on 2026-11-20 with the instruction
“批准，按 Proposal 112 实施”. Proposals 108 through 111 admit classical
reads and equality feedback and separately exclude writes.
This proposal records the additional protected Program vocabulary and semantics
before any implementation. Core owns the contract and Compiler owns lowering.

## Verified requirement and source chain

The unchanged Qualtran v0.7.0 revision is
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The complete `bloqs/mcmt/classically_controlled.py` delegates classical execution
to `_ControlledBase.on_classical_vals`, which selects the child callback using
the original control values and returns the original controls alongside the
selected child outputs. A classical child can update its THRU classical port.

`ftqc-reference-env/g2-classical-write-intake.json` records both activation
polarities and all eight control/data cases for a reference-only CBit flip.
The original callback passes. Native `classically_controlled_ir` rejects an
existing `ClassicalXor((1,), 1, True)` child with the current classical-write
refusal. The intake base is `b18de0d848e670b221b474c9dd1a1bc48e8b584c`.
The reference source remains unchanged.

Core already owns a dynamic `ClassicalXor` assignment using old source values.
Its parser and execution paths support a single `conditions` conjunction.
Compiler already owns Boolean computations, classical word bindings, immutable
read snapshots, nested stored calls and condition composition. A typed owned
word replacement needs an explicit successor contract in this existing model.

## Proposed Program contract

Add one Program-only opcode, `write_classical_register`, through the existing
`Operation` record. Its operands are one owned classical register and exactly
its concrete storage width of Boolean SSA values. Its sole result is an owned
register with exactly the same nominal type. The register operand is consumed
linearly. Boolean operands remain reusable immutable snapshots.

The replacement word uses MSB-first Boolean order. Polynomial coefficients
retain highest-degree-first order, with MSB-first words within each coefficient.
Arrays continue to use the existing row-major scalar register binding. Empty
arrays contain no scalar writes. A zero-width unsigned scalar has no Boolean
operands and returns its same-type zero-width successor.

The operation accepts existing admitted concrete classical storage descriptors,
including CBit, integer, fixed-point, Montgomery, field and polynomial words.
Unsupported quantum or symbolic physical descriptors fail during verification.
Operand count, Boolean kind, register type identity, ownership and dominance
are verified through the existing Program authorities.

Existing `condition_clauses` express an optional disjunction of conjunctions.
If the condition holds, the result carries the replacement bits. Otherwise it
carries the old bits. Every condition and replacement operand reads its value
before this operation. Reads taken before a write keep their original values
through subsequent writes, nested calls, and feedback wrappers.

### Storage representation and numerical domains

This operation replaces raw storage bits. It preserves the nominal descriptor
and does not apply arithmetic reduction, field normalization, saturation or
modular conversion. For bounded integers and nonbinary fields or polynomials,
a replacement can contain an unused numerical encoding. Internal storage reads
continue to expose that exact raw pattern. Equality with a valid activation
compares those bits and therefore leaves an invalid pattern inactive.

Existing external input and activation admission keeps its value and coefficient
range checks from Proposals 108 through 111. Numerical decoding retains the
owning codec's validation and rejects an invalid representation wherever that
codec currently rejects it. This proposal introduces no numeric interpretation
of unused encodings. Acceptance must exercise an internal GF(3) raw word `11`
and a bounded unsigned word at its exclusive bound, distinguishing internal
bit storage, external admission and numerical decoding. Any consumer that
cannot preserve this distinction must reject that path explicitly.

## Conditional control and immutable storage

An unconditional typed write can bind the successor to the immutable replacement
slots without emitting a dynamic assignment. A conditional write allocates fresh
successor slots, copies the old bits and performs guarded assignments from the
replacement snapshots. Sources and conditions retain their previous slots.
Multiple satisfied DNF clauses assign the same stable replacement values and
are idempotent. Lower each conjunction with the existing guarded `ClassicalXor`
spelling. Its metadata parser needs no new DNF form.

The flat `classically_controlled_ir` helper admits classical XOR children whose
write targets are disjoint from the wrapper's control slots. It conjoins the
existing child conditions with the wrapper equality using the shared condition
normalization. Contradictory guards emit no write. Attempts to overwrite wrapper
control slots fail before execution, preserving the source's entry activation.
Sequential reads and writes among child-owned data slots keep their existing
ordered assignment meaning.

The stored feedback transform conjoins its entry snapshots with typed writes
and existing quantum primitives. Pure Boolean computations remain unconditional.
Nested calls keep their declared ownership and shared callee transformation.
Caller control handles return unchanged. Private bookkeeping must preserve
caller-owned values and verified lifetimes.

## Ownership and compatibility

1. Integration records approval, source intake and exact existing artifacts.
2. Core adds the Program opcode admission, type/linearity/condition verification
   and backward reading through the existing records and serialization owners.
3. Compiler adds owned replacement binding and stored/flat conditional writes,
   reusing the existing classical XOR, word encoding and condition authorities.
4. Existing Runtime and Simulation guarded-XOR execution remains authoritative.
   A missing execution guarantee requires its own domain-owned correction under
   this contract and explicit evidence before acceptance.

The existing `circuit.append(op)` entry transports the compiled component.
Measurement, reset and channels retain their existing admission policies.
This extension changes Program vocabulary and the documented classical-write
refusal for the admitted paths. Old read-only, quantum, CBit, integer and field
artifacts must retain exact representations, bytes, hashes and slot allocation.
Older readers may reject the new Program opcode under their existing unknown
opcode policy. New readers must retain all previously admitted artifacts.

## Acceptance

Reuse accepted read/control evidence and compare all original CBit callback
cases. Verify complete source callback results, both control polarities,
contradictory and overlapping DNF clauses, self-source flat assignments,
multiple sequential writes, unchanged entry controls, nested calls, scalar and
shaped words, empty arrays and zero-width unsigned storage. Exercise all admitted
word descriptor families and invalid external and internal representations.

Check immutable snapshots before and after active and inactive writes, including
snapshots used as subsequent quantum feedback. Compare full complex spectator
states, applicable measured histories, clean workspace and append execution.
Replace former execution helpers in the normal path to verify ownership of the
new path. Register malformed-operation, illegal alias, undefined-source and
control-target-overlap rejection cases.

Report dynamic assignments, fresh classical slots and emitted quantum primitives
separately. Conditional width-w replacement emits w initial copies plus w
assignments per retained conjunction, while unconditional immutable rebinding
emits zero assignments and zero fresh slots. Resource checks must verify actual
lowering, including normalized or contradictory clauses. The original classical
wrapper has no physical decomposition or root cost model. Child callback parity
and native physical measurements remain distinct evidence.

Complete-file acceptance of `ClassicallyControlled` and its inherited dependencies
requires a final interface audit. This proposal alone accepts no parent item.
