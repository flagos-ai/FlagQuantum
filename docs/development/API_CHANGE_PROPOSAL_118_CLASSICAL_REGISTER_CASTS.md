# Proposal 118 — Classical register storage views

## Review status

Status: explicitly approved by the user on 2026-11-27. Implementation and
acceptance remain tracked separately. Proposals 108, 110, 111 and 112 admit classical
word declarations, reads, feedback and writes. They do not authorize changing a
word's nominal descriptor while retaining its physical classical slots.

## Source requirement and current gap

Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0,
`qualtran/bloqs/bookkeeping/cast.py` accepts equal-width classical-to-classical
Cast. The unmodified CUInt(4) to CInt(4) callback maps raw words 0 through 15 to
0 through 7 followed by -8 through -1. Its direct Cirq hook returns no operation
and the same register object. This is a storage reinterpretation.

Native `cast_word_function` currently accepts executable quantum words and
CBit identity. It rejects the existing classical unsigned/signed four-bit
descriptors at executable quantum-width validation. Existing quantum
Split/Join cannot be used as a classical-word operation under its contract.
The evidence is retained by `test_cast_foreign_file.py` and
`ftqc-reference-env/g2-cast-foreign-*`.

Original CUInt serialization fails before native input at all three tested
library depths. This source transport error stays separate from the native
owned classical view gap. Neither an SDK object nor a patched source dtype is
required to implement the native operation.

## Proposed contract

Add the Program-only operation `cast_classical_register` using the existing
Operation representation, target `register_type`, register operand and register
result. It consumes one classical word handle and returns one handle with the
canonical target classical descriptor and the same ordered physical slots.

- Source and target must both pass existing logical classical descriptor checks.
- Their total raw bit widths must agree. Logical symbolic widths can be
  declared when equality is provable. Physical binding requires a concrete
  admitted width through the existing classical-storage owner.
- Cover all existing canonical classical word descriptor families, including
  CBit, unsigned, signed, ones-complement, bounded unsigned, Montgomery,
  fixed-point, finite-field and packed finite-field-polynomial storage.
- Preserve raw MSB-first bits without numerical encoding, decoding, write,
  measurement, reset, allocation or feedback instructions.
- Shaped component functions apply the same scalar view in row-major order,
  preserving SSA ownership and each element's slots.
- Keep earlier read snapshots immutable. Subsequent writes through the returned
  handle update its existing backing slots under the already approved write
  semantics. The consumed input SSA handle cannot be reused.
- Retain raw unused encodings internally, as approved for writes in Proposal
  112. External numerical value validation and decoding retain their existing
  descriptor-specific rules.
- Reject quantum-to-classical and classical-to-quantum views in executable
  Program. These continue to require existing explicit measurement or
  preparation operations.
- Reinterpretation is reversible by swapping the descriptors. It has zero
  primitive gate, measurement, workspace and modeled QEC cost. Nominal widths
  and physical slot inventories remain independently reportable.

Reuse `cast_word_function` and `circuit.append(op)` as consumer entry points.
Do not add a public export, vendor reflection registry, numerical kernel or
second classical runtime. Original source adjoint loss of shape/mixed-domain
opt-in remains a recorded source diagnostic.

## Ownership and compatibility

Core owns opcode vocabulary, operand/result type verification, linear register
ownership and the approved operation contract. Compiler owns shaped construction,
transforms, feedback snapshots and slot-preserving binding. Simulation and
Runtime retain their existing execution of the emitted instruction stream.
Ecosystem adapters remain optional boundaries.

This extends protected Program semantics and operation vocabulary and therefore
requires explicit approval under Core/AGENTS.md and PUBLIC_API_PROTECTION.md.
It preserves existing operation meanings, released schema fields, classical
read/write behavior, public exports and refusal of executable mixed domains.
Every reader of the Program opcode vocabulary must be updated in the same
integration batch. Unknown or malformed casts must fail during verification.

## Acceptance

Verify the untouched source callback and direct SDK identity for available
classical dtype families and both directions. Check raw-bit preservation,
shaped and zero-width cases where existing descriptors admit them, chained
casts, old read snapshots, conditional writes, linear ownership and malformed
width/domain/descriptor failures. Exercise the ordinary component binding and
append entry with coherent measured inputs and every small measurement branch.
Compare all complex branch amplitudes and clean work independently of classical
slot identity and the zero-resource model. Run strict typing, dependency, scope,
architecture, IR and protected API checks. Update the full Cast audit and G2
ledger. Proposal 116's signed positive-width and unsigned empty storage is
already approved and implemented. The additional signed-zero profile in
Proposal 124 and symbolic quantum-field declarations in Proposal 126 retain
their independent approval boundaries.

## Implementation record

The approved contract is `contracts/classical-register-cast-contract.toml`.
Core derives the target nominal type and enforces one linear consumption.
Compiler construction, scheduling, feedback wrapping, binding and gate-only
lowering recognize the operation. Physical binding forwards existing classical
slots and emits no cast instruction. See `test_classical_register_casts.py` and
the updated whole-file Cast audit for acceptance evidence.
