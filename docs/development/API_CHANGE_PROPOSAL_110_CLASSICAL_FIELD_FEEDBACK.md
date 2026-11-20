# Proposal 110 — Classical binary-field and polynomial feedback

## Review status

Approved explicitly by the user on 2026-11-20 with the instruction
“批准，按 Proposal 110 实施”. Proposal 108 approved classical integer
words. Proposal 109 approved quantum binary-field polynomial ports. Their
accepted scopes expressly exclude the classical field ports proposed here.

The approved concrete scope is implemented. Acceptance and retained boundaries
are recorded in `CLASSICAL_FIELD_FEEDBACK_AUDIT.md`.

## Verified requirement

The complete frozen `qualtran/_infra/controlled.py` dispatches classical CGF
and CGFPoly controls to ClassicallyControlled. The existing native Simulation
codecs already produce the corresponding neutral descriptors and preserve their
field identity. Six direct witnesses cover GF(2), GF(4), GF(8), and degree-one
polynomials over each field. Active and inactive source callbacks agree with
native numerical callbacks. Current identity Program validation reports
`program.register_type`; classical reads and feedback construction reject the
same descriptors. Evidence is `g2-classical-field-intake.json` in the reference
workspace at base `1e74de50855c4ccb90146e4c068b65f6ce8d0ae2`.

The source is Qualtran v0.7.0, revision
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its controlled.py blob is
`6fabe760e793f1b39f825d76f80e92b6e5fa5c28`. The source classical wrapper has
no physical decomposition or QEC cost. Its value/tensor semantics and the
existing child primitive stream provide separate comparisons.

## Proposed contract

Admit the existing `finite_field` and `finite_field_polynomial` descriptors with
zero quantum width and canonical leading `("storage", "classical")` metadata.
Field metadata must satisfy the existing concrete binary quantum-field
validator after projection of classical width to quantum width and removal of
that storage marker. Polynomial metadata must satisfy the existing approved
Proposal 109 validator through the same projection. No numerical library is
required by Core.

Field width equals the positive coefficient degree. Polynomial width equals
`(polynomial_degree + 1) * coefficient_degree`, with concrete nonnegative
polynomial degree. Canonical parameter order, optional full polynomial identity,
nominal field identity and existing rejection of inconsistent descriptors apply.
Nonbinary fields and unresolved symbolic physical values remain separately
tracked. This proposal adds no class, root export, opcode or operation field.

The existing `read_classical_register` returns the identical typed successor
and MSB-first boolean snapshots, reusing Proposal 108's result convention.
Nested calls retain their existing ownership rules. Split/Join retain their
existing quantum-only operand restriction. Classical fields use owned reads. Arrays are
row-major, with highest-degree-first polynomial coefficients and MSB-first
coefficient bits. Reads allocate no additional classical slots. Existing
feedback consumes stored-bit activation patterns and preserves the field
metadata on control successors.

## Ownership and implementation order

1. Integration records explicit approval, a machine-readable contract and
   pre-extension compatibility witnesses for existing integer/quantum routes.
2. Core extends canonical classical admission using the current field and
   polynomial validators, including malformed descriptor tests.
3. Compiler reuses classical word storage, reads and feedback composition,
   adding only changes demonstrated necessary by the new descriptors.
4. Simulation reuses current codecs and activation callbacks. Ecosystem objects
   remain at their owning boundary.

The user's entry remains `circuit.append(op)`. Independent domain commits and
one concentrated regression batch preserve the current integration workflow.

## Acceptance

Verify scalar, shaped and empty arrays, exact field identity, read arity,
snapshot order, successor ownership, nested calls, incompatible nominal types
and malformed metadata. Compare stored-bit activation with the original
classical callbacks for all small field values and polynomial words. Compare
full complex states and spectators for active/inactive feedback, include
measurement-produced controls and all applicable histories, and verify clean
work and classical output bindings.

Record actual child primitives and classical initializers separately from the
source wrapper's absent physical model. Reuse existing child source models
without presenting them as a wrapper model. Verify unchanged integer and
quantum payload compatibility. Complete strict typing, formatting, scope,
architecture, IR/API checks and local integration before publishing and updating
the ledger.
