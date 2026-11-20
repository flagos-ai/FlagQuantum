# Proposal 111 — Nonbinary classical field words and feedback

## Review status

Approved explicitly by the user on 2026-11-20 with the instruction
“批准，按 Proposal 111 实施”. Proposal 110 separately covers binary fields.
The approved extension follows the contract-first ownership sequence below.

## Verified requirement and source chain

The complete frozen `qualtran/_infra/gate_with_registers.py` delegates control
selection to `controlled.py`, which routes classical field controls through
`ClassicallyControlled`. The existing Simulation codecs support nonbinary
fields and polynomials and already produce neutral register descriptors.

The unchanged Qualtran v0.7.0 source revision is
`bcb17bb25360ed68ce3bc084645db3e929799389`. Controlled source blob is
`6fabe760e793f1b39f825d76f80e92b6e5fa5c28`.
`ftqc-reference-env/g2-nonbinary-field-intake.json` records the intake at native
base `717f2b39dec59af49f3d1d1dfecad1e78df67a1e`.

Six fixtures cover GF(3), GF(5), GF(9), degree-one polynomials over GF(3)/GF(5)
and degree-zero polynomials over GF(9). Every legal activation/input pair agrees
between unchanged source and native numerical callbacks, totaling 902 pairs.
Scalar field tensor blocks agree. Identity Program validation, feedback reads
and stored control construction reject all six native descriptors.

### Original polynomial tensor discrepancy

Nonbinary polynomial integer conversion uses radix equal to the coefficient
field order. Physical bit encoding concatenates fixed-width coefficient words.
The source tensor active-space helper indexes by the former. Over GF(3),
`2*x+2` has polynomial integer 8 and packed word 10. The original tensor applies
X at word 8 and identity at word 10, while the numerical callback activates on
`2*x+2`. Over GF(5), `4*x+4` has polynomial integer 24 and packed word 36.
The intake preserves both active and inactive block counterexamples for each.

The extension follows the existing codec and numerical callback for typed
field values. Source tensor counterexamples remain independently asserted.
The original classical wrapper also lacks a physical decomposition and model.
Unchanged-source mathematical or primitive equivalence through those absent or
inconsistent protocols requires a separately stated comparison scope.

## Proposed contract

Admit canonical concrete classical `finite_field` and
`finite_field_polynomial` descriptors with a leading `storage=classical` marker.
Field metadata keeps its existing canonical key order and optional complete
polynomial identity. Characteristic p is an integer >= 2, degree d is positive,
order equals p**d, and coefficient width is `(order-1).bit_length()`.
Polynomial width is `(polynomial_degree+1)*coefficient_width`, with nonnegative
concrete polynomial degree. Quantum width is zero.

Core validates structural metadata and canonical widths with Python integers.
Numerical field construction, primality and polynomial irreducibility remain
with the existing Simulation codec and its numerical dependency. This preserves
the existing separation between neutral metadata and mathematical construction.
No new Core numerical dependency is introduced. Quantum field admission retains
its independently approved binary contract.

Compiler receives built-in unsigned stored integers. A scalar field input must
be less than its declared order. Every fixed-width polynomial coefficient must
be less than the same coefficient-field order. Unused raw encodings fail before
initialization or feedback construction. Activation words follow the identical
admission rule. Existing binary words retain their complete bit domain.

Reuse `read_classical_register`, its identical-type successor and MSB snapshots,
existing nested calls and conditions, and the existing control builder. Scalar
reads add zero physical gates or classical slots. Polynomial coefficients remain
highest-degree-first, with MSB-first coefficient words and row-major arrays.
Classical Split/Join retain their current quantum-only restriction. The user
continues to write `circuit.append(op)`.

There are no new public methods, root exports, opcodes or operation fields.
Symbolic physical values, nonbinary quantum execution and conditional writes,
measurement or reset require their existing separate admission decisions.

## Ownership and delivery

1. Integration records approval, an executable contract and exact old binary
   Program/Circuit artifacts before implementation.
2. Core owns classical structural field admission, reusing field/poly metadata
   checks while keeping quantum admission independently restricted.
3. Compiler owns stored coefficient-range checks and reuses existing reads,
   control composition, slot binding and nested call transport.
4. Simulation keeps numerical construction and codecs. Source callbacks,
   complete legal tensor blocks and original tensor counterexamples remain
   separate evidence.
5. Publish domain-independent commits after concentrated local regression,
   strict typing, formatting, scopes, architecture, IR and protected API checks.

## Acceptance and compatibility

Exhaust all small legal activation/input pairs and invalid coefficient encodings.
Verify scalar/shaped/empty ports, type identity, linear successors, nested reads,
slot reuse, complete complex spectator states, clean work and applicable measured
child histories. Compare unchanged child primitive streams and operand/feedback
order. Retain exact pre-extension binary-field, polynomial, integer and CBit
fixtures, including Program repr, serialized Circuit bytes, hashes and slots.

Report initializers, emitted quantum primitives, clean work and classical slots
separately from source child resource estimates. Preserve the original wrapper's
model/decomposition refusal and polynomial tensor index discrepancy. This
proposal supplies no new parent acceptance before implementation and validation.
