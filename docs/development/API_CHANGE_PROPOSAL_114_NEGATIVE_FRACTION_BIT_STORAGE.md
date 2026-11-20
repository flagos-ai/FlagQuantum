# Proposal 114 — Negative fractional widths for single-bit storage

## Review status

Approved explicitly by the user on 2026-11-20 for implementation according
to Proposal 114.
Proposal 113 admitted fractional widths 0 and 1. This separately approved
extension admits the negative integer widths specified below.

## Original executable requirement

The unchanged Qualtran v0.7.0 revision is
`bcb17bb25360ed68ce3bc084645db3e929799389`.
`QFxp(1, fraction, False)` also accepts negative integer fractional widths.
`ControlledViaAnd(XGate(), CtrlSpec(dtype, bit))` successfully decomposes these
storage descriptors. For fractions -1, -2 and -7, both control polarities emit
exactly CX or X/CX/X and model one or three Clifford gates.
`g2-fixed-point-negative-fraction-intake.json` records all six source streams,
operands, raw activation bits, real interpretations and native refusals.
For example, fractional width -1 gives raw bit 1 the real interpretation 2.
The original control still consumes raw integer 1.

## Contract and owners

Extend Core's existing canonical executable single-bit unsigned fixed-point
admission from fractional widths 0 or 1 to every built-in integer fractional
width at most 1. Existing constructors and logical validation remain the
metadata authority. Quantum width must be built-in integer 1, classical width
built-in integer 0, sign exactly False, and parameter order canonical.

Compiler continues using the existing unsigned stored-bit predicate, typed
Split/Join, attachment callbacks and exact adjoints. Simulation retains real
conversion. The existing `circuit.append(op)` execution path remains the entry.
The extension introduces no new operation, type, parameter key or dependency.
Signed width-one construction, wider physical storage, noncanonical metadata
and unresolved symbolic storage retain their recorded rejection behavior.

## Acceptance

Reuse Proposal 113's complete source-call, primitive, complex-amplitude,
measurement-feedback, work-zero, inverse and replacement witnesses for negative
fractions -1, -2 and -7, scalar and singleton-array ports, both polarities.
Keep observed primitives separate from source analytical models. Retain source
matrix-singleton defects and wider CtrlSpecAnd refusals. Run concentrated
regression, strict typing, formatting and architecture/IR/API checks before
publication. This proposal supplies no complete parent acceptance by itself.
