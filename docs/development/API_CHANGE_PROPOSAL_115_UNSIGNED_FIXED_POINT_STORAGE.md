# Proposal 115 — Concrete unsigned fixed-point quantum storage

## Review status

Approved explicitly by the user on 2026-11-20 for implementation according
to Proposal 115. Proposals 113 and 114 admitted only canonical
unsigned single-bit storage, with a built-in integer fractional width at most
one. This approved proposal extends the physical storage contract for wider unsigned
words required by the complete multiplication source signatures.

## Requirement and evidence

The immutable Qualtran v0.7.0 revision is
`bcb17bb25360ed68ce3bc084645db3e929799389`, copyright Google LLC, Apache-2.0.
The complete `multiplication.py` classes ScaleIntByReal, MultiplyTwoReals,
SquareRealNumber and InvertRealNumber declare multi-bit unsigned QFxp ports.
Their existing FlagQuantum logical descriptors and typed Split/Join Program
construction succeed. Physical binding rejects the same descriptors before
execution, even for a four-bit identity. The intake records all affected ports,
original bit decoding and native refusals in
`ftqc-reference-env/g2-wide-fixed-point-storage-intake.json`.

The source roots supply resource formulas and signatures without executable
decompositions. Physical storage admission alone supplies no original-root
execution or primitive-order claim. Existing native paper constructions retain
their separately documented specifications and evidence.

## Contract and owners

Core admits canonical unsigned fixed-point quantum descriptors with all of
these properties.

- Quantum width is a positive built-in integer.
- Classical width is built-in integer zero.
- Parameter order is fractional_width followed by signed.
- Fractional width is a built-in integer at most the quantum width. Negative
  values are retained.
- Signed is exactly False, and the descriptor passes logical validation.

Nominal type, original fractional metadata, MSB-first raw-bit ordering and
Split/Join identity survive binding. Core owns admission. Simulation keeps
numerical interpretation and truncation. Compiler continues composing the
existing shared bit/arithmetic owners. Entry remains `circuit.append(op)`.
No new operation, type, public entry, serialized vocabulary or dependency is
introduced. Signed, symbolic, zero-width, classical, malformed and noncanonical
fixed-point descriptors retain physical rejection.

Compiler explicitly preserves the current rejection of multi-bit fixed-point
control ports. Frozen CtrlSpecAnd rejects QFxp words as noninteger predicates.
The broader storage permission must not silently broaden that separate control
contract. The existing one-bit control behavior remains governed by Proposals
113 and 114. Wider arithmetic storage and numerical signed control have
separate ownership.

## Acceptance and compatibility

Compare original concrete QFxp metadata and raw bits with native descriptors
across positive, zero and negative fractional widths. Verify identity views,
full complex states, entangled spectators, shaped ports, input/output aliases,
RIGHT result creation, LEFT consumption, exact adjoint and clean scratch through
the existing physical binder. Demonstrate the normal append path with a
multi-bit fixed-point typed Program and shared arithmetic calls. Keep zero-gate
view counts, emitted primitive counts and source analytical models distinct.

Retain positive and negative witnesses for one-bit controls and explicit wider
source/native control refusals. Check malformed, signed, symbolic and classical
records. Run concentrated regression, strict typing, formatting, team scope,
architecture, IR and API checks. Record independently any original source
failure or missing decomposition. The existing serialized descriptors need no
migration. The accepted-input extension requires explicit approval under Core
AGENTS and PUBLIC_API_PROTECTION.md before implementation. Parent Q-0208 and
Q-0209 remain open until their full remaining definitions are disposed.
