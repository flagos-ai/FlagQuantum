# Typed quantum control values at the Cirq boundary

This closes the concrete transport dependency identified in the complete
Q-0158/Q-0159 controlled-source audit. The source remains frozen at Qualtran
v0.7.0 `bcb17bb25360ed68ce3bc084645db3e929799389`, controlled.py blob
`6fabe760e793f1b39f825d76f80e92b6e5fa5c28`. Copyright 2023 Google LLC,
Apache-2.0. Whole controlled parent acceptance still requires the other
boundaries recorded in `CONTROLLED_INTERFACE_SOURCE_AUDIT.md`.

## Interface and ownership

Ecosystem `_control_values.py` imports one Cirq AND clause into existing native
Register metadata and NumPy activation arrays. Export traverses those registers
in row-major order and emits each stored value in MSB-first order. Explicit
registers retain their object identity, names, shapes, type parameters and
sides. With no registers, import matches the source scalar-QBit or flat-QBit
array convention. The foreign control object stops at the Ecosystem boundary.

Simulation `encode_register_value` composes existing scalar codecs for bit,
opaque, unsigned, signed, bounded, one's-complement, fixed-point, Montgomery,
finite-field and polynomial-field values. It preserves scalar range checks
instead of substituting the unsigned array fast path. Existing typed value
collection preserves field element classes, signed integer storage and shaped
object values on import. These are numerical metadata operations and do not
broaden executable Core type admission.

The two modules have separate ownership and commits. Compiler continues to
receive native Register and ControlRegister values. The consumer test feeds
imported values into the existing controlled Program and appends the emitted
operation through DynamicCircuit.

## Frozen-source defect and deliberate disposition

`CtrlSpec.from_cirq_cv` leaves its cursor at zero for every register. The native
import consumes the exact width of each register before reading the next. For
control bits `(0, 1)` and two scalar QBits, the unchanged source decodes `(0, 0)`
and the native import decodes `(0, 1)`. This is an explicit repair, not a claim
of parity with that defective source result. Export matches the original,
and single-register imports compare directly with the original. Multi-register
imports instead verify exact round-trip and every expected typed value.

The source's existing round-trip test checks truthiness of the reconstructed
CtrlSpec for its multi-register example, so it does not detect this mismatch.
The native regression includes that independent witness and all 64 mixed
unsigned/signed cases with distinct register slices.

## Validation

Tests cover all small integer domains, scalar/vector/matrix shapes, negative
values, stored fixed-point and Montgomery words, fields and polynomial fields,
default shape selection and explicit metadata identity. QAny has no enumerated
source classical domain, so its four two-bit patterns are selected explicitly.
The field test prevents conversion of Galois scalar objects to bare NumPy
integers during import.

Quantum-only transport rejects classical storage, symbolic declarations,
nonbinary SDK controls, multiple OR clauses, unequal register/value counts,
shape mismatches and wrong bit counts. These are explicit boundary errors.
Empty or otherwise unsupported source inputs are not repaired into executable
Programs. Existing Core admission and compiler validation remain authoritative.

The physical consumer compares unchanged source primitive operands and feedback,
all two measurement histories, unnormalized complete complex amplitudes with
an entangled spectator, clean work and actual appended execution. Measured
T/T-dagger count four and one measurement are kept separate from the source QEC
model, whose full fields match the existing native model.

## Remaining whole-file work

Typed one-bit metadata selection beyond scalar/shape-one QBit, all direct SDK
controlled declaration/diagram protocols and generic declared-child count
propagation still require complete audit. Other typed physical encodings and
symbolic controls retain their current admission boundaries. Q-0158/Q-0159
remain open until those requirements have their own complete evidence.

## Batch results

The concentrated regression passes 4002 cases with one existing deprecation
warning. The final boundary additions are rerun in the 58-case transport file.
Strict typing passes five files. Black, Ruff, Simulation/Ecosystem scope,
architecture and multi-level IR checks pass. Dependency scanning reports 687
existing whole-repository findings and zero changed-path findings.

The expanded regression exposed a stale test which expected fixed-point logical
metadata to fail Program verification. The existing logical-width contract
admits that representation while physical binding still rejects it. The test
now proves both outcomes through their normal entry points. No Core rule was
changed. Optional NumPy collection is guarded in that touched test file.
