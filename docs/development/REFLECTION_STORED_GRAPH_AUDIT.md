# Stored zero-reflection dependencies

## Frozen source and ownership

Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0,
remains the unchanged reference. The complete source owners are
`mcmt/multi_control_pauli.py`, `reflections/prepare_identity.py`, and
`reflections/reflection_using_prepare.py`. Their original complete-file
execution and model audits remain applicable. This batch closes the shared
stored MultiControlZ and concrete IdentityPrepare dependencies and the
identity-preparation reflection path. General prepared-state stored reflection
composition remains open.

Compiler owns predicate construction, primitive wrapping and word storage.
Algorithms owns PREPARE and reflection composition. Existing physical binding
and numerical execution remain authoritative. Consumers append bound operations
with `circuit.append(op)`.

## Interfaces and exact dependency graphs

- `multi_control_z_program(values)` retains row-major QBit controls and one
  dirty target. Empty controls omit the controls port and apply atomic Z.
  Both binary polarities use the existing CtrlSpecAnd compute and measured
  erase. CZ retains the source port names `q1` and `q2`.
- `register_identity_program(width)` executes one positive opaque word through
  shared OnEach bit storage. Source Identity is atomic. Its Split, bit Identity
  and Join calls are native implementation below that atomic boundary.
- `identity_prepare_program(prepare)` retains one source-atomic word Identity
  call per concrete scalar THRU selection port. It owns no work or measurements.
  Existing shaped, one-sided and zero-width source refusals remain explicit.
  Symbolic metadata is unchanged, while this stored physical constructor requires
  concrete widths.
- `zero_state_reflection_program(reflection)` retains PREPARE, the optional
  phase-target allocation, open-control X, source word Split calls,
  MultiControlZ, controlled relative phase, Join, PREPARE and optional Free.
  Width-one selections retain their original handles without Split/Join.
  Selection words are MSB-first. External control and selection are THRU.
  Uncontrolled phase-one reflection owns and restores one phase target.
  The predicate separately owns and erases its clean ladder.

Cirq EARLIEST places the final open-control X after the final PREPARE when
an intervening relative phase occupies the target moment. The stored graph
retains the original insertion indices and dependency-scheduled traversal.
Both control polarities and coefficients 1, -1, i, -i and exp(0.31i) are covered.
Source-declared reflection adjoint remains self, including its nonreal-phase
defect. Mathematical inversion uses the conjugate coefficient and is separately
verified against the full conjugate-transpose operator.

## Acceptance evidence

The whole-file frozen source receipts and earlier physical/reflection model
coverage are reused. New evidence checks all X/Z activation patterns through
four controls against original recursive calls, operands and per-wire primitive
streams, including measurement feedback. All 127 X ProgramRecords through six
controls equal the exact published `34309cb98` constructor results.

Complete small complex operators, every raw small measurement history, forward
and mathematical-inverse composition, clean output work, Program-copy execution,
CircuitIR JSON round trip and existing append execution pass. Full operators
use up to three controls for MultiControlZ and three selection
bits for reflection. The six-selection-bit factory uses exact dependency graphs,
primitive inventory and independent model evidence instead of a large dense
operator. Native atomic Identity descendants are explicitly excluded from
claims of a nonexistent source composite recipe.

Persistent external receipts are `g2-reflection-regression-20261122.log`,
`g2-reflection-resources-20261122.json`, `g2-reflection-trace-intake-20261122.log`,
and the strict, formatting, scope, architecture, IR, API and integration receipts.
Resource receipt versions verify Qualtran 0.7.0 and Kickmix 0.1 with Simulator.

## Literal inventory and independent model

All rows use minimal clean pools. Six controls require five measured And
closures and twelve physical wires including the target.

| Six-selection-bit operation | Literal primitives | Original QEC model | Width |
| --- | --- | --- | --- |
| MultiControlZ, zero activation | 24 X, 15 H, 10 T, 10 Tdag, 30 CX, 5 S, 6 CZ, 5 measurement, 5 reset | 5 And, 30 Clifford, 5 measurement | 12 |
| Controlled zero reflection, activation one, coefficient -1 | Same predicate primitives plus 12 I and 1 Z | 5 And, 31 Clifford, 5 measurement | 12 |
| Controlled zero reflection, activation zero, coefficient one | Same predicate primitives plus 12 I and 2 X | 5 And, 30 Clifford, 5 measurement | 12 |

The phase-one open-control model omits the two physically executed X gates.
Identity calls collapse under equal dictionary keys in the original declared
model and have zero logical QEC cost. Explicit physical I leaves remain counted.
These source-model properties are preserved independently from measured gates.

## Remaining concrete gaps

Proposal 119 remains pending explicit approval. Uncontrolled nontrivial scalar
reflection is rejected by this stored constructor. Complete recursive intake
proves that the Ising qubitization factory reaches a registerless GlobalPhase
through its ordinary walk reflection even with a rectangular window. LP and SU2
share the same prerequisite. Controlled reflection has no registerless phase.

General nonidentity PREPARE/junk stored composition and all five complete
qubitization factory graphs remain open. Existing flat execution acceptance is
retained separately. Pending Proposals 116 through 119 remain unimplemented.
No catalog presentation or parent acceptance count changes in this batch.
