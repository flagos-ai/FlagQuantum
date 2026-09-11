# IR-004: Phase 1 Custom Matrix Operations

Status: Approved
Date: 2026-09-01
Applicable phase: Phase 1 CircuitIR importer, verifier, and restricted round-trip.

Approval record: the API owner explicitly approved IR-004 through IR-006 on
2026-09-01. Approval covers internal importer validation and round-trip boundaries
for concrete custom unitaries only. It neither changes public CircuitIR
construction behavior nor authorizes a new public custom-unitary API.

## Background

Public `CircuitIR` currently allows an unregistered opcode with an explicit
`matrix`. Phase 1 cannot remove that input capability. Nor can it declare arbitrary
shapes, dtypes, or nondeterministic objects valid QuantumIR operations merely
because `Instruction` currently checks only whether a matrix exists.

## Decision

Phase 1 represents verifiable concrete matrices with internal
`quantum.custom_unitary` operations:

- For wire arity `k`, matrix shape must be `2**k × 2**k`.
- Matrices must be numerical, finite, two-dimensional, and square.
- Dtypes must map unambiguously to the existing complex64/complex128 contract.
- The importer verifies unitarity using dtype-specific contract tolerances.
- Preserve the original opcode as a typed symbolic name, not a provider-native gate.
- Preserve wire order, matrix element order, and dtype exactly.
- Concrete custom unitaries may participate in Phase 1 round-trip.
- Symbolic, callable, unknown sparse-format, and dynamically generated matrices
  produce structured unsupported diagnostics.
- Stricter importer validation does not change public `CircuitIR` construction.

Matrices with `requires_grad=True` are outside Phase 1's static custom-unitary
scope. Trainable matrices require separate parameterization, gradient, and
identity contracts in the future.

## Rejected Alternatives

- **Accept arbitrary matrices unchanged**: cannot ensure shape, unitarity,
  determinism, or target legalization.
- **Reject all custom matrices**: removes an important subset of expressible
  static CircuitIR.
- **Automatically decompose into basic gates**: changes round-trip behavior and
  introduces target-dependent optimization prematurely.

## Compatibility and Rollback

Validation occurs only in the internal importer. Legacy compiler/runtime behavior
remains unchanged. Removing the importer rolls back the change without affecting
public schemas or user data.

## Acceptance

- Exact round-trip for concrete one- and two-qubit unitaries.
- Nonsquare, incorrectly dimensioned, nonfinite, and nonunitary matrices fail closed.
- Wire/matrix ordering differential tests pass.
- Trainable/symbolic matrices return explicit blockers.
- Public CircuitIR tests remain unchanged.
- [x] API owner approved the architecture decision.
- [ ] Compiler owner confirms matrix verification and dtype tolerances during review.
