# IR-001: Program Semantics and Execution Request Boundaries

Status: Approved
Date: 2026-09-01
Applicable phase: Phase 1 importer and restricted round-trip.

Approval record: the API owner explicitly approved IR-001 through IR-003 on
2026-09-01. Approval covers only this internal program/request boundary. It does
not change public `CircuitIR`, `fq.plan`, `fq.run`, or measurement conflict rules.
Phase 1 implementation remains subject to the Phase 0 exit gate.

## Background

Public `CircuitIR` schema 1.0 carries instructions, observables, and measurements.
Current planners/runtimes already define the sole request source and conflict
rules. Interop also classifies observables and measurements as execution-boundary
information. Internal QuantumIR must separate the program from what this execution
returns without changing public schemas or `fq.plan/fq.run` behavior.

## Decision

The `CircuitIR` importer returns an internal, nonpublic composite:

```text
ImportedCircuitProgram
  module: QuantumModule
  request: InternalExecutionRequest
  constraints: ImportConstraints
  source: SourceIdentity
  diagnostics: tuple[Diagnostic, ...]
```

Ownership rules:

- `n_wires` and `instructions` enter `QuantumModule`.
- `observables` and request-style `measurements` enter `InternalExecutionRequest`.
- `dtype`, `shape`, and batch/runtime limits enter typed `ImportConstraints`.
- Schema version and source content hash enter `SourceIdentity`.
- Routing and other compilation evidence enter provenance, not the source program
  semantic hash.
- Semantic instruction metadata such as `is_channel`, `is_dynamic`, and `condition`
  must enter typed operations/diagnostics, not ordinary provenance.

Within Phase 1's static scope, dynamic instructions return structured unsupported
diagnostics. Unifying explicit terminal measurement operations with execution
requests awaits a ProgramIR/dynamic ADR; Phase 1 does not expand capability here.

## Preserved Public Rules

- Public `CircuitIR` remains unchanged.
- Conflicts between `fq.plan(..., measurements=...)` and IR measurements still fail.
- Shots, seeds, and options do not enter program identity.
- The importer does not merge, replace, or infer requests itself.
- `fq.run` does not invoke the importer by default.
- Fail closed when a lossless split is impossible.

## Rejected Alternatives

### Keep All Observables/Measurements in QuantumModule

This would continue mixing programs and requests and allow shot changes to
contaminate program identity and compilation caches.

### Change CircuitIR Schema 1.0

This would break the protected serialization contract and exceed Phase 0-1 authorization.

### Carry Requests in Free-Form Metadata

Types, conflicts, identities, and round-trip behavior could not be verified reliably.

## Compatibility and Rollback

This object exists only in `_compiler`. Removing the importer rolls it back;
public objects, caches, deployment artifacts, and user data do not depend on it.

## Acceptance

- Public API snapshots remain unchanged.
- Requests, constraints, and sources have immutable typed models.
- Existing measurement conflict tests pass.
- Static round-trip is exact within the supported scope.
- Dynamic, unknown semantic metadata, and lossy cases fail with structured diagnostics.
- [x] API owner approved the architecture decision.
- [ ] Compiler owner confirms typed model and importer details during implementation review.
