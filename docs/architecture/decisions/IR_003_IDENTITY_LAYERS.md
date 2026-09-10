# IR-003: Program, Compilation, and Execution Identity Layers

Status: Approved
Date: 2026-09-01
Applicable phases: Phase 1 identity; Phase 2-3 cache/artifact extensions.

Approval record: the API owner explicitly approved IR-001 through IR-003 on
2026-09-01. Approval covers identity layering and Phase 1 internal program
identity research only. It does not authorize replacing existing ExecutionPlan,
deployment, checkpoint, or public result identities. Phase 2-3 use still requires
review by the corresponding owners.

## Background

`CircuitIR.content_hash` currently covers the complete public payload, including
measurements, metadata, dtype, and shape. ExecutionPlan and deployment rely on
existing hashes/contracts. The long-term architecture must avoid rerunning
placement/routing merely because shots change, while preventing reuse of incorrect
artifacts after target, pipeline, or capability changes. Phase 1 cannot replace
any existing public identity.

## Decision

Internal identities have three layers:

```text
program_identity
  = canonical quantum program semantics
    + static parameter values that affect structure/semantics
    + semantic numerical constraints approved by schema

compilation_identity
  = program_identity
    + target identity
    + compiler/pipeline digest
    + compilation options
    + capability snapshot
    + calibration snapshot when required

execution_identity
  = artifact/compilation identity
    + runtime parameter bindings
    + execution request
    + shots/seed
    + execution policy
```

Phase 1 implements internal `program_identity` only, retaining:

- `source_circuit_ir_hash`: existing public `CircuitIR.content_hash`;
- `internal_program_identity`: used only for differential evidence and future
  cache research;
- `identity_schema_version`: an internal version with no user compatibility promise.

These identities cannot substitute for one another. Existing ExecutionPlan,
DeploymentPackage, checkpoints, and public results retain contract-required identities.

## Phase 1 Program Identity Rules

Include:

- operation schemas/versions, ordering, and nested structure;
- qubit/value dataflow;
- opcodes, typed operands/results, parameters, and values;
- typed attributes affecting quantum or numerical semantics;
- dtype/precision constraints explicitly classified as program semantics by an ADR.

Exclude:

- source locations, diagnostics, and wall time;
- shots, seeds, queues, credentials, and accounts;
- routing, targets, and compiler pipelines;
- nonsemantic provenance/debug metadata;
- execution requests, unless a future ADR makes a terminal operation part of the program.

Fields still unclassified after the IR-001 metadata inventory must not be silently excluded.

## Determinism Requirements

- Canonical ordering cannot depend on Python object addresses, accidental dict
  insertion order, or process seeds.
- Identical inputs and importer/schema versions produce identical identities.
- Semantic changes must change program identity.
- Source locations and nonsemantic provenance must not change program identity.
- Evidence records the identity algorithm, schema version, and canonical encoder.
- Phase 1 does not promise internal identity stability across versions.

## Rejected Alternatives

### Reuse CircuitIR.content_hash for Every Identity

The current public payload mixes programs, requests, constraints, and metadata,
preventing correct compilation/execution cache layering.

### Replace ExecutionPlan Identity in Phase 1

This would change a protected execution contract before TargetIR/artifacts exist.

### Exclude All Metadata

Some current metadata affects channel, dynamic, condition, runtime, and routing semantics.

## Compatibility and Rollback

Initially, internal identities appear only in tests/evidence, not public results,
checkpoints, deployments, or provider payloads. Removing them leaves existing
cache keys and user data unaffected.

## Acceptance

- Source hashes and internal identities coexist explicitly.
- Identity mutation tests cover included and excluded fields.
- Cross-process and fixed-platform determinism tests pass.
- Existing plan/deployment/checkpoint identities remain unchanged.
- [x] API owner approved identity layering.
- [ ] Compiler owner confirms canonical encoding during implementation review.
- [ ] Runtime owner confirms compilation/execution identity integration before Phase 2-3.
