# API Change Proposal 003: ExecutionPlan Identity and Execution Semantics

## Status

**Contract frozen — implementation, full verification, root manifest migration, and contract freezing are complete.**

- Target version: first public alpha.
- Affected interfaces: `fq.ExecutionPlan`, `fq.plan`, `fq.run`, `Circuit.plan`,
  `Circuit.run`, and `ExecutionResult.plan`.
- Machine-readable candidate: `contracts/execution-plan-v1-candidate.json`.
- Prerequisites: Proposal 001 lists `ExecutionPlan` as a planned Stable Core
  addition; Proposal 002 has frozen `ExecutionOptions` and the current program
  execution entry points.

Implementation approval: the API owner explicitly authorized implementation on
2026-08-31. This allowed `fq.run(plan)` and the corresponding `fq.run` input
rename, but did not authorize adding `ExecutionPlan` to the stable root manifest
or freezing the contract.

Root-level approval: on 2026-09-01, the API owner explicitly authorized adding
`ExecutionPlan` to the Stable Core root manifest. This did not constitute the
final freeze of the entire FlagQuantum API or Proposal 003 serialization contract.

Freeze record: on 2026-09-01, the API owner explicitly approved freezing the exact
Proposal 003 contract bound to the review packet. This does not constitute the
overall first public alpha freeze.

## Problem

The object returned by `fq.plan(program)` describes circuit analysis, layers,
memory estimates, and recommended modes, but is not yet the sole execution
source of truth:

1. It does not own normalized `CircuitIR` and cannot execute independently.
2. It lacks stable identity, schema version, options, and environment fingerprints.
3. Free-form mappings such as `runtime_config` and `routing_plan` cannot serve as
   long-term serialization contracts.
4. `fq.run(program)` plans again, so the inspected plan may differ from the one executed.
5. `ExecutionResult.plan` still accepts several internal objects and cannot be
   reliably matched to an explicit plan.
6. Environment or plugin changes lack uniform stale-plan detection; executors may
   select again or fall back.

The following workflow therefore cannot yet be reliably promised:

```python
plan = fq.plan(circuit, options=options)
result = fq.run(plan)
```

## Decision Summary

`ExecutionPlan` becomes an immutable local plan with identity, created by
`fq.plan` and directly executable. `fq.run(program, options=...)` must use the
same internal `plan -> validate -> execute` path. `fq.run(plan)` must execute the
supplied plan without replanning.

```python
automatic = fq.run(circuit, options=options)

plan = fq.plan(circuit, options=options)
explicit = fq.run(plan)

assert automatic.plan.identity == plan.identity
assert explicit.plan.identity == plan.identity
```

An `ExecutionPlan` can be serialized and restored across processes, but executes
only when its version, plugin, and environment constraints are satisfied. It is
not a deployment package for cloud or QPU submission. `DeploymentPackage` remains
responsible for long-term delivery across machines, signatures, credentials, and
provider job parameters.

## Object Boundaries

### Creation

The only stable creation entry points are:

```python
fq.plan(program, *, options=None)
fq.ExecutionPlan.from_dict(payload)
fq.ExecutionPlan.from_json(text)
```

The `ExecutionPlan(...)` constructor is not public contract. Internal planner
structures may evolve without allowing users to construct partially valid plans.
`from_dict/from_json` must fully validate the schema, identity, and structure;
they cannot bypass planner validation.

### Local Plans and Deployment Packages

```text
CircuitIR + ExecutionOptions + current capability environment
                  |
                  v
            ExecutionPlan
       inspect / cache / restore / execute
                  |
                  | explicit deployment conversion
                  v
           DeploymentPackage
       sign / submit / provider lifecycle
```

`ExecutionPlan` excludes:

- Provider credentials, job IDs, queues, and retry policies.
- Python callables, live Torch/JAX objects, CUDA graphs, and process-group handles.
- Unversioned pickle payloads.
- Absolute paths, hostnames, PIDs, rank IDs, and creation timestamps.
- Raw environment variables and free-form runtime configuration.

## Stable Public Surface

`ExecutionPlan` is an immutable value object created through factories. Its first
stable public properties are:

| Property | Type | Meaning |
| --- | --- | --- |
| `identity` | `str` | Canonical plan identity: 64 lowercase SHA-256 hexadecimal characters |
| `schema_version` | `str` | Serialization schema version; initially `"1.0"` |
| `program_fingerprint` | `str` | Content hash of the complete normalized `CircuitIR` |
| `options_fingerprint` | `str` | Canonical hash of resolved execution semantics |
| `environment_fingerprint` | `str` | Canonical hash of required environment constraints, not the entire host |
| `compiler_fingerprint` | `str` | Canonical hash of the compilation pipeline, passes, and relevant plugin versions |
| `mode` | `str` | Selected mathematical state representation |
| `backend` | `str` | Selected execution backend |
| `device` | `str` | Selected logical device constraint |
| `target` | `str` | Selected output target |
| `batch_size` | `int` | Planned batch size |
| `precision` | `str` | Planned complex precision |
| `world_size` | `int` | Required logical rank count |
| `state_bytes` | `int` | Planner estimate of state working-set bytes |
| `is_distributed` | `bool` | Read-only derived value of `world_size > 1` |

Stable methods are:

```python
plan.summary() -> Mapping[str, object]
plan.to_dict() -> dict[str, object]
plan.to_json(*, indent: int | None = None) -> str
ExecutionPlan.from_dict(payload) -> ExecutionPlan
ExecutionPlan.from_json(text) -> ExecutionPlan
```

`analysis`, `layers`, candidate costs, routing details, backend kernel selection,
and noisy execution details are not frozen as stable properties in the first
version. They belong in controlled internal sections of the versioned payload or
in subsequent stable diagnostic interfaces, not free-form public mappings.

## Identity Model

### Canonical Identity

Plan identity is:

```text
sha256(canonical_json(identity_payload))
```

`canonical_json` must use UTF-8, sorted object keys, deterministic number encoding,
and no extra whitespace. `identity_payload` contains only information affecting
execution semantics or executability:

```text
ExecutionPlan schema name/version
program_fingerprint
resolved execution semantics
compiler fingerprint
required environment constraints
selected execution decision
versioned backend/compiler extension identities
```

Identity must exclude:

- Creation time, runtime, hostname, PID, and current rank.
- Instantaneous available-memory samples and performance counters.
- Mapping insertion order, Python `repr`, and object addresses.
- Log paths, cache paths, credentials, and provider transport metadata.

Identical IR, resolved options, compilation pipelines, environment constraints,
and planner decisions must yield identical identities. Different user inputs may
share an identity when they resolve to exactly the same execution semantics.

### Environment Constraints

`environment_fingerprint` hashes the **capabilities required by the plan**, not
all properties of the current machine. Constraints include at least:

- Backend/provider names and compatible protocol versions.
- Device kind and required capabilities.
- Precision, world size, and distribution semantics.
- Required communication, gradient, noise, or approximation capabilities.
- Compiler/backend extension versions affecting executability.

The current environment may provide a superset of these capabilities. Changes to
CPU models, GPU ordinals, or free GPU memory must not invalidate a plan merely
because strings differ. Unavailable backends, insufficient world size, unsupported
precision, or incompatible plugin protocols must fail before execution.

## Serialization Boundary

Initial schema:

```json
{
  "schema": "flagquantum.execution_plan",
  "version": "1.0",
  "identity": "<sha256>",
  "program": {"kind": "flagquantum.circuit_ir", "version": "1.0"},
  "requested_options": {"schema": "flagquantum.execution_options", "version": "1.0"},
  "resolved_options": {},
  "fingerprints": {
    "program": "<sha256>",
    "options": "<sha256>",
    "environment": "<sha256>",
    "compiler": "<sha256>"
  },
  "environment_requirements": {},
  "decision": {},
  "extensions": []
}
```

The machine-readable candidate constrains the exact fields:

- `program` uses the existing versioned `CircuitIR` schema.
- `requested_options` preserves field-level inheritance intent;
  `resolved_options` stores the concrete execution values.
- `decision` stores canonical decisions needed for execution, not live runtime objects.
- Each extension requires a namespace, kind, schema version, and identity.
- Reject unknown top-level fields, unsupported schemas/versions, duplicate
  extensions, and identity mismatches.
- `from_dict` must recompute every fingerprint and the identity.
- After the first public alpha freeze, the same major schema must continue to
  read existing fixtures.
- Successful deserialization does not establish executability in the current
  environment; environment validation occurs during execution preflight.

Pickle is prohibited as a stable serialization format.

## `plan` and `run` Semantics

Target signatures after approval and implementation:

```python
fq.plan(
    program,
    *,
    options: ExecutionOptions | None = None,
) -> ExecutionPlan

fq.run(
    program_or_plan,
    *,
    options: ExecutionOptions | None = None,
    measurements: Sequence[MeasurementNode] | None = None,
    noise_model: Any | None = None,
) -> ExecutionResult
```

### Program Input

For a program input:

1. Normalize to `CircuitIR`.
2. Resolve options through Proposal 002's sole resolver.
3. Create an `ExecutionPlan`.
4. Validate its identity and the current environment.
5. Execute that plan without invoking another planner.
6. Return the actual stable plan in `ExecutionResult.plan`.

### Plan Input

For an `ExecutionPlan` input:

- `options`, `measurements`, and `noise_model` must all be `None`.
- Any additional execution semantics immediately raise `TypeError`; overriding
  and merging are prohibited.
- Validate the schema, identity, environment, and backend/compiler extension compatibility.
- Execute the IR, resolved options, and decision sealed in the plan directly.
- Do not invoke a planner or recompile through a pipeline that changes identity.
- Do not silently fall back, reduce world size, lower precision, or change mode.
- Within one process, `result.plan is plan` should hold. Across serialization,
  identity equality is the minimum guarantee.

### Measurement and Noise Boundaries

This section originally deferred to Proposal 004 and is now superseded by it:
measurements enter canonical `CircuitIR` before planning; noise models enter a
versioned plan extension with identity validation. Neither may bypass the plan
through transient execution. Existing plans still reject all temporary call
arguments.

## Stale Plans and Failure Stages

Execution preflight checks in a fixed order:

1. Schema/version.
2. Payload structure and extension uniqueness.
3. Program/options/compiler/environment fingerprints.
4. Plan identity.
5. Current backend, device, precision, world size, and capabilities.
6. Executable extension compatibility.

Failures must occur before kernels launch, distributed workers are created, or
provider jobs are submitted. Stable reason codes include at least:

```text
unsupported_schema
identity_mismatch
program_fingerprint_mismatch
options_fingerprint_mismatch
compiler_incompatible
environment_incompatible
backend_unavailable
world_size_mismatch
extension_incompatible
```

The unified error-model proposal determines exact public exception classes. Until
then, Proposal 003 freezes only failure stages, reason codes, and the prohibition
on replanning and fallback, avoiding permanent commitments to incidental
low-level exception types.

## Migration from the Existing Implementation

`flagquantum.compilation.models.ExecutionPlan` is an internal migration source,
not automatically stable. Implementation should:

1. Introduce canonical plan payload and identity helpers.
2. Include `CircuitIR`, requested/resolved options, environment constraints, and
   planner decisions in the plan.
3. Replace free-form stable fields with read-only properties; move backend
   details to versioned extensions.
4. Add `_execute_plan(plan)` shared by automatic and explicit paths.
5. Migrate internal callers and `ExecutionResult.plan` before exposing root-level
   `ExecutionPlan`.
6. Update the stable manifest and signature snapshot after approval.
7. Retain old internal analysis/layer data without accidentally freezing it as
   permanent public fields.

The repository is not yet open source, so no deprecation obligation is introduced
for the unpromised internal `ExecutionPlan(...)` constructor form.

## Explicit Exclusions

This proposal does not decide:

- Every `ExecutionResult` field, accessor, or metadata schema.
- Final measurement replacement/composition rules.
- The stable serialization location of noise models.
- Deployment packages, signatures, provider submission, or remote job lifecycles.
- Backend/provider/plugin registration protocols themselves.
- Permanent plan executability across FlagQuantum major schemas.
- Stable formats for performance caches, compiled binaries, or CUDA graphs.

## Acceptance Criteria

- [x] API-owner approval of object boundaries, stable properties, and factory-only construction.
- [x] API-owner approval of identity inputs and exclusions.
- [x] API-owner approval of the serialization schema and compatibility window.
- [x] API-owner approval that plan inputs reject all options/measurement/noise overrides.
- [x] Plans created by `fq.plan` execute independently without a second planner/compiler call.
- [x] Automatic and explicit paths have equivalent identities, outputs, and provenance.
- [x] Stale, tampered, or environment-incompatible plans fail closed before execution.
- [x] `ExecutionResult.plan` on executable plan paths is the actual `ExecutionPlan` executed.
- [x] Round-trip, unknown-field, identity-tampering, and schema-rejection tests pass.
- [x] Default, runtime, distributed, and documentation contracts pass.
- [x] The API owner separately approved the root-level export.
- [x] The API owner separately approved the contract freeze.

## Implementation Record

Implementation and verification completed on 2026-08-31:

- `ExecutionPlan` has canonical program/options/environment/compiler fingerprints,
  SHA-256 identity, strict JSON round trips, tamper detection, and world-size preflight.
- The normal `fq.run(program)` path and `fq.run(plan)` share one exact-execution path.
- Plan inputs reject options, measurements, and noise_model overrides; contract
  tests prove that neither planner nor compiler is invoked again.
- Proposal 004 brought measurement and noise program paths into that same
  exact-execution path.
- Default, runtime, distributed, benchmark/release contracts, API snapshots,
  architecture, Ruff, and Black passed. The root-level `ExecutionPlan` export
  received approval on 2026-09-01; contract freezing was not yet enabled at this
  point in the record.

## Approval Boundary

This proposal received implementation authorization and a second approval for
the root manifest. `ExecutionPlan` moved from planned additions to Stable Core.
That approval **did not itself freeze the contract** or authorize changes to other
stable measurement/noise/result semantics. Freezing required separate API-owner
approval, recorded in the Status section above.
