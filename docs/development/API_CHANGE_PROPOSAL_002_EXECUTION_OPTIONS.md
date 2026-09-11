# API Change Proposal 002: ExecutionOptions

## Status

**Implemented and verified — implementation, root-manifest authorization, and full validation complete.**

- Target: first public alpha.
- Affected interfaces: `fq.ExecutionOptions`, `fq.plan`, `fq.run`, `Circuit.run`,
  `Circuit.plan`, and `RuntimePolicy`.
- Machine candidate: `contracts/execution-options-v1-candidate.json`.
- Prerequisite: approved Proposal 001 selected ExecutionOptions as a planned
  Stable Core addition without approving its fields or semantics.

Approval record: on 2026-08-31, the API owner authorized implementation, followed
by root exports and Stable Core manifest migration. This did not authorize
ExecutionPlan inputs or ExecutionResult changes. The contract was subsequently
frozen after full validation.

## Problem

Execution behavior may come from several sources:

- Circuit `bsz`, `device`, `dtype`, and `runtime_config`.
- RuntimePolicy `mode`, `backend`, fallback, and MPS parameters.
- `fq.plan(state_mode=..., bsz=..., config=...)`.
- `fq.run(mode=..., **options)`.
- Task-local RuntimeConfig, environment variables, and backend-specific parameters.

This creates four public risks:

1. Different names for one concept, such as mode/state_mode and batch_size/bsz.
2. Unknown `**options` keys whose errors may appear only inside a backend.
3. Callers cannot establish which configuration source won.
4. Automatic selection, explicit selection, and Module training may produce
   different execution semantics.

## Decision Summary

Add immutable, serializable `fq.ExecutionOptions` as the sole stable execution
configuration input to plan/run. It is a **field-level override object**, not an
already resolved runtime configuration:

```python
options = fq.ExecutionOptions(
    mode="mps",
    device="cuda:0",
    precision="complex64",
    target="expectation",
    require_gradients=True,
)

plan = fq.plan(circuit, options=options)
result = fq.run(circuit, options=options)
```

All fields default to `None`, meaning unspecified at this layer. After merging all
sources, built-in defaults produce internal resolved options. The resolved type
does not enter Stable Core.

## Stable Type

Proposed implementation:

```python
@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    mode: str | None = None
    backend: str | None = None
    device: str | None = None
    target: str | None = None
    batch_size: int | None = None
    precision: str | None = None
    shots: int | None = None
    seed: int | None = None
    memory_limit_bytes: int | None = None
    require_gradients: bool | None = None
    allow_approximate: bool | None = None
    allow_backend_fallback: bool | None = None
```

Field order, names, types, defaults, `frozen=True`, and `slots=True` are stable contracts.

### Field Semantics

| Field | Stable meaning | Initial built-in default |
| --- | --- | --- |
| `mode` | Mathematical state representation, excluding distributed topology and kernel implementation. | `"auto"` |
| `backend` | Tensor/autograd implementation or registry name. | `"auto"` |
| `device` | Logical device request, including any device index in its string. | `"auto"` |
| `target` | Required result form for planning, not request contents. | `"auto"` |
| `batch_size` | Program batch-size constraint. | Program declaration, otherwise `1`. |
| `precision` | Complex computation precision. | `"complex64"` |
| `shots` | Sample count; `None` means sampling not requested. | `None` |
| `seed` | Reproducible stochastic execution seed. | `None` |
| `memory_limit_bytes` | Planning memory limit per device/rank. | `None` |
| `require_gradients` | Plan must support parameter gradients. | `False` |
| `allow_approximate` | Permit approximate representations or truncation. | `False` |
| `allow_backend_fallback` | Permit substitution when an explicit backend/device is unavailable. | `False` |

`allow_approximate=False` and `allow_backend_fallback=False` are fail-closed
defaults. `mode="auto"` authorizes initial planner selection, not fallback.

### Initial Controlled Values

`mode` accepts only:

```text
auto, statevector, mps, tensor_network, density_matrix
```

`distributed_statevector`, `distributed_mps`, and `distributed_tensor_network` are
no longer mathematical modes. Distribution is an execution layout determined by
resources/planning and recorded in ExecutionPlan. Aliases such as `tn`,
`distributed`, and `distributed_tn` do not enter the stable object.

`target` accepts only:

```text
auto, state, expectation, samples, amplitudes
```

Observable, bitstring, and measurement-request contents remain program/request
inputs, not ExecutionOptions fields. Without explicit measurements,
`target="auto"` preserves full-state result semantics.

Initial `precision` values are complex64 and complex128 only. Real dtype, JAX x64,
and kernel precision must derive consistently from this value, without a second
authority.

Backend/device names are validated registry strings, not permanently closed
Literals, allowing third-party implementations. `triton` is a kernel provider,
not a complete autograd backend.

## Configuration Precedence

Resolve each field in this fixed order, earlier overriding later:

```text
Explicit ExecutionOptions for each plan/run call
    > Module RuntimePolicy execution_options
    > Program constraints captured by Circuit
    > Task-local / process RuntimeConfig adapter values
    > FlagQuantum built-in defaults
```

Rules:

- Only non-None fields override values.
- A field cannot be supplied through both options and a transitional legacy keyword.
- Conflicts raise TypeError before planning.
- Environment variables supply topology, available devices, and internal tuning;
  they cannot override explicit mathematical semantics, precision, targets, or
  fallback authorization.
- All environmental effects enter the subsequent ExecutionPlan environment/
  provenance fingerprint.

## Boundaries with Existing Objects

### RuntimeConfig

RuntimeConfig remains a task-local internal/expert configuration and serialization
object, not a stable execution entry. One adapter converts it to lower-priority
ExecutionOptions. Planners cannot read it directly in competition with explicit inputs.

Drawing styles, JAX matmul precision, and operator-kernel switches are outside
Stable Core execution semantics and do not enter ExecutionOptions.

### RuntimePolicy

RuntimePolicy should consolidate into:

- `execution_options: ExecutionOptions`;
- Module/training-specific observable and correctness policy.

Existing mode, backend, allow_backend_fallback, mps_max_bond, and mps_cutoff fields
need pre-release migration or explicit compatibility adapters, not two equivalent
authorities alongside execution_options. MPS bond/cutoff belongs to backend
extension policy, not Stable Core ExecutionOptions.

### Circuit

Circuit input batches and tensor dtype/device are program constraints, not
higher-priority execution preferences:

- Batch-size conflicts with intrinsic program batches raise ValueError during planning.
- Required precision/device conversions must be explicit in plans; unsupported
  conversions fail during planning.
- Circuit.run cannot silently inject bsz/device/dtype/config to alter explicit options.
- `Circuit.run(options=x)` must equal `fq.run(circuit, options=x)`.

## Target Stable Signatures

Following approval, Phase 2 targets:

```python
fq.plan(program, *, options: ExecutionOptions | None = None)
fq.run(
    program,
    *,
    options: ExecutionOptions | None = None,
    measurements: Sequence[MeasurementNode] | None = None,
) -> ExecutionResult
Circuit.plan(*, options: ExecutionOptions | None = None)
Circuit.run(
    *,
    options: ExecutionOptions | None = None,
    measurements: Sequence[MeasurementNode] | None = None,
) -> ExecutionResult
```

The final stable location of noise_model, `fq.run(ExecutionPlan)`, and plan identity
belong to later proposals and are not frozen incidentally by Proposal 002.

## Validation and Errors

- Unknown constructor keywords or incorrect Python types: TypeError.
- Values outside controlled sets or integers below allowed bounds: ValueError.
- Multiple sources supplying the same explicit field: TypeError.
- Intrinsic batch/precision/device conflicts with resolved options: planning ValueError.
- Explicit backend/device unavailable without fallback permission: planning failure.
- Executors cannot silently correct, discard, or reinterpret options.

Integer rules: `batch_size >= 1`, `shots >= 1`, `memory_limit_bytes >= 1`, and
`seed >= 0`. Python bool is not accepted as an integer.

## Serialization

`ExecutionOptions.to_dict()` uses:

```json
{
  "schema": "flagquantum.execution_options",
  "version": "1.0",
  "mode": null,
  "backend": null,
  "device": null,
  "target": null,
  "batch_size": null,
  "precision": null,
  "shots": null,
  "seed": null,
  "memory_limit_bytes": null,
  "require_gradients": null,
  "allow_approximate": null,
  "allow_backend_fallback": null
}
```

Serialization retains None because it represents field-level inheritance.
`from_dict()` rejects unknown schemas, versions, and fields rather than silently
ignoring future fields.

## Explicit Exclusions

Stable Core ExecutionOptions excludes:

- world_size, ranks, nodes, process groups, and communication backends;
- MPS max_bond/cutoff, TN slicing/paths, and statevector kernel switches;
- routing strategies, coupling maps, and compiler passes;
- noise models, observables, bitstrings, and measurement requests;
- checkpoints, optimizers, and training steps;
- vendor credentials and provider job parameters;
- arbitrary extras/kwargs dictionaries.

These belong to resources, programs/requests, compiler policy, backend extensions,
deployment, or experimental namespaces. Extras cannot bypass stable API review.

## Migration Plan

1. Approve this proposal and machine candidate before runtime changes. (Complete.)
2. Implement ExecutionOptions, strict validation, serialization, and the resolver.
3. Make plan/run/Circuit/RuntimePolicy share that resolver.
4. Inventory legacy keyword migrations; unknown keys fail immediately.
5. Migrate production code, tests, examples, and docs.
6. Run automatic/explicit paths, Module, noise, distributed, and backend conformance.
7. Add ExecutionOptions to the root manifest after separate approval. (Complete.)
8. Add ExecutionPlan input semantics after Proposal 003.

## Acceptance Criteria

- [x] API owner approved fields, order, types, defaults, and fail-closed policy.
- [x] No backend-specific or distributed orchestration fields in ExecutionOptions.
- [x] One resolver merges stable configuration sources field by field.
- [x] Plan/run produce identical resolved options for identical program/options.
- [x] Circuit.run and fq.run are equivalent.
- [x] Unknown/conflicting fields and invalid values fail before planning.
- [x] Serialization round-trip and unknown-field rejection tests pass.
- [x] Legacy keyword inventories reach zero in production, tests, examples, and current docs.
- [x] Complete Stable Core, runtime, distributed, and documentation contracts pass.
- [x] API owner separately approved root-manifest inclusion.

## Implementation Record

Public integration completed on 2026-08-31:

- `flagquantum.runtime.options.ExecutionOptions` implements immutable slots
  dataclasses, strict validation, and v1 serialization.
- `flagquantum.runtime.options_resolver.resolve_execution_options` implements
  unique field precedence, the RuntimeConfig adapter, program batch conflict
  checks, and field-source records.
- Root ExecutionOptions, plan/run, Circuit.plan/run, and RuntimePolicy use this
  contract; backend-specific controls move into fq.experimental.
- Semantic contracts cover plan/run consistency, Circuit equivalence,
  reproducible sampling, old-keyword rejection, and explicit-backend fail-closed
  behavior. The candidate was frozen after default, runtime, and distributed tiers passed.

## Authorization Boundary

Implementation and root-manifest authorization alone do **not constitute API
freeze**, nor do they authorize `fq.run(ExecutionPlan)`, ExecutionResult changes,
or additional Stable Core names.
