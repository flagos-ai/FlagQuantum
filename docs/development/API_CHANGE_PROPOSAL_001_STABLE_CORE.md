# API Change Proposal 001: Stable Core Consolidation

## Status

**Approved — implementation authorized, not frozen.**

Approval record: on 2026-08-31, the API owner explicitly approved Stable Core
classification and namespace migration for the first public alpha. Approval does
not freeze the API; final freeze requires the acceptance conditions below.

This document defines root API consolidation before the first public alpha, not
a frozen contract. Machine classification lives in
`contracts/public-api-v1-candidate.json`. API Change Proposal 018 supersedes the
original `flagquantum.backends` decision.

## Decision Summary

The current root namespace has 60 stable exports, including backend executors,
distributed research entries, MPS production acceptance objects, deployment tools,
and project introduction functions. The proposed final Stable Core has 22 entries:

- Retain 20 existing root interfaces.
- Add `ExecutionOptions` and `ExecutionPlan` as core types.
- Move 19 interfaces to stable extension namespaces.
- Move 18 interfaces to `experimental`.
- Remove `get_version`, `hello`, and `info` from the public root before release.

This changes commitment levels and discoverability without removing capabilities.

## Final Stable Core Candidate

| Domain | Interfaces |
| --- | --- |
| Construction and IR | `Circuit`, `CircuitIR`, `Instruction`, `IR_VERSION` |
| Parameters and observables | `Parameter`, `ParameterExpression`, `ObservableNode`, `MeasurementNode`, `MeasurementResult` |
| Execution | `ExecutionOptions`, `ExecutionPlan`, `ExecutionResult`, `RuntimePolicy`, `plan`, `run` |
| Training | `Module`, `TrainingResult`, `train` |
| Errors | `IRSerializationError`, `IRValidationError` |
| Boundaries | `experimental`, `__version__` |

Selection asks whether an interface belongs to most users' golden paths, is
backend-neutral, has sustainable semantics, and needs root-level discovery.
The 22 entries fit below the 25-entry budget and cover
`Circuit → plan → run → ExecutionResult` and `Module → train → TrainingResult`.

## Namespace Layers

### Stable Extensions

These capabilities may be public and stable without occupying the root namespace:

| Namespace | Responsibility |
| --- | --- |
| `flagquantum.runtime` | Backend-native execution and device policy. |
| `flagquantum.compiler` | Expert compilation entries. |
| `flagquantum.deployment` | Packages, providers, and Pauli measurement deployment. |
| `flagquantum.noise` | Noise models and specialized simulation. |
| `flagquantum.operators` | Gate schemas and queries. |
| `flagquantum.simulation.mps` | MPS-specific simulation. |
| `flagquantum.simulation.tensor_network` | TN-specific amplitude/expectation operations. |

Ordinary users still access these capabilities through `fq.run`. Stable extensions
serve users who explicitly need backend-native objects or expert controls.

### Experimental

The following do not yet warrant long-term compatibility promises:

- Distributed TN and explicit distributed training.
- MPS production plans, acceptance gates, and benchmark/release evidence.
- Specialized runtime-selection and cost-selection planners.

Move them into `flagquantum.experimental.distributed`,
`flagquantum.experimental.mps`, and `flagquantum.experimental.planning`. Once they
meet stability criteria, they can become stable extensions without enlarging the root API.

### Remove Before Open Source

`hello`, `info`, and `get_version` do not implement quantum AI capabilities.
Standard `fq.__version__` remains available, so these convenience functions need
no permanent compatibility promise.

## Compatible Migration Rules

Because the repository is not public, complete the approved migration before the
first alpha rather than transferring internal design history to users. Follow this order:

1. Establish target namespaces and verify new paths.
2. Migrate repository code, tests, and documentation.
3. Run golden paths and complete API contracts.
4. Narrow `fq.__all__`, `__getattr__`, and `__dir__`.
5. Update the final manifest, retaining the v0.2 baseline for audit.
6. Mark the candidate frozen only after the alpha release audit.

If external trial users already exist at approval time, add explicit
`DeprecationWarning` for migrated entries rather than removing them directly.

## Outside This Proposal

- ExecutionOptions fields and configuration precedence.
- `fq.run(ExecutionPlan)` implementation.
- Narrowing ExecutionResult fields or the error hierarchy.
- Promising that every stable extension meets stability criteria at the first alpha.

Subsequent configuration, executable-plan, and result-contract proposals cover
these separately so naming, signature, and behavior changes remain reviewable.

## Approval Criteria

- [x] API owner individually approved all 22 Stable Core candidates.
- [x] Target namespaces exist for migrated interfaces and pass import contracts.
- [x] Critical repository golden paths no longer depend on historical root extensions.
- [x] ExecutionOptions follow-up implemented and verified (Proposal 002).
- [x] ExecutionPlan follow-up implemented and approved at the root (Proposal 003, freeze pending).
- [x] CI proves all 60 exports are classified exactly once.
- [ ] Establish whether external users need compatibility before the first public alpha.

After all migration and acceptance conditions pass, API freeze requires separate
approval. This proposal's approval does not freeze it automatically.
