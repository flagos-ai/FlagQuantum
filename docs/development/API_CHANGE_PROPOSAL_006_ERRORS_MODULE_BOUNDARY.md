# API Change Proposal 006: Public Errors and Module Construction

## Status

**Contract frozen — the candidate received separate freeze approval.**

- Target: first public alpha.
- Affected interfaces: `flagquantum.errors`, existing domain errors, and `fq.Module` construction.
- Root name changes: none.
- Root signature change: remove `deployment_binding` from `fq.Module`.
- Authorization: the API owner explicitly authorized Proposal 006 on 2026-09-01.
  That implementation authorization did not freeze the entire API.
- Freeze record: on 2026-09-01, the API owner explicitly approved the exact
  Proposal 006 contract bound by the review packet. This is not the overall first
  public alpha freeze.

## Problem

Stable entry points previously mixed built-in, domain, and incidental backend
exceptions, making failure handling by validation/planning/execution stage
difficult. `fq.Module` also accepted an unvalidated
`deployment_binding: Mapping[str, Any]` that played no part in quantum execution,
planning, or training. Only higher-level application models used it for deployment
summaries. This placed provider state in the wrong owner: a PyTorch quantum layer.

## Public Error Decision

The stable `flagquantum.errors` namespace provides:

```text
FlagQuantumError
├── ValidationError      (also ValueError)
├── PlanningError        (also ValueError)
├── SerializationError   (also ValueError)
├── CompilationError     (also RuntimeError)
├── ExecutionError       (also RuntimeError)
└── CapabilityError      (also NotImplementedError)
```

Multiple inheritance preserves existing built-in exception handling. Incorrect
Python argument types and unknown keywords still raise `TypeError`; invalid values
of valid types raise `ValidationError`. Stable boundaries must not make incidental
PyTorch, JAX, NCCL, or provider exceptions public contracts. Preserve underlying
exceptions through `__cause__`.

Existing domain exceptions retain their names within unified categories:

- `IRValidationError` → `ValidationError`;
- `IRSerializationError` → `SerializationError`;
- `ExecutionPlanContractError` → `PlanningError`;
- `TrainingStateError` → `ExecutionError`;
- missing result data → `ExecutionError`.

## Module Construction Decision

Retain three verified construction forms:

1. Callable builder with flat parameters.
2. Callable builder with named parameter groups.
3. Parameterized Circuit template.

Remove `fq.Module(..., deployment_binding=...)`. Module retains RuntimePolicy,
parameters, and training state only. Application models or `flagquantum.deployment`
own deployment bindings. `HybridQuantumClassifier` and `VariationalEnergyModel`
store bindings as their own state, persisted through PyTorch extra state so
higher-level model checkpoints retain deployment information.

Old Module checkpoint extra-state `deployment_binding` fields remain readable but
are ignored, preserving loading of private-development checkpoints. The first
public version does not promise Module-level bindings.

## Executable Acceptance Criteria

- [x] Seven error types are exposed only in stable `flagquantum.errors`.
- [x] Categories inherit both FlagQuantumError and the corresponding built-in exception.
- [x] IR, Plan, Result, and training-state errors use unified categories.
- [x] Invalid semantic values in stable Circuit/options/policy/Module/train raise ValidationError.
- [x] Unsupported planning/measurement/Module paths raise CapabilityError.
- [x] Module construction no longer includes deployment_binding.
- [x] Module accepts and ignores bindings in old checkpoint extra state.
- [x] Higher-level application model checkpoints retain bindings.
- [x] API owner separately approved contract freeze.
