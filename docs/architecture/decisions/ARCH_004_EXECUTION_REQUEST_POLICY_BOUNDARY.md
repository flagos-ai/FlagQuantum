# ARCH-004: Execution Request and Policy Boundaries

Status: Proposed

Date: 2026-09-03
Basis: Phase 0 inventories from eight teams; no change to `fq.run`, `fq.plan`, or stable options.

## Context

Core does not yet own an authoritative `ExecutionRequest`. Stable Runtime
`ExecutionOptions`, Compiler-private options with the same name,
`RequestedExecution`, importer requests, and free-form distributed requests coexist.
Request facts, user authorization, planner decisions, resource discovery, and
provider credentials can easily become mixed in one object.

## Decision Candidates

1. A Core request is an immutable envelope of intent: artifact identity,
   parameter/input bindings, measurement/output requirements, target constraints,
   a compatible projection of stable options, precision/approximation/fallback
   authorization, budget/seed/deadline/recovery limits, and evidence level.
2. Policy is decision-making behavior owned separately by Runtime, Compiler, and
   Compute Service. It does not enter request identity. The request records only
   the caller's explicit constraints and authorizations. Policy output must become
   a plan or decision with an identity.
3. A single Runtime attempt receives a validated request/plan and a bounded
   resource lease. Tenant, authentication, quota, billing, queue, and retry or
   idempotency across attempts belong to an external Compute Service envelope.
4. Provider credentials, process groups, device handles, and vendor SDK objects
   are resolved only inside adapters.

## Prohibited Practices

- Do not duplicate stable `ExecutionOptions` fields or change their defaults,
  precedence, Literal values, or exception behavior.
- Do not present planner selection, discovered resources, or actual fallback as
  the user's request.
- Do not substitute `Mapping[str, Any]`, Runtime-private types, or Compiler
  re-exports for the Core boundary.
- Runtime must not silently replan, recompile, or substitute a target while
  executing an established plan.

## Compatibility

The first version constructs requests through adapters from existing `fq.run/plan`
call forms, preserving public signatures. Compiler-private `ExecutionOptions`
must be disambiguated as a binding-specific concept. Changes to stable options,
plan/result fields, or failure stages require a separate API Change Proposal.

## Migration Sequence

1. Approve artifact/capability identity dependencies and the request schema draft.
2. Provide a request fake, strict parser, and stable-options adapter.
3. Compiler produces plan decisions; Runtime executes only what was decided.
4. Remove free-form distributed requests and private options with conflicting names.
5. Add the durable-task envelope through an anti-corruption adapter in the external
   Compute Service.

## Acceptance Tests

- Canonical identity, rejection of unknown fields/versions, and deterministic
  binding/seed/deadline handling.
- Unsupported requirements fail at the earliest knowable stage; unauthorized
  fallback fails.
- `fq.run(program)` and `fq.run(fq.plan(program))` preserve their existing
  equivalence; plan inputs are not replanned.
- Tenant, credential, vendor, and process-group objects cannot be serialized into
  a request.
- A Runtime fake consumer can switch between two Compiler/Provider implementations
  without changes to its calling code.

## Open Questions

- Clock semantics for deadlines and timeouts; minimum authorization for retries
  within one attempt.
- Whether tensors/initial states use in-process handles or artifact references.
- How measurement requests compose with stable `ExecutionOptions`, and the request
  minor-version policy.
