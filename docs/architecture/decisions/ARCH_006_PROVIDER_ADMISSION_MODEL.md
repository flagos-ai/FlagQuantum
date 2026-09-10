# ARCH-006: Platform/Execution Provider Layers and Admission Dependencies

Status: Superseded by ARCH-001 (2026-09-08 revision)

> Retained as design history. The current architecture uses Compute and Remote
> control boundaries instead of the Platform/Execution Provider layer names.
> The revised ARCH-001 is authoritative.

Date: 2026-09-03
Basis: refinement of approved ARCH-001; this proposal does not register, certify, or implement Providers.

## Context

PlatformRuntime, ExtensionRegistry, Deployment `QuantumProvider`, and Runtime/
Simulation execution paths each work, but lack a shared Core-owned replacement
contract. Platform supplies device/kernel/precision/communication facts; Execution
Provider receives complete execution requests. Their lifecycles, failure domains,
and evidence differ and must not be combined into an oversized interface.

## Decision Candidates

### Platform Provider

Supplies discovery, identity, device lifecycle, memory, kernel/precision,
communication/topology facts, and controlled in-process handles. Produces
capability discovery/evidence, not complete execution-request handling, planning,
task polling, or user results.

### Execution Provider

Consumes complete requests and target snapshots. The proposed minimal asynchronous
protocol has capabilities, submit, status, and result; handles/statuses must be
serializable. Cancellation, calibration, realtime, gradients, and checkpoints are
narrow capability-declared extensions. Runtime drives polling, deadlines,
retry/cancel orchestration, and evidence assembly.

Simulation Execution Provider composes a Simulation Engine and a Platform Provider.
QPU/Remote Service Providers do not depend on simulation algorithms.
ExtensionRegistry remains the sole extension registration/lifecycle authority;
no second registry is introduced.

Agent Services are **Agent-facing deterministic application services**, offering
candidate capabilities, validate, plan, preflight, and execute/explain operations
to Agent/protocol adapters. The LLM and Reasoning Layer belong to an external
Compute Service. They are replaceable, optional, and may compose these operations,
but must not bypass artifact/schema validation, capability fail-closed behavior,
established plan identity, or result/evidence assembly. Natural-language
explanations do not override structured facts.

## Prohibited Practices

- Do not combine both layers into a generic Provider with many optional methods.
- Vendor objects, live jobs, credentials, and nonserializable streams/events must
  not cross adapter boundaries.
- Discovery, contract conformance, or A800 development material cannot establish
  domestic-accelerator, QPU, or production capability.
- Providers cannot rewrite request policy, hide unknown states, or silently fall
  back to another backend or CPU.
- Do not put LLMs, MCP, tenant state, or durable task control planes in the main
  repository's Agent Services. External reasoning cannot bypass deterministic
  services by directly invoking numerical implementations.

## Compatibility

Existing PlatformRuntime, Deployment providers, and Extension SDK remain
currently authoritative; adapters first satisfy candidate contracts. Tightening
`Any` in stable extension protocols, root exports, Deployment schema changes, or
exception changes require separate compatibility/API proposals. Native states and
errors use namespaced extensions, while standard terminal states and failure
categories remain closed sets.

## Admission Dependencies and Migration Sequence

1. Approve versioned artifact, capability, request, and result/evidence contracts
   from ARCH-002 through ARCH-005.
2. Core supplies two narrow protocols, fakes, failure taxonomy, and conformance tests.
3. CPU Platform and Local Simulation pass first, proving that one Simulation
   Engine can use interchangeable Platforms.
4. Remote/QPU contract fakes pass, proving Runtime driver interchangeability across
   Execution Providers.
5. Add real adapters individually, each with capability-specific environment,
   failure, fallback, and evidence material.
6. Update capability maturity only after real hardware/service review. Retire old
   provider lifecycles/results last.

### Conditions for Teams to Begin Implementation

| Team | Admission conditions |
| --- | --- |
| Core | Review the five ADRs; identify internal candidate types and changes needing API Change Proposals. Deliver versioned values, fakes, and conformance first, without changing stable exports. |
| Compiler | Artifact/capability/request contracts are on the baseline. New implementations produce only Core-owned artifacts/decisions, preserving stable plans, failures, and identities. |
| Runtime | Request/result/evidence contracts and provider fakes are on the baseline. Attempt drivers neither import Compiler implementations nor own durable task control planes. |
| Simulation | Simulation request/result projections and a Platform fake are approved. The first local statevector slice preserves autograd, dtype/device, and fail-closed behavior. |
| Platform | Four capability states, JSON-safe identities/handles, and evidence ceilings are approved. Submit real-device tests for specific hardware, kernels, dtypes, and communication scopes. |
| Execution | Handle/status/failure/result/evidence contracts and a Runtime driver fake pass. Real providers additionally require sandbox, bit-order, idempotency, cancellation, and calibration evidence. |
| Ecosystem | Core metadata values and wire/bit/parameter semantics are approved. Adapters do not leak external objects; execution moves to the Execution Provider boundary. |
| Agent | Capability vocabulary and application-service schemas are approved. Services consume only Core snapshots/requests/results; protocols, tenants, and durable tasks remain external. |

## Acceptance Tests

- Platform: JSON-safe identity, unknown/unmeasured states, lifecycle cleanup, and
  no silent device substitution.
- Execution: five states, not-ready, timeout, idempotency, cancellation races,
  recoverable handles, and identity mismatches.
- One Engine switches between CPU and fake Platforms without consumer changes.
- A Runtime driver switches between Local Simulation and a remote fake without
  consumer changes.
- Asymmetric counts/bit order, provider recompilation, calibration snapshots, and
  executed-artifact evidence.
- Real accelerator/multinode/QPU admission requires corresponding hardware tests
  and audits; mocks do not count.

## Open Questions

- Synchronous versus async-neutral Core protocol ports, and separate streaming/
  realtime boundaries.
- Handle recovery periods, idempotency-key scope, and cancel-after-terminal semantics.
- Controlled Platform stream/event handles, topology sources, and communication
  provider ownership.
- Environments and approving parties for the first real QPU sandbox, domestic
  device, and multinode certification.
