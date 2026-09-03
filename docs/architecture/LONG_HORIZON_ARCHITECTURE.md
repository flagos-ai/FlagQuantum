# FlagQuantum Long-Horizon Architecture

Status: candidate architecture for staged adoption. This document does not
promote an experimental capability to production maturity.

## Objective

FlagQuantum keeps semantics stable while compiler technology, execution modes,
simulation algorithms, accelerator stacks, QPU providers, and agent protocols
evolve independently. The architecture is successful when a new implementation
is added through a versioned contract instead of a branch in an unrelated
subsystem.

## Constitution

1. Core owns backend-neutral semantics and versioned data contracts only.
2. Compiler transforms program artifacts and never submits work.
3. Runtime owns one execution attempt, resources, orchestration, recovery hooks,
   and evidence; it never implements numerical simulation algorithms. Durable
   user-facing Job state belongs to the external Compute Service.
4. Simulation owns numerical methods and implements an execution-provider
   contract; it does not own cluster or user-session policy.
5. Providers adapt external accelerators, QPUs, and services. Vendor SDK objects
   do not cross the provider boundary.
6. Ecosystem adapters translate external framework objects at the boundary and
   never establish a second canonical IR.
7. MCP, REST, gRPC, and CLI are replaceable northbound gateways owned by the
   external Compute Service. FlagQuantum exposes protocol-neutral application
   services; protocol adapters never call kernels or devices directly.
8. Unsupported capabilities fail at the earliest knowable stage. Approximation,
   precision downgrade, and CPU fallback are explicit and auditable.
9. Public contracts are versioned and evolve additively before an old version is
   removed through the public API change process.
10. Architecture rules are executable checks, not naming conventions.

## Stable concepts

The long-lived contract vocabulary is deliberately small:

- program artifacts: source, circuit, logical, physical, pulse, network,
  simulation plan, and executable;
- target capabilities: quantum, accelerator, precision, communication, timing,
  fault tolerance, and network;
- execution modes: job, session, realtime session, and workflow;
- result evidence: value, statistics, provenance, actual execution path,
  precision, communication, fallback, checkpoint, and failures.

Existing `CircuitIR`, runtime plan, numerical contracts, target capabilities,
and execution results remain authoritative for their current stable scopes. A
program-artifact envelope adds type, provenance, and future-stage composition;
it does not replace `CircuitIR`.

## Dependency direction

```text
SDK / Ecosystem / external gateways
            |
            v
  protocol-neutral services
            |
            v
Compiler   Runtime   Simulation
     \        |        /
      \       v       /
        versioned Core

Providers depend on public contracts plus their vendor SDK. No general-purpose
package depends on a concrete provider.
```

## Domain responsibilities

| Domain | Owns | Must not own |
| --- | --- | --- |
| Core | IR, artifacts, capability vocabulary, schemas | hardware, scheduling, algorithms |
| Compiler | capture, validation, passes, lowering | job submission, numerical kernels |
| Runtime | execution attempts, sessions, resources, recovery hooks | durable service jobs, circuit optimization, amplitudes |
| Simulation | statevector, MPS, TN, noise, differentiation | user policy, provider credentials |
| Accelerator provider | device runtime, kernels, precision, collectives, topology | public semantics |
| QPU provider | discovery, calibration, submission, sessions, result decoding | generic compiler passes |
| Agent services | discovery, validation, planning, preflight, explanation | LLM or transport implementation |
| Compute Service gateway | tools, resources, protocol lifecycle, authorization | numerical and compiler semantics |

## Evolution model

- Fault tolerance adds logical and physical artifacts, QEC workflows, and
  capability components.
- Pulse control adds a pulse artifact and target compiler/provider support.
- Realtime feedback adds a realtime-session contract and bounded feedback
  channel without changing batch jobs.
- Multi-QPU and quantum networking add a resource graph, network artifact, and
  entanglement-resource provider.
- A new domestic accelerator adds a provider implementing device, kernel,
  precision, collective, topology, and evidence contracts.
- A new MCP revision changes only the external Compute Service gateway unless a
  genuinely new product capability is introduced.

## Compute Service and MCP boundary

The existing FlagQuantum Compute Service and MCP implementation remains an
independent product/control-plane repository. It owns tenants, authorization,
budgets, durable jobs, REST/OpenAPI, MCP transport, and public projections.
This repository owns deterministic IR validation, compilation, local runtime,
provider execution, and machine-readable evidence.

The integration direction is one way:

```text
MCP host -> Compute Service -> FlagQuantum agent service/runtime contracts
```

FlagQuantum does not depend on the Compute Service or an MCP SDK. Deleting or
replacing MCP leaves local SDK execution and runtime semantics unchanged.

## Existing extension and provider authority

`flagquantum.extensions.sdk` remains the current authority for third-party
extension manifests, capability negotiation, lifecycle containment, and
provider registration. `flagquantum.runtime.platforms` remains the authority
for accelerator device lifecycle. The target provider taxonomy is introduced
by evolving those contracts through the protected API process, not by creating
a parallel SPI.

## Adoption rule

The target architecture is adopted through compatibility adapters. Existing
stable APIs remain operational. A subsystem moves only after its new contract,
focused tests, compatibility path, and rollback boundary exist.
