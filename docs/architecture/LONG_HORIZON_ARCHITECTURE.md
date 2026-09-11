# FlagQuantum Long-Horizon Architecture

> Status: candidate architecture, adopted in stages.
>
> Scope: the main FlagQuantum repository and its external integration boundaries.
>
> Machine-readable contract: `contracts/long-horizon-architecture-v1.json`.
>
> Precedence: `AGENTS.md`, Stable Core API protection, capability maturity policy,
> and scientific evidence requirements take priority over this document.

## 1. Purpose

This document defines boundaries intended to support more than ten years of
FlagQuantum development. It does not promise unchanged implementations for that
period. Domain responsibilities, dependency directions, and contract evolution
should remain stable; compilation algorithms, simulation methods, hardware SDKs,
communication libraries, agent protocols, and deployment mechanisms may change.

Success means:

1. A new compiler, simulator, domestic accelerator, or QPU connects through
   versioned contracts.
2. Replacing an implementation does not require changes to its callers.
3. Unsupported capabilities fail before execution; precision reduction, CPU
   fallback, and backend substitution are auditable.
4. Single-device, multi-GPU, multi-node, QPU, and service execution share core semantics.
5. Teams work against stable interfaces without editing other domains' internals.

## 2. Goals and Non-Goals

### 2.1 Goals

- **Shared semantics:** one authoritative representation for circuits, hybrid
  control, target capabilities, execution requests, and result evidence.
- **Separate compilation and execution:** Compiler transforms programs; Runtime
  organizes an execution.
- **Separate lifecycle and algorithms:** Runtime owns resources and lifecycle;
  Simulation owns numerical methods.
- **Hardware isolation:** domestic accelerator, QPU, and external service SDK
  objects do not enter Core.
- **Replaceable ecosystem adapters:** PyTorch, JAX, Qiskit, OpenQASM, and QIR
  conversions happen at boundaries.
- **Replaceable protocols:** MCP, REST, gRPC, and CLI are not numerical-kernel dependencies.
- **Built-in evidence:** results include capability, precision, communication,
  fallback, and performance measurement details.
- **Incremental migration:** protect stable APIs and keep a minimum end-to-end
  path working at every step.

### 2.2 Non-Goals

- A universal IR containing every possible future object.
- Moving all existing code into a new layout at once.
- Preventive abstractions without a current use case.
- Duplicating Compute Service tenancy, authentication, billing, or persistent jobs
  in the main repository.
- Compatibility layers that conceal semantic differences or unsupported hardware.

## 3. Architecture Overview

```text
Users and ecosystem
Python SDK | PyTorch/JAX | OpenQASM/QIR | Qiskit | CLI
                            |
                    Boundary conversion
                            v
Ecosystem / API: external adapters, public APIs, program capture
                |                           |
                v                           v
Compiler                            Application Services
Validate, analyze, optimize, lower   Capability aggregation and reusable
Transform ProgramArtifact only      execution/deployment preflight
                | Executable / Plan         | ExecutionRequest
                +---------------------------+
                            v
Runtime: target selection, resources, sessions, distributed execution,
         recovery, observability, and evidence collection
                |                  |                    |
                v                  v                    v
Simulation                   Compute                  Remote
SV / MPS / TN                CPU / GPU / NPU          QPU / GPU/HPC / cloud
Numerical execution          In-process resources     External job control
                |                                       |
                +------------- Noise / Twin ------------+
                    Noise semantics, device models,
                    calibration and observations
                                  |
                                  v
Core: versioned IR, ProgramArtifact, Capabilities, Request, Result, Evidence

External control plane: MCP / REST / gRPC -> public APIs or Application Services
```

Arrows indicate permitted dependency or call directions. Core does not depend
back on upper layers. The detailed import rules in Section 11 define the domain
boundaries, including those summarized by Noise / Twin above.

## 4. Nine Stable Domains

| Domain | Responsibility | May own | Must not own |
| --- | --- | --- | --- |
| Core | Stable semantics and data contracts | IR, artifacts, capabilities, requests, results, evidence, error categories | Hardware SDKs, scheduling, numerical algorithms, network frameworks |
| Compiler | Transform one program artifact into another | Capture, validation, analysis, passes, lowering, code generation | Job submission, device lifecycle, simulation |
| Runtime | Organize an execution and its lifecycle | Planning, resources, sessions, distributed orchestration, recovery, observability | Compiler optimization, SV/MPS/TN algorithms, persistent service jobs |
| Simulation | Quantum numerical computation | Statevector, density matrix, MPS, tensor networks, differentiation | User policy, credentials, calibration, cluster governance |
| Noise | Backend-independent noise semantics | Channels, device noise profiles, readout errors | Numerical evolution, vendor communication, Runtime scheduling |
| Compute | Resources directly controlled by the current process | CPU/GPU/NPU lifecycle, device facts, communication primitives | External job submission, numerical algorithms, Runtime scheduling |
| Remote | External job control | Discovery, submission, status, cancellation, result decoding | Local device lifecycle, numerical algorithms, Runtime scheduling |
| Twin | Build and validate a model of a particular real QPU | Frozen snapshots, calibration-conditioned predictions, drift and hardware validation | Numerical kernels, vendor credentials, remote job transport |
| Ecosystem | External developer ecosystems | Framework adapters, format import/export, plugin entry points | A second canonical IR, Runtime scheduling, direct device calls |

Application Services is a small composition layer. A reusable workflow belongs
there only when it combines stable APIs and adds validation, policy, or structured
failure semantics. MCP, REST, CLI, IDEs, and user code call public APIs directly
for single compilation, planning, or execution operations; they do not need a
forwarding facade.

Compute and Remote are divided by control boundary, not hardware type. Resources
on which the process creates tensors, selects devices, and invokes kernels belong
to **Compute**. Resources accessed by submitting jobs, polling status, and fetching
results belong to **Remote**. The same GPU model may appear as local Compute or
behind an HPC service. Real QPUs usually belong to Remote.

## 5. Target Code Layout

This is a target layout, not an instruction to move everything at once.
Directories named `contracts` contain interfaces and data models, not implementations.

```text
flagquantum/
├── __init__.py              # Public fq facade; stable entry composition only
├── core/                    # Backend-independent stable semantics
│   ├── ir/
│   ├── artifacts/
│   ├── capabilities/
│   ├── execution/
│   ├── evidence/
│   └── errors/
├── compiler/                # Public compiler facade
│   ├── capture/
│   ├── analyses/
│   ├── passes/
│   ├── lowering/
│   ├── codegen/
│   └── pipelines/
├── runtime/
│   ├── planning/
│   ├── execution/
│   ├── sessions/
│   ├── distributed/
│   ├── recovery/
│   └── observability/
├── simulation/
│   ├── statevector/
│   ├── density_matrix.py
│   ├── mps/
│   ├── tensor_network/
│   ├── differentiation/
│   └── kernels/
├── noise/                   # Backend-independent channels and device profiles
├── compute/                 # Resources directly controlled by this process
│   ├── cpu/
│   ├── accelerators/        # Domestic GPU/NPU and heterogeneous accelerators
│   └── communication/       # Collectives, P2P, heterogeneous interconnects
├── remote/                  # External job control
│   ├── qpu/
│   └── services/            # GPU/HPC services and cloud platforms
├── twin/                    # Calibration-conditioned, hardware-validated QPU models
├── ecosystem/
│   ├── pytorch/
│   ├── jax/
│   ├── qiskit/
│   ├── openqasm/
│   ├── qir/
│   └── extensions/
├── services/                # Capability aggregation and reusable preflight
├── algorithms/              # User-facing algorithm composition
├── benchmarking/            # Evaluation and evidence generation
└── testing/                 # Contract, replacement, and conformance tools

contracts/                   # Snapshots and architecture policy; Core owns runtime types
docs/architecture/           # Architecture, ADRs, and focused designs
```

### 5.1 Public Facade and Call Order

`flagquantum/__init__.py` is the sole facade for `import flagquantum as fq`.
It may compose stable Compiler and Runtime entry points but does not implement
compilation, scheduling, numerics, or vendor adaptation. Ordinary users must not
construct internal fingerprints, proofs, or scheduling objects to run a program.

Core supplies shared semantics throughout the flow; it is not the first service
in a call chain. A typical local or domestic-accelerator simulation follows:

```text
User / PyTorch / JAX
 -> fq public facade
 -> Ecosystem boundary conversion (only for external inputs)
 -> Compiler: validation, optimization, lowering, target legality
 -> Runtime: capability matching, target selection, planning, lifecycle
 -> Runtime resolves device, precision, and communication through Compute
 -> Runtime calls Simulation: statevector / MPS / TN / noise / gradients
 -> ExecutionResult + ExecutionEvidence
 -> Runtime
 -> Ecosystem result conversion (if needed)
 -> User
```

Compute explains how this process uses computation and communication devices.
Remote explains how to submit jobs and retrieve results through an external
control plane. Runtime chooses the path and manages lifecycle. Simulation owns
numerical computation, does not import Compute, and does not disguise remote
services as local devices.

A real QPU task bypasses local Simulation and Compute:

```text
User -> fq public facade -> Compiler -> Runtime
     -> Remote -> Real QPU
     -> ExecutionResult + ExecutionEvidence -> Runtime -> User
```

GPU/HPC services also sit behind Remote. Algorithms compose applications through
the public facade. Application Services provide useful composite preflight using
public contracts. Protocol adapters call stable APIs for single operations.
Benchmarking uses the same execution paths as users; it has no special fast path
that bypasses capability, safety, or evidence checks.

### 5.2 Migration Ledger

The complete machine-readable ledger is
`contracts/long-horizon-architecture-v1.json`. Each item declares its owner,
milestone, current and target authority, adapter, completion evidence, old
implementation exit condition, and status.

| Migration | Current authority | Target | Completion evidence | Exit condition |
| --- | --- | --- | --- | --- |
| Core contracts | `core`, Runtime execution objects, platform/provider local contracts | `core` | Core defines cross-domain artifacts, capabilities, requests, results, and evidence; serialization tests pass | No callers of duplicate private contracts; provider protocols have neutral projections |
| Compiler consolidation | `compiler` | `compiler` | Replace a pipeline without changing Runtime or user APIs | Old `_compiler` and `compilation` entries removed |
| Simulation extraction | `simulation`, parts of `runtime/executors` | `simulation` | Real engines and contract fakes pass the same conformance suite | Runtime owns no numerical algorithms |
| Compute consolidation | `compute` | `compute` | Two platforms pass capability, precision, communication, fallback, and replacement tests | Generic code no longer imports vendor runtimes |
| Execution targets | `runtime/executors`, `deployment` | `remote` | Simulation, QPU, and remote services share result contracts | Backend selection and decoding stay behind providers |
| Ecosystem consolidation | `ecosystem` | `ecosystem` | Boundary conversion and round-trip tests pass | External framework objects stay outside core domains |
| Services/gateway separation | `services` | Composite workflows in the repository; protocol gateways at system edges | Local path works without MCP SDK; adapters call public APIs or composite services | No production MCP transport dependency or serialized forwarding facade in the repository |

A target directory without an exit condition is not a migration plan. Completion
requires removing or closing the old authoritative entry; parallel authorities
must not persist indefinitely.

## 6. Four Core Contract Families

Core owns all four cross-domain contract families. Compiler, Runtime, Simulation,
and providers implement or consume them without copying their definitions.
Runtime accepts Core-defined executable artifacts and requests; it does not
depend on Compiler packages or `compiler_contracts`.

### 6.1 ProgramArtifact

`ProgramArtifact` is a versioned envelope between compilation stages. It does not
replace `CircuitIR`. The target minimum fields are:

- `kind`: source, circuit, logical, physical, pulse, network, simulation_plan, executable;
- `schema_version`: artifact schema version;
- `payload`: canonical data for that stage;
- `provenance`: source, generating tool, input digests, parent artifacts;
- `requirements`: precision, dynamic control, communication, QEC, pulse, or network needs;
- `extensions`: optional namespaced extensions that do not redefine core fields.

Compiler functions follow:

```text
ProgramArtifact + TargetCapabilities + CompileOptions
    -> ProgramArtifact + CompileEvidence
```

### 6.2 TargetCapabilities

Capabilities describe what a target actually provides, not what a user wants:

- Quantum: gate set, qubits, measurement, dynamic circuits, parameterization.
- Accelerator: device type, memory, native/software-extended precision, kernels.
- Communication: P2P, collectives, inter-node communication, topology, bandwidth class.
- Timing: batch, session, real-time feedback latency bounds.
- Fault tolerance: logical qubits, code families, correction cycles, magic states,
  resource estimation.
- Network: multiple-QPU topology, links, entanglement generation, resource state.

Each capability carries maturity and evidence provenance. An interface declaration
is not a production capability.

### 6.3 ExecutionRequest

An execution request contains stable inputs for one execution:

- artifact and parameters;
- target constraints and execution mode;
- precision, fallback, approximation, and backend-substitution policy;
- resource budget, random seed, timeout, and recovery policy;
- required evidence level.

| Mode | Purpose | Lifecycle owner |
| --- | --- | --- |
| Job | One batch execution | Runtime manages an attempt; Compute Service may manage a persistent job |
| Session | Repeated low-overhead execution | Runtime manages session resources and state |
| Realtime Session | Bounded-latency quantum-classical feedback | Dedicated Runtime/Provider capability |
| Workflow | Directed compilation, execution, training, mitigation, and correction flow | Workflow orchestrator composes atomic contracts |

### 6.4 ExecutionResult and Evidence

Results include more than numbers. Minimum evidence covers:

- request, program, compiled artifact, target, and environment digests;
- actual backend, device, precision, and execution path;
- compute/communication paths, topology, processes, and device residency;
- approximation, truncation, noise, software-extended precision, and error definitions;
- CPU fallback, backend substitution, retries, and recovery;
- timing, throughput, memory, and measurement methods;
- structured reasons for success, failure, or partial completion.

Undeclared CPU fallback is a contract violation.

## 7. Three End-to-End Paths

### 7.1 Local or Distributed Simulation

```text
User program
 -> Ecosystem/API capture
 -> CircuitIR / ProgramArtifact
 -> Compiler validation and simulation-oriented optimization
 -> Runtime execution plan and resources
 -> Simulation Provider
 -> Statevector / MPS / TN / Noise / Diff
 -> ExecutionResult + Evidence
```

Runtime does not implement tensor contraction or quantum gate kernels. Simulation
does not choose clusters, accounts, or fallback policy.

### 7.2 Real QPU

```text
User program
 -> Canonical IR
 -> Compiler logical/physical lowering for QPU capabilities
 -> Runtime Job, Session, or Realtime Session
 -> QPU Provider isolates vendor SDK, credentials, calibration, result formats
 -> Shared ExecutionResult + QPU Evidence
```

A real QPU is a **provider implementation**. Runtime owns its scheduling
lifecycle; Compiler owns QPU transformations; Core owns capability vocabulary and
result schemas.

### 7.3 Application Services and Protocol Adapters

```text
LLM / IDE / MCP Host
 -> Thin MCP / REST / gRPC / CLI adapter
 -> Stable public API (single step) or FlagQuantum Services (composite preflight)
 -> Compiler / Runtime / Deployment public contracts
 -> Structured results and evidence
```

- The main repository has no MCP SDK dependency or LLM decision logic.
- Services retain reusable capability aggregation and execution/deployment preflight.
- Adapters decode protocols and serialize results without wrapping single-step APIs
  in additional services.
- Compute Service owns tenancy, authentication, quotas, budgets, persistent jobs,
  and protocol lifecycle.
- Replacing MCP versions or gateways leaves local SDK and compute semantics unchanged.

## 8. Runtime and Simulation

| Question | Owner |
| --- | --- |
| Which target, how many devices, which execution mode? | Runtime |
| How are statevectors updated, MPS truncated, and TNs contracted? | Simulation |
| When are communication groups, checkpoints, and recovery established? | Runtime |
| How do forward, backward, and gradient kernels compute? | Simulation |
| Is lower precision or CPU fallback allowed? | Request Policy + Runtime |
| What precision, communication, and fallback occurred? | Compute/Remote collect; Runtime aggregates Evidence |

Runtime composes Simulation and Compute. It resolves device, precision, and
communication resources, then passes explicit data and device parameters to
Simulation. Simulation does not import Compute, allowing numerical engines and
device adapters to be replaced independently. Runtime sends external jobs to
Remote rather than through the local Simulation path.

## 9. Domestic Heterogeneous Compute

Each domestic GPU/NPU or heterogeneous chip connects through a separate Compute
adapter without exposing vendor objects upstream. Compute provides at least:

1. Device discovery, lifecycle, and memory capabilities.
2. Kernel registration, compilation, or invocation.
3. Native and software-extended precision declarations, including Double-Single.
4. Intra-node P2P, collectives, and inter-node communication capabilities.
5. Device topology, heterogeneous interconnects, and unavailability reasons.
6. Actual residency, kernel, communication, precision, and CPU fallback evidence.
7. Contract conformance and replacement tests.

Upper layers depend on capabilities rather than vendor-name branches. Runtime
Planner selects hardware using request constraints, capability evidence, and
policy. Single-precision devices may declare software-extended double precision
only with explicit workloads, numerical validation, overhead, and execution-path
evidence. This is not native double precision.

## 10. Future Extensions

### 10.1 Fault-Tolerant Quantum Computing

Add logical, physical, and fault-tolerant artifacts, extend
`TargetCapabilities.fault_tolerance`, and compose logical-to-physical mapping,
QEC cycles, magic-state distillation, and resource estimation through Workflows.
The base Runtime does not implement specific code algorithms.

### 10.2 Quantum-Classical Real-Time Feedback

Add `RealtimeSession` with explicit feedback latency, control location,
instructions, and timeout semantics. Ordinary remote Jobs do not imply real-time
support. Compiler validates dynamic-circuit semantics; Provider validates hardware
capabilities.

### 10.3 Multiple QPUs and Quantum Networks

Add Network Artifacts, distributed QPU targets, network topology, and entanglement
resource providers. Runtime orchestrates QPUs, Compiler partitions programs, and
Provider performs and evidences actual entanglement operations.

### 10.4 Pulse Compilation and Control

Add Pulse Artifacts and corresponding Compiler pipelines. Pulse objects do not
belong in generic CircuitIR. Only providers declaring pulse capability may accept them.

### 10.5 New Simulation Methods and AI Optimization

New numerical methods implement the Simulation Contract. AI optimizers connect as
Compiler passes or Planner policies. Model suggestions require deterministic
validation and cannot bypass capability or semantic-equivalence checks.

## 11. Dependency and Import Rules

```text
Core <- Compiler
Core <- Runtime
Core <- Simulation
Core <- Noise
Core <- Compute + Vendor Runtime
Core + Noise + Simulation/Remote public APIs <- Twin
Core + Deployment <- Remote + External SDK
Core + Compiler/Runtime/Deployment public APIs <- Application Services
Public API + Core <- Ecosystem
Public APIs + Application Services <- Protocol Adapters
```

Mandatory rules:

- Core does not import Runtime, Simulation, Compute, Remote, Ecosystem, or gateways.
- Compiler does not import Runtime, Compute, Remote, or device SDKs.
- Runtime does not import Compiler; Core owns shared data contracts.
- Simulation does not import Runtime policy or deployment code.
- Compute and Remote remain separate interfaces.
- Generic code does not import concrete Compute or Remote implementations.
- Ecosystem converts objects at the boundary without leaking them into core domains.
- Gateways do not directly call kernels, devices, or compiler internals.
- Temporary exceptions have a registered owner and removal condition and are
  tracked by architecture checks.

`architecture.toml` and `tools/check_architecture.py` enforce these rules.

## 12. Team Ownership

Divide ownership by domain rather than technology stack.

| Team/workstream | Main scope | Stable dependencies |
| --- | --- | --- |
| Core/IR | `core`, contract schemas | No downstream implementations |
| Compiler | `compiler` | ProgramArtifact, Capabilities |
| Runtime | `runtime` | Core ExecutionRequest, ExecutionResult |
| Simulation | `simulation` | Simulation Contract, Evidence |
| Noise | `noise` | Core IR, backend-independent noise semantics |
| Compute | `compute` | Core capability, precision, and device-fact contracts |
| Remote | `remote` | Core request, result, and remote-job contracts |
| Twin | `twin` | Noise, Simulation, and Remote public APIs |
| Ecosystem | `ecosystem` | Public API, ProgramArtifact |
| Application Service / Gateway | `services`, protocol edges | Public API, composite workflow contract |

Cross-domain changes start with contract proposals and tests, followed by
implementation. Importing another team's internals is not an integration shortcut.
Each domain maintains an owner, public entry points, contract tests, replacement
fakes, and a change record.

## 13. Capability Maturity and Release

Classify each capability separately:

```text
declared -> prototyped -> validated -> production
```

- **declared:** contract and failure semantics exist.
- **prototyped:** implementation runs, but evidence is incomplete.
- **validated:** representative environment, precision, replacement, and conformance
  tests pass.
- **production:** continuous testing, documentation, operating boundaries, and
  release commitments exist.

An interface, passing mock, or single-machine demonstration does not establish
production maturity.

## 14. Migration Sequence

Establish contracts before replacements and a working vertical slice before
expanding across domains. Do not reorganize directories all at once. The controlling
sequence is:

```text
Freeze new cross-domain abstractions
 -> Complete and simplify Compiler boundaries
 -> Make the minimum CPU vertical slice work
 -> Move proven paths into target directories
 -> Remove or explicitly freeze historical and transitional code each round
```

Do not start additional cross-domain architecture tracks before the minimum CPU
path works. Add a cross-domain abstraction only when the current authority cannot
support a demonstrated path and the contract proposal and simplification review
approve it. Directory migration follows working slices, not empty directories or
parallel implementations.

### Phase 0: Freeze Facts and Boundaries

- Inventory public APIs, authoritative implementations, and actual capabilities.
- Enforce forbidden dependency directions.
- Establish behavior contracts and evidence baselines.

### Phase 1: Minimum End-to-End Contracts

- Stabilize ProgramArtifact, TargetCapabilities, ExecutionRequest, and ExecutionResult.
- Adapt existing `CircuitIR` into new services compatibly.
- Run capture, validation, planning, simulation, results, and evidence end to end.

### Phase 2: Separate Compiler, Runtime, and Simulation

- Establish public facades.
- Replace cross-layer internal calls with contracts.
- Prove each extracted implementation can be replaced without changing consumers.
- Record additions, reuse, deletion, and frozen code. A new path without retirement
  of the old authority is not a completed migration.

### Phase 3: Consolidate Compute and Remote

- Place directly controlled devices and communication in Compute.
- Place QPU, remote GPU/HPC, and cloud job control in Remote.
- Integrate domestic accelerators, real QPUs, and remote services individually.
- Do not introduce another device registry or capability discovery system.

### Phase 4: Ecosystem and Services

- Unify PyTorch/JAX, OpenQASM/QIR, and third-party boundaries.
- Stabilize a small set of reusable Application Services.
- Let thin MCP/REST/gRPC/CLI adapters reuse public APIs and composite workflows.

### Phase 5: Future Capability Plugins

- Add fault-tolerant, real-time, multiple-QPU, network, and pulse artifacts in
  response to actual projects.
- Mature capabilities independently without rewriting the base execution model.

## 15. Completion Criteria

A module is decoupled only when all these conditions hold:

1. Responsibilities, inputs, outputs, and failures are documented.
2. External interfaces expose versioned contracts, not internal or vendor objects.
3. At least two implementations exist, or one real implementation and a contract fake.
4. Replacing an implementation leaves consumers unchanged.
5. Contract, conformance, and architecture checks pass.
6. Unsupported capabilities, degradation, and fallback are machine-readable.
7. Documentation states current maturity without presenting target design as implemented.
8. Ordinary changes generally stay within one main domain. If similar changes
   repeatedly cross four or more domains, stop expanding implementation and review boundaries.
9. Every migrated domain has a short README and an executable ten-minute golden
   path covering responsibilities, exclusions, dependencies, public entry, and a minimal change.
10. Public APIs do not require knowledge of internal fingerprints, provenance
    records, legality proofs, capability snapshot IDs, scheduling objects, or evidence internals.
11. Readable scenario tests accompany contract, determinism, and tamper tests. A
    new contributor can use the README and golden path to implement, test, and
    explain a representative small change independently.
12. Every nontrivial change receives a simplification review before integration:
    remove unused scaffolding, forwarding wrappers, duplicate validation,
    duplicate representations, and mechanical tests. Passing tests do not justify
    abstractions without a current product behavior and domain responsibility.

Proposals for contracts, identities, verdicts, registries, or intermediate
representations explain why current authoritative types cannot express the
validated need. Split large internal modules by responsibility after behavior and
boundaries stabilize; shorter files alone do not justify public concepts.
The ten-minute target is a human usability acceptance goal, not a synthetic CI
claim about contributor experience. New managers, registries, factories, protocols,
helper layers, or intermediate objects require an independent current
responsibility or a second concrete use case, not hypothetical future demand.

## 16. Decision Governance

An Architecture Decision Record (ADR) is required for:

- core data-model or dependency-direction changes;
- new cross-domain contracts or provider types;
- Stable Core API or serialization changes;
- new production dependencies, long-lived compatibility layers, or cross-repository protocols;
- new precision-reduction, CPU fallback, or backend-substitution policies.

ADRs include the problem, constraints, alternatives, decision, compatibility
impact, migration, validation, owner, and exit condition. Implementation may
evolve continuously; expedient temporary exceptions do not override core boundaries.

## 17. Current Adoption

This target architecture does not raise the maturity of experimental capabilities.
Existing `CircuitIR`, Runtime plans, numerical contracts, target capabilities,
and results remain authoritative within their stable scope. Adoption proceeds
through compatible adapters and the protected API process. Each migration needs
focused tests, a rollback boundary, and verifiable evidence.
