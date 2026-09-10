# Unified Heterogeneous Execution and Cloud Ecosystem Roadmap

Status: in progress; IR Phase 1 has formally exited and Stage 2 is active.

Scope: multi-level IR, Quafu end-to-end execution, multiple quantum clouds,
classical compute clouds, and quantum-classical orchestration.

Branch: `codex/open-source-api-convergence`

This roadmap does not authorize changes to Stable Core, public serialization
schemas, or root exports.

Related documents:

- [Multi-level IR architecture](../architecture/MULTI_LEVEL_IR_ARCHITECTURE.md)
- [IR Phase 0–1 implementation plan](../architecture/MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md)
- [IR Phase 0 exit audit](../development/IR_PHASE_0_EXIT_AUDIT.md)
- [IR implementation status](../development/IR_IMPLEMENTATION_STATUS.md)
- [IR Phase 2 entry and Batch A approval packet](../development/IR_PHASE_2_ENTRY_APPROVAL_PACKET.md)
- [Quafu Provider and portal task contract](../development/API_CHANGE_PROPOSAL_012_QUAFU_PROVIDER_CONTRACT.md)
- [Stage 2 Quafu entry audit](../development/STAGE_2_QUAFU_ENTRY_AUDIT.md)
- [Public API protection](../development/PUBLIC_API_PROTECTION.md)
- [Capability maturity](CAPABILITY_MATURITY.md)
- [Known limitations](../reference/KNOWN_LIMITATIONS.md)

## 1. Purpose

This roadmap joins IR, quantum-cloud, and classical-compute work into one
verifiable product sequence. It defines ordering, Core versus deployment
responsibilities, portable programs without silent semantic changes, phase exit
criteria, and evidence of a working execution loop beyond adapter counts.

Names such as `ExecutionTarget`, `ExecutionBinding`, `Job`, and
`ComputeFabricAdapter` are internal candidates, not public API commitments.

## 2. Product Goal

FlagQuantum aims to provide shared quantum-AI heterogeneous execution
infrastructure:

```text
fq.Circuit / fq.Module
          |
          v
 Public CircuitIR + internal multi-level IR
          |
          v
 Compiler / Planner / Execution Policy
          |
          v
 Unified Job and Execution Provenance
          |
    +-----+------------------+--------------------+
    |                        |                    |
    v                        v                    v
local CPU/GPU/NPU   classical compute fabric   quantum cloud/QPU
statevector/MPS/TN   Docker/Slurm/Kubernetes   Quafu/Braket/others
    |                        |                    |
    +------------------------+--------------------+
                             |
                             v
              normalized result + full provenance
```

The intended experience is:

- Circuit, training, and stable result-reading code survive backend changes.
- Local, distributed, noisy, and QPU execution share one program authority.
- Users select backends or authorize explicit planning policies.
- Fallback requires permission and actual execution location stays visible.
- Quantum and classical resources compose recoverable, auditable workflows.

The goal extends beyond collecting cloud SDKs or offering simulation only as a
backup for unavailable QPUs.

## 3. Recorded Baseline

### 3.1 Available Foundations

- Protected, serializable, hashable public `CircuitIR` 1.0.
- Formally completed internal QuantumIR Phase 1: skeleton, importer, verifier,
  identity, differential corpus, and performance gates.
- `fq.run` unifies local paths; `ExecutionOptions` includes backend, device,
  shots, and basic fallback fields.
- Statevector, MPS, TN, density matrix, and distributed execution foundations.
- `CloudBackendProfile`, `DeploymentPackage`, `ProviderTaskHandle`,
  `DeploymentResult`, and deployment artifact identity chains.
- Local, Quafu, and Amazon Braket provider implementations or adapters.
- QASM/QCIS export, routing, experimental dynamic circuits, Quafu calibration conversion.
- API protection, maturity, test tiers, and fail-closed policies.

### 3.2 Missing at This Baseline

- `fq.run` and `deploy_circuit` remain separate local/cloud entries.
- No production unified Job, persistence, idempotent submission, recovery, or
  shared cancellation semantics.
- Incomplete cross-provider status, error, result, and capability conformance.
- No shared resolver combining capabilities, queues, costs, and constraints.
- No abstraction validated on two structurally different real quantum clouds.
- No Slurm/Kubernetes classical control-plane adapters.
- No production QPU–GPU hybrid workflow orchestration.
- No real cloud provider marked release-certified in the capability matrix.

### 3.3 Planning Scores

These scores prioritize work; they are not public capability or performance claims.

| Area | Planning completion | Basis |
| --- | ---: | --- |
| Public IR/API governance | 75/100 | Certified public IR and protection mechanisms |
| Multi-level IR migration readiness | 60/100 | Phases 0–1 exited; Phase 2 compiler migration not started |
| Provider/deployment foundations | 60/100 | Several implementations, no unified certification |
| Quafu production loop | 35/100 | Candidate contracts and partial foundations; P0 incomplete |
| Transparent backend switching | 30/100 | Unified local execution; separate cloud entry |
| Classical cloud control plane | 20/100 | Compute Runtime exists; resource control largely absent |
| Quantum-classical orchestration | 15/100 | Algorithms/executors exist; no shared recoverable workflow |

## 4. Boundaries

### 4.1 Program Plane

The program plane describes what executes: `Circuit`, `Module`, public
`CircuitIR`, internal ProgramIR/QuantumIR/TargetIR, logical qubits, measurements,
parameters, control flow, target legalization, program and compilation identities.

Program semantics/hashes exclude portal `backend_id`, provider job IDs,
credentials, queue state, prices, temporary resource addresses, retry counts,
and scheduler Pod/Job IDs.

### 4.2 Execution Control Plane

The control plane describes where, when, and under which constraints execution
occurs. Candidate objects are:

```text
ExecutionRequest
ExecutionPolicy
ExecutionTarget
BackendCapabilities
ExecutionBinding
Job
ExecutionProvenance
```

Freeze `ExecutionBinding` at submission, linking at least:

```text
program_identity
compilation_identity
execution_identity
requested_target
actual_target
external_backend_id
provider_job_id
device_snapshot_id
idempotency_key
contract_version
```

Portal IDs belong in adapters and task records, not Circuit/IR or Quafu-specific
`fq.run` arguments.

### 4.3 Execution Data Plane

Keep quantum and classical interfaces distinct:

```text
QuantumProvider
  discover / submit / status / cancel / result

ComputeFabricAdapter
  probe / allocate / launch / status / cancel / logs / checkpoint
```

Share minimal Job, identity, error, and observability semantics while retaining
specific extensions.

### 4.4 Results

Separate stable measurement/count/sample/shot/status and logical-wire/classical-bit
ordering from execution facts and provider extensions. Execution facts include
requested/actual target, fallback, device, and calibration snapshots. Raw provider
fields, diagnostics, and proprietary results must not alter stable result equality
or program identity.

## 5. Overall Sequence

Establish essential IR contracts, validate Quafu vertically, generalize across
quantum clouds, then build classical cloud and hybrid orchestration. Quafu need
not wait for all IR phases; cloud details must not determine program semantics.
IR remains a continuous workstream:

```text
IR / compiler lane                    Execution / ecosystem lane

Phase 0 baseline/decisions       <--> Stage 0 baseline/owners/authorization
          |
Phase 1 QuantumIR skeleton      <--> Stage 1 program identity/semantics
          |                               |
Phase 2 compiler migration      <--> Stage 2 real Quafu vertical slice
          |                               |
Phase 3 TargetIR/ABI             <--> Stage 3 conformance/second quantum cloud
          |                               |
Phase 3 completion              <--> Stage 4 unified Target/Job/entry
          |                               |
Phase 4 ProgramIR/dynamic       <--> Stage 5 classical cloud/workflow foundation
          |                               |
Phase 4 production validation  <--> Stage 6 hybrid orchestration/certification
          |
Phase 5 advanced capabilities mature independently,
not as one mandatory bundle for the first production release
```

### 5.1 Required IR Depth

| Product objective | Minimum IR phase | Reason |
| --- | --- | --- |
| Minimal Quafu integration | Phase 1 | Compatible deployment can verify identity, ordering, task boundaries |
| Unified static compilation | Phase 2 | Compilation, routing, emitters share one pipeline |
| Multiple quantum clouds and near-transparent switching | Phase 3 | TargetCapabilities, TargetIR, executable ABI |
| Dynamic circuits and hybrid workflows | Phase 4 | Measurement def-use, control flow, ProgramIR |
| QIR, timing, advanced control | Relevant Phase 5 item | Independent certification, not all-or-nothing Phase 5 |

Stage 1 exit permits platform validation; it does not finish the IR roadmap.
Final acceptance requires production validation through Phase 4 and individual
gates for every claimed Phase 5 capability.

## 6. Stages and Exit Gates

### Stage 0: Facts, Owners, and Authorization

Protect stable contracts from changes that conceal implementation problems.
Establish a clean branch/test/maturity baseline; assign IR, Provider, Runtime,
platform, and API owners; require separate API proposals; map issues/milestones
to code evidence. Shared adapter repositories contain only OpenAPI, schemas,
mocks, documentation, and necessary thin bindings.

Exit gates:

- [ ] Every workstream has an owner, evidence location, dependencies, and rollback.
- [ ] Stable Core/candidate API boundaries are machine-checkable.
- [ ] The full current test baseline is reproducible in Docker.

### Stage 1: IR Phase 0 Exit and Phase 1 Skeleton

Establish program authority and identity layers independent of cloud details.

1. Complete Phase 0 gates in `MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md`.
2. Obtain owner review and explicit Phase 1 authorization.
3. Implement opt-in QuantumIR skeleton, verifier, identity, importer.
4. Fix logical/physical/classical/ancilla and measurement mapping boundaries.
5. Separate program, compilation, and execution identities.
6. Differentially verify states, measurements, gradients, and ordering.
7. Preserve the default `fq.run` fast path and the ability to disable the new path.

Exit gates:

- [x] All Phase 0 machine evidence passes.
- [x] Inputs outside Phase 1 support fail closed.
- [x] Public API, `CircuitIR` 1.0, and default execution remain unchanged.
- [x] Identical logical programs produce stable program identity.
- [x] Providers, credentials, queues, and backend IDs stay outside internal program IR.

### Stage 2: Minimal Real Quafu Loop

Validate program-to-result boundaries on the first real platform before freezing
a general public Provider API. Start IR Phase 2 using real deployment corpora.

```text
Circuit/CircuitIR
  -> target capability preflight
  -> deployment compilation and artifact
  -> Quafu submit/status/cancel/result
  -> counts and measurement normalization
  -> immutable execution binding and provenance
```

Tasks:

- Complete Proposal 012 P0 and remove retired `fq.experimental.QPUTwin` from real deployment.
- Distinguish portal `external_backend_id`, internal target, actual device, and `provider_job_id`.
- Test asymmetric bit ordering, shot accounting, failure, cancellation, and timeouts.
- Return `counts_bit_order`, program/calibration/noise-model identities, and physical qubits.
- Add idempotency keys; distinguish accepted submission from successful execution.
- Test real installation/end-to-end behavior without copying FlagQuantum into adapter repos.
- Migrate canonicalization, decomposition, routing, and emitters into opt-in Phase 2.
- Add static Quafu deployments to legacy/new differential corpora. Cloud success
  cannot replace compilation equivalence tests.

Exit gates:

- [ ] Every Issue #4 P0/P1 item has evidence or an explicit follow-up issue.
- [ ] Submit, poll, and retrieve using a real portal-registered backend ID.
- [ ] QPU, ideal simulation, and noisy simulation provenance remains distinct.
- [ ] Fallback defaults off and records reasons when enabled.
- [ ] Credentials stay out of IR, results, logs, and serialized deployment assets.
- [ ] Phase 2 scope, differential oracle, performance budget, and rollback switch exist.

### Stage 3: Provider Conformance and Second Quantum Cloud

Complete Phase 2 and enter Phase 3. Validate TargetIR/executable ABI against
cross-platform semantics rather than renamed Quafu concepts.

Tasks:

- Build one Provider conformance suite and minimum Pending/Running/Finished/Failed/
  Cancelled states.
- Establish candidate error categories while retaining raw provider errors.
- Align qubits, gates, topology, shots, dynamic circuits, result types, and calibration versions.
- Normalize logical/physical and measurement/classical mappings.
- Choose a second real platform with substantially different task/device/result models.
- Keep vendor fields namespaced without expanding the minimum public contract.
- Complete Phase 2 passes, routing, QASM/QCIS emitters, performance gates.
- Start minimal TargetCapabilities, TargetIR, ExecutableArtifact, RuntimeAdapter contracts.
- Share artifact/runtime boundaries across local, Quafu QASM, and a different target.

Exit gates:

- [ ] Same conformance suite passes for local, Quafu, and second providers.
- [ ] At least two real platforms complete controlled end-to-end jobs.
- [ ] Capability incompatibility fails before submission.
- [ ] Generic results require no provider-specific fields.
- [ ] Hardware certification is separate from mock/CPU contract evidence.
- [ ] IR Phase 2 formally exits.
- [ ] Phase 3 minimum target legality/artifact identity passes three target classes.

### Stage 4: Unified Target, Job, and Execution Entry Candidate

Complete Phase 3 and support near-transparent code-level switching with visible
execution facts.

Tasks:

- Build internal `ExecutionTarget`, `ExecutionPolicy`, `BackendResolver`, and unified `Job`.
- Support `strict`, `fallback`, and `auto` policies.
- Share status, cancellation, waiting, and result semantics across synchronous
  simulators and asynchronous QPUs.
- Add persistence, idempotency, retries, recovery, irreversible terminal states.
- Use queue/cost/quota/capability in planning, not program identity.
- Validate in candidate/non-root/deployment namespaces before proposing `fq.submit`.
- Obtain separate approval for stable `fq.run` signature/result changes.
- Complete compatibility/migration of TargetIR, ExecutableArtifact, RuntimeAdapter,
  and DeploymentPackage.
- Keep Target/Job consumers of artifacts/requests, outside program IR.

Exit gates:

- [ ] Same Circuit runs locally, on Quafu, and on the second cloud without rewriting.
- [ ] Generic result-reading semantics are consistent.
- [ ] Actual backend, fallback, calibration snapshots are queryable.
- [ ] Network retries do not duplicate paid QPU submissions.
- [ ] Querying recovers after task-process restart.
- [ ] Stable API expansion has usage evidence and migration review.
- [ ] Phase 3 exits with verified ABI across local, QASM, non-QASM targets.
- [ ] Tampered artifacts, capability mismatch, and credential boundary violations fail closed.

### Stage 5: Classical Cloud Control Plane

Connect local/distributed Runtime to actual resource scheduling and start Phase 4
ProgramIR/classical control for dynamic circuits and workflows.

Order:

1. `LocalComputeFabric` for existing CPU/GPU fast paths.
2. `DockerComputeFabric` with image/dependency/device/result identity.
3. One real Slurm or Kubernetes adapter.
4. A second scheduler to validate the abstraction.
5. Public-cloud-specific resources and elasticity later.

Tasks:

- Define CPU, memory, accelerator, world-size, node, and interconnect requests.
- Identify images, code, data, checkpoints, and execution.
- Support launch/status/cancel/logs/checkpoints/recovery.
- Record ownership, communication semantics, peak memory, actual devices.
- Distinguish sharding, data parallelism, and replication.
- Preserve local CPU/single-GPU latency.
- Implement bounded Phase 4 functions, blocks, measurement def-use, conditional,
  reset, and loops.
- Establish ProgramIR-to-QuantumIR lowering and OpenQASM 3 profile conformance.
- Keep Pod/Slurm IDs and resource state outside ProgramIR.

Exit gates:

- [ ] Same request runs on Local, Docker, and one real cluster.
- [ ] Failed jobs resume from checkpoints or report nonrecoverability explicitly.
- [ ] Execution identity includes image, code, environment, topology.
- [ ] Distributed claims pass accelerator/multinode evidence review.
- [ ] Cost, logs, and resource release are traceable.
- [ ] Bounded Phase 4 dynamics pass verifier, statistical oracle, batch performance gates.
- [ ] Dynamic program versus resource-workflow ownership is fixed.

### Stage 6: Hybrid Orchestration and Production Certification

Complete Phase 4 production validation for sustained QPU–GPU/NPU workflows.
Phase 5 timing, QIR, gradient/distributed lowering, and pulse references mature
independently according to hardware/ecosystem needs.

Initial scenarios: VQE/QAOA optimizer/measurement loops; ideal/noisy/calibrated
hardware comparisons; accelerator training with QPU validation/inference; error
mitigation DAGs; comparable device evaluations of the same logical program.

Tasks:

- Recoverable DAGs with classical, QPU, and conversion nodes.
- Late parameter binding, batches, quotas, partial failures.
- Intermediate schemas, privacy, retention, replay rules.
- Cost/queue/precision/completion-time policies.
- Real workloads, long runs, fault injection, version-upgrade tests.
- Shared static/dynamic semantics without provider-private feedback metadata.
- Separate schema, oracle, target evidence, performance baseline, and maturity
  for every claimed Phase 5 sub-capability.

Exit gates:

- [ ] At least one hybrid algorithm recovers after process/node failure.
- [ ] Simulation and QPU results stay distinguishable in UI, API, and audits.
- [ ] Parameters, programs, compilation, snapshots, results have complete lineage.
- [ ] At least one quantum cloud and classical cluster meet release certification.
- [ ] Public documentation claims only capabilities supported by corresponding evidence.
- [ ] Phase 4 exits with shared static/dynamic semantics.
- [ ] Public Phase 5 sub-capabilities are independently certified; others remain experimental or undiscoverable.

## 7. Cross-Stage Workstreams

### 7.1 Capabilities

Converge incrementally rather than freezing all fields at once:

```text
program formats
native gates and topology
qubit/classical-bit limits
shots range and step
dynamic circuit features
result types
gradient/parameter binding support
queue/cost/quota hints
calibration and capability snapshot identity
```

Static semantics belong in target/compiler contracts. Volatile queues, prices,
and inventory belong in resolver snapshots.

### 7.2 States and Errors

Use a closed state machine while retaining raw provider state:

```text
Pending -> Running -> Finished
                   -> Failed
                   -> Cancelled
```

Accepted cancellation is not completed cancellation. Distinguish validation,
capability, resource, authentication, quota, transport, provider-runtime, and
internal errors.

### 7.3 Fallback

- Default to `strict/forbid`.
- QPU-to-simulator substitution requires explicit permission.
- Real-hardware-required tasks never fall back.
- Re-run capability preflight and compilation after fallback.
- Record requested/actual targets, reasons, and semantic differences in provenance.
- Use distinct labels for ideal simulation, noisy simulation, and QPU results.

### 7.4 Security and Tenancy

- Resolve credentials through secret references, outside programs/ordinary metadata.
- Bind backends to portal instance, tenant, and external backend ID.
- Redact logs, exceptions, and traces by default.
- Apply the same tenant authorization to submission, cancellation, and result reads.
- Keep raw third-party payloads outside core Runtime.

### 7.5 Observability and Reproducibility

Each job links program/compilation/execution identities, requested/actual targets,
provider jobs, devices/calibration snapshots, shots/seeds/bit ordering/measurement
maps, software/images/dependencies, world size/ownership/memory/communication, and
fallback/retry/cancellation/error timelines.

## 8. Testing and Certification Matrix

| Level | Establishes | Does not establish |
| --- | --- | --- |
| Schema/mock | Request/response/failure structure | Real platforms, physical precision, capacity |
| Unit | Conversion, state machines, bit order, idempotency | Network/resource behavior |
| Local integration | Compilation-to-result loop | Real QPU or multinode capabilities |
| Provider sandbox | SDK/API compatibility, authentication | Production-device performance |
| Real QPU | Submission, states, results, device identity | Generalization to other providers |
| Single-node accelerator | Local performance/device path | Multinode scaling |
| Multinode/multi-GPU | Actual sharding, communication, capacity scaling | Physical QPU correctness |
| Long-run/fault injection | Recovery, idempotency, resource release | Scientific accuracy alone |

Provider conformance covers asymmetric bit order, shot conservation,
program/deployment identities, unsupported gates/qubits/topology, missing
measurements, all five states, timeout/retry/idempotency, calibration/actual device,
and credential/sensitive-field protection.

## 9. Priorities

Current recommended effort:

```text
45%  IR Phase 2 compiler migration and differential verification
45%  Real Quafu vertical slice
10%  Target/Job/Provider conformance candidate design
```

After Stage 2:

```text
40%  Provider conformance and second quantum cloud
35%  Unified Target/Job and task reliability
25%  First classical-cloud adapter validation
```

Before Stage 3 exit, provider count is not the primary success metric.

## 10. Deferred Work

Do not immediately publish ProgramIR/QuantumIR/TargetIR or stable root `fq.submit`.
Keep portal backend IDs out of `fq.run`, Circuit, and IR. Do not generalize
Quafu-specific states/errors/fields prematurely, build multiple cloud control
planes simultaneously, or silently replace QPUs with simulators. Do not call
mock/CPU/single-machine tests QPU/multi-GPU/multinode certification, copy simulation/
noise/scheduling code into adapter repositories, or rewrite stable APIs merely
for short-term interface uniformity.

## 11. Risks and Controls

| Risk | Consequence | Control |
| --- | --- | --- |
| Premature Provider API freeze | Coupling to first platform | Review after two different real providers |
| Cloud fields in IR | Invalid hash/cache/portability semantics | Separate program, compilation, execution identities |
| Silent fallback | Incorrect scientific provenance | Prohibited by default, explicit permission, full provenance |
| Open-ended status strings | Unreliable polling/recovery | Closed minimum state machine plus raw state |
| Duplicate retries | Duplicate charges/experiments | Idempotency, persistent binding, provider-job linkage |
| Parallel control-plane expansion | Many adapters without working loop | Validate one quantum cloud and scheduler at a time |
| IR overhead on local path | Poor developer experience | Opt-in, caching, bypass, performance gates |
| Simulation treated as hardware proof | Misleading capabilities | Tiered certification, auditable maturity |

## 12. Stage Decisions

Reviews require merged implementation, reproducible tests or actual platform
records, schemas/hashes/logs/provenance, explicit limitations/failure/rollback,
and written API owner approval for protected changes.

Design-only work, mock-only success, symmetric Bell/GHZ-only ordering tests,
provider classes without real jobs, forward-only evidence for training claims,
and relaxed snapshots/exceptions/schema tests do not establish completion.

## 13. Next Executable Batch

In order:

1. Quafu private bindings and asymmetric bit-order end-to-end contracts.
2. Minimum status/cancellation/error/idempotency/metadata loop.
3. Phase 2 scope, differential oracle, performance budget, rollback.
4. Compiler/routing/emitter migration using static Quafu corpora.
5. Provider conformance extracted from real Quafu evidence.
6. Second quantum cloud validating Phase 3 TargetIR/executable ABI.
7. Unified Target/Job review after Phase 3 and two-cloud validation.
8. First classical scheduler and bounded Phase 4 dynamic semantics.
9. Stable execution API proposals only after evidence stabilizes.

## 14. Final Acceptance

Completion is measured by user behavior, not directory or provider counts:

```python
circuit = build_once()

local_result = run_on_local(circuit)
cloud_result = run_on_classical_cloud(circuit)
qpu_result = run_on_qpu(circuit)
```

All three paths must provide unchanged Circuit/Module code, provably identical
logical input, traceable compilation/execution differences, consistent generic
results, transparent resources/fallback/noise/calibration, query/cancel/recovery/
audit, recoverable hybrid composition, and appropriately scoped evidence for
public claims.

IR Phases 0–4 must pass their formal machine-verifiable exit gates. Phase 1 alone
is not completion. Phase 5 need not finish as a bundle, but every claimed item
requires independent certification. Retiring compiler/dynamic paths requires an
explicit compatibility window, migration evidence, and rollback plan.

These conditions distinguish a unified quantum-classical platform from a
collection of execution backends and providers.
