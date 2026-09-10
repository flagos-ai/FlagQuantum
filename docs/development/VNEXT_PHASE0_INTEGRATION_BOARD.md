# FlagQuantum vNext Phase 0 Integration Report

> Historical note: this report records the team names and code boundaries at
> Phase 0. Current code has reduced `Agent Services` to a few composed workflows
> in `flagquantum.services` and removed the unreleased `AgentApplicationService`
> serialization facade. Current rules are in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).

> Status: all eight first-round team deliveries reviewed, merged, and jointly verified.
>
> Integration branch: `codex/flagquantum-vnext-architecture`
>
> Shared governance baseline: `d7c56603e363bba95d5e98b9a77a75adb3c52e0d`
>
> Integration completed: 2026-09-03 (Asia/Shanghai)

## 1. Round Conclusion

Phase 0 completed factual inventories, boundary reviews, behavior freezes, and
integration verification for eight teams. Every team commit passed path-ownership
and architecture checks. Teams were merged individually in this dependency order:

```text
Core -> Compiler -> Runtime -> Simulation
     -> Platform -> Execution -> Ecosystem -> Agent Services
```

The primary deliverables are current-state inventories, contract proposals, and
characterization tests. They do not establish implementation of the proposed Core
contracts, Provider interfaces, or directory migrations. The only product change
was in Agent Services: validate required capabilities before planning a
`ProgramArtifact` and fail closed on unknown artifact schemas.

## 2. Team Deliveries and Merge Records

| Order | Team | Final team commit | Integration merge commit | Result |
| ---: | --- | --- | --- | --- |
| 1 | Core | `6cf86e15459b06ec5261ec24d63c53ccd3425756` | `83d45119` | Passed |
| 2 | Compiler | `c8d7ac3a1f5c5a00ab7fccec5940ccf22ac4d3b7` | `d6f43652` | Passed |
| 3 | Runtime | `adba20d6f22193ad0a00b1132518ea91205998da` | `991d309a` | Passed |
| 4 | Simulation | `5ab9c74e34743c7dd768cc179ad584f92bbb2a57` | `2d4b8948` | Passed |
| 5 | Platform Provider | `42aa894574133c59010d022bf4536573b1bc2e77` | `810f0b5a` | Passed |
| 6 | Execution Provider | `c3e9129a1244e820985ff36f7ccb6b824b7d9405` | `33d3b6a5` | Passed |
| 7 | Ecosystem | `0fca0f02705a7f8f9fdfb5b9719009f6d9f42354` | `c2f66391` | Passed |
| 8 | Agent Services | `669adfa7ddcb9c2c7e2aca8a3c5e7f703b1807f8` | `75693ca1` | Passed |

All team worktrees were clean at delivery verification. The eight branches shared
one governance baseline and changed nonconflicting paths. Integration reran all
already-merged team tests after each merge; no order-dependent regression appeared.

## 3. Unified Verification Results

### 3.1 Local Integration-Branch Verification

- Team scope policy: passed.
- Architecture boundaries: passed.
- Eight-team characterization tests: `41 passed, 2 skipped`; skips required
  optional Qiskit/PennyLane installations unavailable locally.
- Cross-domain Agent, Runtime, Platform, Deployment, Provider, and Interop tests:
  `118 passed, 1 skipped`.

### 3.2 Standard Linux Docker Gates

- `pr-runtime`: `175 passed, 33 skipped`, no failures.
- `pr-default`: `1831 passed, 12 skipped, 1 failed`.
- The sole failure was the existing compiler import/verify latency budget test,
  `test_approved_import_verify_budget_is_machine_enforced`. It reproduced on the
  shared baseline and multiple independent team branches under comparable
  environments. It was not introduced by these merges. This round changed no
  thresholds, snapshots, or related compiler implementation.

### 3.3 Specialized Environment Evidence

- Ecosystem ran `51 passed` interoperability tests in Docker with Qiskit 2.5.2,
  Aer 0.17.2, and PennyLane 0.45.1.
- Platform ran `15 passed` platform tests on each of `a800-node-0` and
  `a800-node-1`, and verified NCCL sharded-statevector correctness on two nodes
  with one GPU per node.
- Remote results establish only the NVIDIA A800 + CUDA/NCCL development path,
  not FlagOS, domestic-chip support, or production scaling efficiency.
- Execution and Agent teams did not present A800 verification as real QPU evidence.

## 4. Module Boundaries Frozen in the First Round

### Core

Candidate authority for stable cross-domain data semantics, identity,
serialization, and evidence. Same-named types have only been classified; merging
or deleting them requires a decision.

### Compiler

Owns program import, normalization, analysis, optimization, target legalization,
lowering, and executable artifact generation. `flagquantum.compilation` remains
the current stable entry point; `flagquantum._compiler` is the internal
consolidation candidate. This round did not switch the default compilation path.

### Runtime

Owns validation, orchestration, lifecycle, distributed organization, training,
checkpointing, recovery, observation, and result aggregation for one execution
attempt. Long-lived tenant jobs, authentication, billing, and persistent queues
are outside the main repository's Runtime.

### Simulation

Owns state evolution, MPS factorization/truncation, tensor contraction, noise
trajectories, and forward/backward numerical kernels. Device selection,
ranks/topology, communication lifecycles, checkpoint policy, and execution evidence
belong to Runtime/Provider.

### Platform Provider

Owns facts about devices, lifecycles, memory, streams/events, kernels, dtypes,
precision, topology, and communication capabilities. Platform reports hardware/
software capabilities; it does not submit remote quantum tasks.

### Execution Provider

Owns submission, status, cancellation, result retrieval, error mapping,
calibration identity, and execution evidence for complete execution targets.
Simulation, QPU, and Remote Service Providers share lifecycle semantics, but
cannot be forced to share every operation.

### Ecosystem

Owns boundary conversions for external formats, ML frontends, plugins, and
third-party SDKs. Qiskit, PennyLane, CUDA-Q/QX objects must not enter stable Core,
Compiler, Runtime, or Simulation contracts.

### Agent Services

Owns protocol-independent, deterministic capabilities, validate, plan, and
preflight application services. MCP/REST are external gateways. Tenancy,
authentication, quotas, billing, and long-lived jobs belong to an external Compute Service.

## 5. Confirmed Architecture Debt

1. Several similar types represent `ProgramArtifact`, capabilities, execution
   requests/plans/results/evidence. Reconcile fields, lifecycles, identities, and
   consumers before consolidation.
2. Runtime retains ten registered Runtime-to-Compiler dependencies, including
   shared contracts, internal calls, and historical noise/routing coupling.
3. Numerical algorithms and orchestration remain spread across `simulation` and
   `runtime/executors`. The first migration candidate is the single-device
   PyTorch dense statevector engine.
4. Platform and Execution Providers lack approved minimum Core contracts.
   Existing Provider/Extension protocols cannot simply become one universal interface.
5. `ExecutionResult`, `TargetExecutionResult`, and `DeploymentResult` coexist;
   status, errors, cancellation, bit order, and calibration evidence remain unaligned.
6. OpenQASM/QCIS, Qiskit Aer execution, and Braket IQM dynamic dialects still have
   cross-layer or duplicate implementations.
7. `flagquantum.services.capabilities()` still accesses the Runtime backend
   registry indirectly.
8. Default `CUDAPlatformRuntime.event()` events do not enable timing and cannot
   directly establish elapsed-time evidence.
9. A reproducible Compiler import/verify performance-budget failure remains in
   standard gates. Compiler must fix it separately without relaxing thresholds.

## 6. Second-Round Entry Order

The second round should not begin with large directory moves. Proceed through
small reviewable slices:

1. Integration/Core reconcile `ProgramArtifact` and metadata value domains.
2. Consolidate minimum `TargetCapabilities`, strictly separating requirements,
   discovered facts, and execution evidence.
3. Define minimum Execution Request and Result/Evidence boundaries.
4. Approve Platform and Execution Provider method sets after those types stabilize.
5. Runtime removes its first pure-type dependency on Compiler.
6. Simulation extracts the first replaceable single-device statevector engine.
7. Ecosystem moves Qiskit Aer execution to Execution Provider and consolidates
   duplicate format implementations.
8. Agent Services replaces indirect Runtime registry access with public capability snapshots.
9. Compiler separately fixes the import/verify budget and restores passing default gates.

Every slice still requires one responsible team, a minimal diff, contracts/
characterization tests first, team-scope and architecture gates, individual
integration-branch merges, and verification after each merge.
