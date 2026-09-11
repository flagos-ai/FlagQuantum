# Runtime Decoupling Inventory

Status: current integration-branch boundaries (2026-09-08).

Path classification: this round adds only documentation and characterization
tests, changing no execution semantics. Covered local paths are
`single_device_fast_path`. The trajectory checkpoint example explicitly records
`replicated_per_rank` and is not distributed scalability evidence.

## 1. Conclusion

Runtime currently performs three levels of work:

1. Execution attempts already close to the target boundary, such as
   `execute_plan()` validating the environment, executing a fixed plan, and
   normalizing results.
2. Resource, distributed, training, checkpoint, recovery, and evidence components
   that are not yet unified.
3. Compilation, routing, noise lowering, mode selection, and plan construction
   that call Compiler implementations directly for historical entry-point compatibility.

`architecture.toml` currently permits seven Runtime paths to depend directly on
Compiler. There are two categories:

- `runtime/execution.py` and `runtime/planner/__init__.py` are explicit
  program-to-plan orchestration entry points and may compose Compiler before execution.
- The other five support backend planning, noise, or dynamic-circuit compatibility
  entry points that still accept raw programs. Close them individually when actual
  call chains already carry compiled IR.

The goal is not zero Compiler dependencies across all Runtime. It is to prohibit
execution kernels, distributed lifecycles, Simulation, and Providers from
recompiling during execution. Do not duplicate scheduling, routing, or noise
lowering to shorten the allowlist, or introduce `ExecutablePlanContract`,
free-form dictionaries, or forwarding layers solely to remove imports.

## 2. Current Execution Lifecycle Entry Points

### 2.1 Planning

| Entry point | Current responsibility and facts | Boundary assessment |
| --- | --- | --- |
| `fq.plan()` / `flagquantum.runtime.planner.plan()` | Produces stable `ExecutionPlan` with program/options/environment/compiler fingerprints and final decisions | Runtime owns planning policy; `compilation` temporarily holds shared plan artifacts |
| `flagquantum.runtime.execution.run()` | Programs call `plan()` first; plans go directly to `execute_plan()` | Convenience orchestration combines compilation requests and execution attempts |
| `run_native()` / `run_distributed()` | Compatibility entry points call `compiler.compile()`, `select_execution_mode()`, `plan_advanced()` internally | Historical coupling; eventually accept executable plans or explicitly request Compiler services |
| `flagquantum.runtime.planner_adapter` | Exposes capability queries and JAX training planning to Compiler | Sole declared narrow Compiler-to-Runtime seam; keep query-only |

### 2.2 One Execution

| Entry point | Current behavior | Evidence/gap |
| --- | --- | --- |
| `flagquantum.runtime.plan_execution.execute_plan()` | Validates environment, reads program/decisions, calls `run_native()` once, rejects executor plan replacement, normalizes outputs | Core attempt shape exists; plan reading/validation still depends on Compiler internals |
| `flagquantum.runtime.execution.run()` | Stable entry: plan programs, execute plans directly | Characterization tests fix no-replanning/no-recompilation behavior for `fq.run(plan)` |
| `run_advanced()` | Experimental entry with backend controls and normalized results | Not a stable durable-task interface |
| `run_native()` | Dispatches local statevector, MPS, TN, noise, JAX, and other modes | Oversized and combines compilation, selection, execution; primary decoupling target |
| `run_distributed()` / `DistributedExecutor` | Distributed statevector entry and local development simulation dispatch | Preserve distinctions among `single_device_fast_path`, development semantic evidence, and real sharding |
| `run_target()` | Sparse target-driven selection and execution | Temporarily owned by Execution Provider; inventory only this round |

### 2.3 Sessions and Resources

No generic `Session`, `RealtimeSession`, resource lease, or `ExecutionAttempt`
state machine currently spans the full execution lifecycle. Existing concepts are:

- `OperatorBackendSession`: optional operator-backend activation state returned by
  `operator_backend()`, covering only temporary in-process replacement.
- `runtime_config()`, `runtime_backend()`, `runtime_dtype()`: in-process configuration contexts.
- `init_torch_distributed()` / `destroy_torch_distributed()`: process-group resources.

Do not present `OperatorBackendSession` as a product-level Runtime Session. A
target Session manages short-lived resources, caches, device contexts, and cleanup
leased by Compute Service to one or more attempts. It does not own tenants,
queues, billing, or durable task state.

### 2.4 Distributed Organization

- `runtime.distributed.protocols` defines `DistributedExecutionRequest`,
  `DistributedExecutionRecord`, and executor protocols; these remain internal,
  relatively weak record shapes.
- `runtime.distributed.models` owns process groups, rank placement, shard/task
  ownership, communication hierarchies, and summary metadata.
- `runtime.executors.mps.execution.run_distributed_mps()` orchestrates distributed
  MPS. Numerical-directory ownership migration requires separate Simulation coordination.
- `runtime.executors.tensor_network.execution` orchestrates slice tasks and
  reductions but still directly depends on Compiler TN memory-calibration records.
- Distributed results must report `world_size`, `local_world_size`, `node_count`,
  rank ownership, memory, communication, `distribution_semantics`,
  `scalability_claim_allowed`, and blockers. CPU characterization proves contract
  semantics, not multi-GPU capacity expansion.

### 2.5 Training

- Stable `fq.train()` / `runtime.training.train()` orchestrates the PyTorch loop:
  `Module.execute()` -> objective -> backward -> optimizer step -> detached result/callback.
- `runtime.module.Module.execute()` is the PyTorch-facing single-step entry point.
- `runtime.executors.mps.compiled_training` and backend training implementations
  have parallel paths. `runtime/executors/**` temporarily belongs to Simulation
  in this round and is not modified.
- Training does not uniformly produce `ExecutionRecordContract` or associate each
  step with an `attempt_id`, a major lifecycle-evidence gap.

### 2.6 Checkpoint and Recovery

| Entry point | Current capability | Limitation |
| --- | --- | --- |
| `save_training_checkpoint()` / `load_training_checkpoint()` | Atomic model/optimizer, IR/workload identity, precision, RNG, topology, and runtime-plan storage; in-memory rollback on restore failure | No durable task/attempt identity or external persistence-reference contract |
| `save_trajectory_checkpoint()` / `load_trajectory_checkpoint()` | Safe atomic tensor/primitives-only trajectory checkpoints | Trajectory execution only |
| `TrajectoryCheckpoint.pending_ids()` | Restores rank-owned pending trajectories from completion/failure records, optionally retrying retryable failures | Local recovery, not durable task retry scheduling |
| `merge_trajectory_checkpoints()` | Merges disjoint rank-local progress and online statistics | Not a task state machine spanning attempts |

Recovery mainly reconstructs local execution state from versioned checkpoints.
There is no unified attempt retry, resource rebinding, idempotency, cancellation,
timeout escalation, or durable cross-process recovery entry point.

### 2.7 Observation and Result Aggregation

- `StrictExecutionScope`, `RouteExplanation`, `FallbackEvent`: enforce fallback
  policy and retain route audits; host debug fallback revokes production eligibility.
- `PerformanceMonitor`, `classify_no_progress()`: performance sampling and stall classification.
- `RuntimeProvenance`, `create_evidence_artifact()`, `verify_evidence_artifact()`:
  immutable, signable runtime evidence envelopes.
- `record_execution()`: builds `ExecutionRecordContract` from Core
  `RuntimePlanContract`, observations, ownership, measurements, failures, and provenance.
- `_normalize_execution_output()`, `normalize_execution_result()`,
  `execute_measurements()`: normalize backend outputs into stable `ExecutionResult`.
- `TensorWelford`, `merge_trajectory_statistics()`: online trajectory statistics
  and cross-rank aggregation.

These components lack one authoritative attempt coordinator. Successful `fq.run()`
returns `ExecutionResult`; failures raise `ExecutionError`. Callers still separately
assemble `ExecutionRecordContract`, fallback events, checkpoint references, and
signed evidence.

## 3. Runtime-to-Compiler Dependency Classification

Categories:

- **Shared data contract**: needed by Runtime and Compiler, but its authority
  should not belong to Compiler implementation.
- **Compiler service call**: Compiler behavior requested through a stable service
  port instead of implementation imports.
- **Incorrect internal implementation call**: Runtime/planning reuses private
  Compiler algorithms or validation.
- **Historical noise/routing coupling**: old directories and compatibility entry
  points mix responsibilities in one file.

| Registered `architecture.toml` path | Actual dependency | Primary category | Target replacement |
| --- | --- | --- | --- |
| `runtime/executors/statevector/noisy.py` | `lower_noise_model()` | Raw-program expert entry | Preserve entry behavior; main execution calls the already-lowered internal function |
| `runtime/executors/statevector/planning.py` | `schedule_layers()` | Raw-program backend planning | Currently accepts Circuit/IR; no duplicate scheduling or new plan contract solely for this |
| `runtime/dynamic/routing.py` | `CouplingMap`, `route_to_topology()` | Dynamic compilation compatibility | Route before execution; move orchestration as the real dynamic call chain consolidates |
| `runtime/execution.py` | `compile()`, `lower_noise_model()` | Top-level execution orchestration | Valid composition root; dispatch must not recompile an existing plan |
| `runtime/noise_registry.py` | `lower_noise_model()` | Stable density-matrix convenience entry | Plan execution already consumes lowered IR; retain raw-program convenience entry temporarily |
| `runtime/planner/__init__.py` | `compile()`, `lower_noise_model()`, `schedule_layers()` | Top-level planning orchestration | Valid program-to-plan boundary |
| `runtime/planner/noise_selection.py` | `lower_noise_model()` | Raw-program noise candidate evaluation | Consolidate once callers generally have lowered IR; no hidden parameters or duplicate models |

Stable distributed-statevector `ExecutionPlan.layers` can reuse Compiler layering,
but does not contain amplitude shards, fusion blocks, communication segments,
buffers, node topology, or JAX preflight results. `run_distributed()` therefore
still builds a backend-specific plan; JAX summaries also rebuild the same topology.
Reusing `layers` alone cannot eliminate this duplication. Until backend plans enter
an approved serializable executable-plan extension, do not add hidden public
parameters or a second distributed-plan contract.

### 3.1 Unacceptable Replacements

- Copying `ExecutionPlan`, `NoisyExecutionPlan`, `BackendSelection`,
  `TNWorkingSetCalibration`, or `CouplingMap` into Runtime.
- Hiding cross-layer contracts in `dict[str, Any]`.
- Adding Runtime-to-Compiler re-exports to pass architecture checks.
- Copying scheduling, routing, noise lowering, or backend cost selection into
  private Runtime implementations.
- Editing architecture allowlists before replacement contracts and conformance tests exist.

## 4. Runtime Attempts and Compute Service Durable Tasks

```text
Compute Service durable task
  ├─ tenant/auth/quota/budget/queue/cancel/retry policy
  ├─ task_id + idempotency + durable attempt history
  └─ leases one attempt request
       ↓
Runtime execution attempt
  validate immutable executable plan
  acquire bounded session/resources
  execute exactly once
  observe routes/fallback/memory/communication/progress
  emit result OR typed failure + optional checkpoint
  assemble attempt evidence and release resources
       ↑
Compute Service persists outcome and decides whether/when to create another attempt
```

Proposed Runtime attempt boundary:

- Input: an immutable compiled executable plan with verifiable identity, an
  attempt resource lease, and `attempt_id`.
- Bounded kernel retries or collective recovery within an attempt require
  explicit plan/policy authorization and complete disclosure.
- Completion means a successful result, typed failure, recoverable checkpoint,
  or completed cleanup after cancellation.
- Each attempt emits observations, ownership, fallback/degradation, failure,
  checkpoint references, and provenance, not just tensors on success.
- A Runtime Session reuses bounded resources across consecutive attempts under
  an external lease; it is not a tenant task database.

Compute Service durable-task responsibilities:

- Tenancy, authentication, authorization, quotas, budgets, billing, queue priorities.
- Durable `task_id`, idempotency, cross-attempt history, retry/backoff/cancellation policy.
- Choosing checkpoints and creating attempts after node failure or service restart.
- Aggregating attempts/provider jobs and exposing durable protocol state through
  MCP/REST/gRPC.
- Recording provider job identity without presenting it as Runtime `attempt_id`.

Runtime must not duplicate this durable control plane. Compute Service must not
bypass Runtime plan validation, fallback disclosure, or evidence assembly.

## 5. Characterization Evidence

Added `tests/team/runtime/test_execution_lifecycle_characterization.py`:

| Scenario | Established current behavior |
| --- | --- |
| Success | `fq.run(plan)` retains the same plan without replanning/recompiling and returns a normalized result summary |
| Failure | A kernel failure launches once, becomes `ExecutionError`, and preserves its cause; no hidden retries |
| Fallback disclosure | Host debug fallback is recorded, revokes production eligibility, and enters Runtime provenance |
| Checkpoint/recovery | Atomic trajectory checkpoints restore identity, progress, retryable/terminal pending IDs, and distribution metadata |
| Result evidence | `record_execution()` retains successful measurement evidence or failure codes/retryability/blockers for the same plan |

These characterize existing behavior. They do not establish a unified attempt
coordinator or GPU, multinode, or scalability release evidence.

## 6. Current Consolidation Principles

Do not introduce Core plan contracts merely to reduce import counts. Propose a
new contract through the public API process only when existing `CircuitIR`,
`ExecutionPlan`, and result types cannot express a real user workflow.

Consolidate one actual call chain at a time: compile above, consume existing IR
below, verify no recompilation, then remove one allowlist entry. If an entry must
still accept raw programs, retain the controlled dependency and stop splitting.

## 7. Compatibility Execution Entry Points (2026-09-06)

`run_native()` and `run_distributed()` remain protected Runtime expert interfaces.
`run_advanced()` serves only internal backend characterization and distributed
tests. `fq.experimental.execution` is explicitly empty, so `run_advanced()` is no
longer in `runtime.execution.__all__` and must gain no product consumers. It remains
temporarily to avoid rewriting many low-level tests at once, not as a new public
entry point. Migrate real consumers gradually to stable `fq.run()` or domain expert
interfaces; remove the implementation once no consumers remain.

## 8. Removal of the `backends` Namespace (2026-09-09)

API Change Proposal 018 removed the forwarding-only `flagquantum/backends/`
package. It mixed Runtime execution, device policy, and Simulation numerics under
one name without a distinct domain responsibility.

| Authoritative entry | Sole responsibility | Does not own |
| --- | --- | --- |
| `flagquantum.runtime` | Backend-native execution, target calls, device policy | Numerical algorithms or device SDKs |
| `flagquantum.simulation.mps` | MPS numerics | Resource planning or device selection |
| `flagquantum.simulation.tensor_network` | TN numerics and expert queries | Resource planning or device selection |
| `flagquantum/compute/` | Compute platforms directly controlled by this process | External submission, simulation kernels, Runtime scheduling policy |
| `flagquantum/remote/` | Control-plane adapters for real QPUs, remote GPU/HPC services, cloud platforms | Local device lifecycles, simulation kernels, Runtime scheduling policy |

Repository consumers have migrated to these authoritative entries. No compatibility
package remains, and these interfaces are not duplicated elsewhere.
