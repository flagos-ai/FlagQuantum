# Discrete-Event Simulation Framework For Distributed FlagQuantum Development

## Goal

Provide a discrete-event simulation (DES) framework that lets FlagQuantum
develop, benchmark-plan, and fail-closed validate distributed execution paths
without requiring immediate access to multiple GPUs or devices.

The DES layer is a development and verification surrogate. It is not a
performance claim engine and must never be promoted as production scalability
evidence by itself.

## Why This Fits FlagQuantum

FlagQuantum already has the right control-plane surfaces:

- planner-style summaries in `runtime/executors/statevector`,
  `runtime/distributed`, and `runtime/executors/jax`
- a shared distributed evidence contract in `runtime/audit`
- Phase 4 and Phase 5 fail-closed gates
- existing ownership, memory, communication, and blocker vocabularies

The missing piece is a runtime-neutral execution surrogate that can:

- simulate rank placement and message flow
- simulate memory reservations and peak lifetimes
- model optimizer/update ownership and synchronization
- emit the same summary shape as real or planned execution
- make incomplete distributed paths executable in development without faking
  production evidence

## Product Positioning

The DES framework should be described as:

- a distributed control-plane simulator
- a protocol validator
- a benchmark preflight and topology explorer
- a non-claimable development execution backend

It should not be described as:

- a production runtime
- a replacement for NCCL, XLA, or torch.distributed execution
- proof of multi-GPU throughput
- release-grade scalability evidence

## Design Principles

1. Reuse the current evidence contract.
2. Preserve current blocker vocabulary where possible.
3. Simulate one logical workload across explicit rank ownership.
4. Make all DES payloads clearly non-claimable.
5. Keep backend-neutral event semantics and only add backend-specific adapters
   at the edges.
6. Let the same circuit or training step run through real execution, preflight,
   or DES with minimal branching above the executor layer.

## Proposed Package Layout

Add a new internal package:

```text
flagquantum/runtime/des/
    __init__.py
    engine.py
    events.py
    topology.py
    resources.py
    ownership.py
    executors.py
    emitters.py
    scenarios.py
```

Recommended responsibilities:

- `engine.py`
  - event queue
  - simulation clock
  - event dispatch loop
  - deterministic ordering rules
- `events.py`
  - event dataclasses
  - communication, compute, allocation, barrier, failure, fallback events
- `topology.py`
  - rank placement
  - node grouping
  - link bandwidth and latency models
  - transport backend descriptors
- `resources.py`
  - per-rank memory accounting
  - temporary buffer lifetime tracking
  - peak memory and overlap calculation
- `ownership.py`
  - site shard ownership
  - bond ownership
  - parameter/gradient/update ownership
  - boundary-adjoint routing ownership
- `executors.py`
  - backend-family simulation entrypoints
  - statevector, MPS, TN, hybrid scenario drivers
- `emitters.py`
  - conversion from DES trace to current audit/runtime summary payloads
- `scenarios.py`
  - reusable scenario builders for tests and docs

## Core Data Model

### Simulation Context

The root object should carry:

- `world_size`
- `local_world_size`
- `node_count`
- `rank_placement`
- `network_backend`
- `bandwidth_by_tier`
- `latency_by_tier`
- `memory_capacity_by_rank`
- `state_mode`
- `backend_family`
- `claim_evidence_type="development_smoke"`
- `distribution_semantics`
- `scalability_claim_allowed=False`

### Event Types

The first version only needs a small event vocabulary:

- `ComputeStart`
- `ComputeEnd`
- `TensorAllocate`
- `TensorFree`
- `SendStart`
- `SendEnd`
- `RecvReady`
- `CollectiveStart`
- `CollectiveEnd`
- `Barrier`
- `OwnershipWrite`
- `FallbackAttempt`
- `BlockedExecution`

Each event should carry:

- `timestamp`
- `rank`
- `category`
- `bytes`
- `tensor_shape`
- `dtype`
- `dependency_ids`
- `instruction_id` or `gate_id`
- `route`
- `transport`
- `blockers`

## Backend-Family Scenario Coverage

### Statevector

DES should simulate:

- amplitude or qubit-address sharding
- all-reduce for scalar observables
- all-to-all or pair exchange for shard reshuffles
- per-rank state and buffer residency
- gradient redistribution path

It should emit:

- `rank_ownership`
- `memory_plan`
- `communication_plan`
- transport evidence
- Phase 4-compatible gradient and optimizer semantics

### MPS

DES should simulate:

- contiguous site shard ownership
- bond ownership
- boundary tensor exchange
- boundary adjoint return path
- owner-rank parameter VJP and optimizer update routing
- canonicalization/truncation placeholders with explicit blockers

It should emit:

- `site_shard_ownership`
- `bond_shard_ownership`
- `parameter_gradient_ownership`
- `boundary_gradient_ownership`
- `boundary_adjoint_exchange`
- `mps_backward_memory_plan`
- `mps_backward_communication_plan`
- MPS runtime summary via `_attach_mps_runtime_summary(...)`

### Tensor Network

DES should simulate:

- slicing plan
- rank/task assignment
- peak intermediate sizes
- reduction tree for partial outputs
- node tensor gradient routing back to parameterized gates

It should emit:

- slice ownership
- reduction communication evidence
- parameter pullback blockers
- explicit no-dense-statevector reconstruction semantics

## Integration Points

### 1. Runtime Selection

Extend runtime candidates with a DES-backed option for distributed modes that
currently return planning-only or blocked summaries.

Example intent:

- `jax_sharded_statevector_des`
- `jax_sharded_mps_des`
- `tensor_network_des`

These should report:

- `claim_evidence_type="development_smoke"`
- `distribution_semantics="sharded_across_ranks"` when ownership is explicit
- `scalability_claim_allowed=False`
- backend-specific simulated execution status

### 2. Existing Planner APIs

Add optional DES execution on top of existing plan builders rather than adding
new public planning APIs first.

Preferred shape:

- plan function creates ownership and topology plan
- DES executor consumes that plan
- DES emitter produces the same summary schema as a real executor

### 3. Audit Layer

DES should not bypass audit. It should flow through:

- `attach_distributed_evidence_contract(...)`
- `evaluate_statevector_training_claimability(...)` where relevant
- `evaluate_mps_backward_readiness(...)` where relevant

DES outputs should fail closed by policy:

- `claim_evidence_type="development_smoke"`
- `production_training_claimable=False`
- `release_gate_allowed=False`

## Evidence Policy

Every DES payload should include explicit simulation markers, for example:

- `execution_mode="discrete_event_simulation"`
- `simulation_scope="control_plane_and_resource_model"`
- `performance_semantics="simulated_not_measured"`
- `hardware_execution="not_executed"`

Recommended invariant:

- no DES payload may set `scalability_claim_allowed=True`
- no DES payload may use `claim_evidence_type="production_runtime"`
- no DES payload may satisfy release-gate benchmark promotion

## Minimal v1 Milestone

The first usable milestone should be intentionally narrow:

1. Statevector DES for shard ownership plus all-reduce/all-to-all planning.
2. MPS DES for two-rank contiguous shards with boundary exchange and owner-rank
   gradient/update flow.
3. Shared emitters that attach the current distributed evidence contract and
   Phase 4/5 summaries.
4. Test fixtures proving that DES and planner summaries agree on ownership and
   blockers for the same circuit shape.

This is enough to make distributed development materially better without
building a full cluster simulator.

## Suggested v1 Public Surface

Keep the first release internal or semi-internal.

Internal helpers are safer first:

- `flagquantum.runtime.des.execute_statevector_des(...)`
- `flagquantum.runtime.des.execute_mps_des(...)`
- `flagquantum.runtime.des.execute_tn_des(...)`

If a public API is eventually needed, prefer one wrapper:

- `fq.simulate_distributed_execution(...)`

That wrapper should return a normal audited summary object, not a custom DES
report format.

## Testing Strategy

### Unit

- event ordering is deterministic
- memory peaks match synthetic traces
- communication bytes aggregate correctly by tier
- ownership writes fail if a non-owner rank writes
- fallback attempts emit blockers and never silently claim success

### Contract

- DES summaries attach `distributed_evidence_contract_v1`
- Phase 4/5 gates remain non-claimable under DES evidence type
- blocker vocabulary remains aligned with current runtime summaries

### Differential

For small circuits with local executability:

- planner ownership matches DES ownership
- DES communication estimates match protocol records
- DES blocker set is a superset of real executor blockers where execution is
  incomplete

### Policy

- release gate rejects DES payloads
- benchmark audit classifies DES payloads as non-release evidence

## Implementation Order

1. Add core DES engine and event dataclasses.
2. Add shared topology, tier, and memory accounting helpers reusing existing
   `local_world_size` and node-count semantics.
3. Add emitter that produces `distributed_evidence_contract_v1` payloads.
4. Implement statevector DES v1.
5. Implement constrained MPS DES v1 aligned with the current runtime contracts.
   ownership protocol.
6. Add runtime-selection integration for DES candidates.
7. Add TN DES after statevector and MPS emitters stabilize.

## Expected Benefits

- Distributed development continues without immediate GPU access.
- Protocol regressions become testable in ordinary CI.
- Ownership, memory, and communication semantics become executable earlier than
  full backend implementation.
- Runtime-selection recommendations can be backed by more than static plans.

## Explicit Limits

- DES does not validate CUDA runtime behavior.
- DES does not validate NCCL or XLA collectives.
- DES does not prove throughput, latency, or scaling curves.
- DES does not replace real accelerator or multi-node benchmark evidence.
- DES should not be used to justify release-grade scalability claims.
