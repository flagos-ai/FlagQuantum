# ARCH-007: Planning Information Between Simulation and Runtime

Status: Proposed

Date: 2026-09-03

Basis: Phase 0 Simulation, Runtime, and Platform inventories; no change to current planners or execution implementations.

## Context

Simulation owns numerical algorithm knowledge. Runtime owns execution organization
and final policy. Platform supplies actual device, kernel, precision, memory,
communication, and topology facts. Prohibiting cross-layer calls without defining
information exchange would force Runtime to duplicate numerical estimates or
Simulation to inspect environments and select devices beyond its authority.
Layers define ownership boundaries, not information isolation.

## Decision Candidates

Simulation supplies Runtime with Core-owned, versioned, read-only planning inputs
and outputs:

- workload features: qubits, depth, gate/observable shape, batch, shots, dtype,
  gradients, and dynamic behavior;
- algorithm constraints: exact/approximate conditions, supported gates, layout,
  numerical stability, gradient requirements, and truncation limits;
- resource requirements: state/workspace/peak memory estimates, communication
  primitives, temporary storage, and device capabilities;
- candidate partitionings: local execution, amplitude/site/bond sharding,
  slice/task partitioning, and their applicability conditions;
- cost estimates: computation, memory, communication, recomputation, and error
  costs, with model version, confidence, and unknowns.

Runtime combines these candidates with Platform discovery/evidence, user
requests/policy, resource leases, and failure/fallback authorization. It decides
the final backend, mode, placement, partition, and fallback, and records actual
choices in plans/results/evidence. Simulation does not select tenant resources
or physical devices; Runtime does not rewrite algorithm constraints.

## Prohibited Practices

- Simulation must not read cluster environments, initialize process groups,
  choose providers or release claims, or silently fall back.
- Runtime must not duplicate algorithm cost models, alter error/gradient
  constraints, or treat unknown estimates as exact capacity guarantees.
- Serializable planning contracts must not contain torch/JAX/vendor handles,
  credentials, or free-form dictionaries.
- Local kernel optimizations need not traverse the full planning stack. Changes
  without external semantic effects use the ARCH-008 fast path.

## Compatibility

Existing `ExecutionPlan`, planners, and Simulation calls remain unchanged. The
first version validates information shapes through adapters/fakes only. Changes
to stable plan fields, identity inputs, default choices, or failure stages require
an API Change Proposal. Existing backend-specific plans may supply adapters but
cannot become a second authoritative contract.

## Migration Sequence

1. Freeze current planner choices and Simulation characterization fixtures.
2. Core approves minimal workload/constraint/resource/candidate/cost value objects
   and unknown-value semantics.
3. A local statevector fake and the existing implementation produce candidates;
   a Runtime fake policy makes the final choice.
4. Add MPS/TN estimates and then Platform snapshots. Remove Simulation environment
   reads and duplicated Runtime estimates individually.
5. Distributed candidates enter default or certification paths only after passing
   the corresponding semantic and evidence gates.

## Acceptance Tests

- Identical inputs and model versions yield deterministic candidate identities;
  unknown costs remain unknown.
- Replacing a Simulation estimator does not change its Runtime consumer;
  replacing a Platform snapshot does not change Simulation.
- Runtime cannot choose approximation/fallback candidates when the user prohibits
  them.
- Actual paths/partitions agree with plans/results/evidence; estimates cannot be
  presented as measurements.
- Local fast paths do not initialize distributed execution. Sharded classification
  is not interchangeable with sliced or replicated classification.

## Open Questions

- Cost-model versions, confidence intervals, calibration data, and cache identity.
- Ownership of ranking across algorithms, and read-only inclusion of Compiler
  schedules in workload features.
- Whether Simulation expresses distributed communication as primitives or logical
  data movement, and how training optimizer ownership is represented.
