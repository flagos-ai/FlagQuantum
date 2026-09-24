# API Change: CPU noisy-MPS counts through `fq.run`

## Status

Implemented for review on 2026-09-24 with explicit API-owner authorization.
This proposal extends the frozen Result/Measurement/Noise contract without
adding a new root-level function or changing `ExecutionOptions` fields.

## Problem

FlagQuantum already implemented noisy MPS trajectories, but their aggregated
result could not satisfy a `counts` or `sample` measurement. The stable
`fq.run(...)` path also attached every `NoiseModel` to density-matrix execution,
so an approximate route selected by the planner was not executable or preserved
across an `ExecutionPlan` JSON round trip.

This left users with two disconnected paths: the unified stable entry point for
small exact workloads and an expert Runtime function for larger MPS workloads.

## Decision

1. `MPSMonteCarloResult.sample(...)` samples the empirical mixture of retained
   trajectory states without constructing a dense state or density matrix.
2. Computational-basis samples apply independent or correlated readout
   confusion after quantum-trajectory evolution.
3. Measurement statistics disclose empirical-mixture semantics, completed and
   retained trajectory counts, and noise-model identity.
4. Stable `fq.run(..., mode="auto")` keeps exact density execution when it fits.
   When density exceeds `memory_limit_bytes`, MPS is selected only if
   `allow_approximate=True`.
5. The memory budget determines a finite MPS bond cap. The chosen representation,
   evolution semantics, 32-trajectory policy, seed, and error flags survive plan
   serialization and execute without replanning.
6. Explicit `mode="density_matrix"` remains exact and never falls back to MPS.
7. `flagquantum.services.run_managed_quafu_simulator(...)` is the deployment
   bridge for `quafu:Baihua-sim`, `quafu:Shenglian-sim`, and
   `quafu:Dongling-sim`. It binds each target to its physical-device
   calibration, calls the same stable exact/MPS policy, and returns a receipt
   containing the calibration and execution identities.
8. `QuafuProvider.fetch_calibration(device, calibration_id=...)` reads current
   or immutable historical Task API snapshots without sending credentials, so
   a job can be analyzed against the calibration identity stored in its receipt.

The bridge is intentionally service-side. Client `fq.run(target="quafu:...-sim")`
continues to use the Quafu task API, so task lifecycle and provenance are not
lost and a remote request is never silently converted into local work.

The initial stable policy fixes the trajectory count at 32. Fine-grained
trajectory, cutoff, and bond controls remain on the expert Runtime API until a
separate stable-options proposal defines their compatibility semantics.

## Compatibility

This is an additive success path. A stable auto-mode workload that fits the
density budget retains its exact behavior. A workload that previously exceeded
the declared density budget can now run only after the caller explicitly opts
into approximation. Plans remain schema version 1.0 because the representation
is derived from existing decision, memory, seed, and noise-extension fields;
no serialized field is added or reinterpreted incompatibly.

## Acceptance evidence

- Deterministic noisy-MPS counts and readout-confusion tests.
- Fail-closed sampling when trajectory states were not retained.
- Two-qubit noisy counts agree with the exact density oracle within statistical
  tolerance.
- CPU noisy-MPS count smoke tests at 16, 20, and 24 qubits.
- A 100-qubit low-bond full-register counts test that avoids the former
  signed-`int64` basis-index limitation; explicit integer-index samples still
  fail clearly above 63 qubits.
- Stable 24-qubit `fq.run(...)` selection and ExecutionPlan JSON round trip.
- Executable target-binding tests for Baihua-sim, Shenglian-sim, and
  Dongling-sim, plus exact-to-MPS routing and mismatched-calibration rejection.
- Existing MPS, modern-measurement, plan-contract, and seeded-sampling suites.

## Accuracy claim boundary

The acceptance evidence above establishes numerical agreement with the exact
density-matrix implementation for small circuits. A paired 2026-09-24 live run
also measured three circuits, one physical mapping, and two QPU repetitions for
each of Baihua, Shenglian, and Dongling. Mean local-MPS-to-QPU TVD was 2.08%,
2.48%, and 4.20%, respectively; this is workload-specific evidence, not a
device-wide QPU fidelity claim.

The checked-in artifact preserves complete provenance and discloses two
important limits: the public calibration payload lacks gate durations, so the
local MPS model excludes timing-derived T1/T2 relaxation; and the run uses
4,096 expert trajectories rather than the stable managed route's 32-trajectory
policy. A broader target release may claim hardware agreement only from paired
evidence using the complete calibration snapshot, identical physical mapping
and compiled circuit, repeated QPU jobs, and a predeclared workload suite. No
single percentage is transferable across devices, calibration times, mappings,
and workloads.
