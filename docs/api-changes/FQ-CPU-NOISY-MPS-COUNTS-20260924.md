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
- Stable 24-qubit `fq.run(...)` selection and ExecutionPlan JSON round trip.
- Existing MPS, modern-measurement, plan-contract, and seeded-sampling suites.
