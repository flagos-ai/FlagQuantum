# FlagOS statevector F2 single-node scale profile

F2 extends the F1.1 two-card correctness result into a deterministic 2/4/8-card
single-node statevector scale ladder.  It remains entirely inside FlagQuantum:
the work changes neither Torch-FL nor FlagCX and requires no external pull
request.

The first frozen hardware result is recorded in
[FlagOS statevector F2 result on A800](FLAGOS_STATEVECTOR_F2_A800_RESULT.md).

Run the controller in an explicitly provisioned eight-device Torch-FL
environment:

```bash
python tools/validate_flagos_statevector_scale.py \
  --output flagos-statevector-scale-profile.json
```

The controller starts an independent bounded `torchrun` for each world size.
Every worker imports Torch-FL before PyTorch and initializes the public boundary
with `device="flagos"` and `backend="flagos"`.

## Measured matrix

Each of 2, 4, and 8 ranks executes both `complex64` and `complex128` for three
cases:

- `cross_shard_reference`: gates exercise every rank-address wire and the local
  shard is compared with the matching slice of a bounded CPU complex128
  reference;
- `persistent_layout_reference`: the same bounded correctness check runs with
  persistent wire layout and maps physical shard indices back to logical wire
  order before comparison. Its communication may be reported as layout
  exchanges with zero remaining distributed gates;
- `capacity_invariant`: a larger state is never materialized on CPU or gathered
  across ranks; it is checked through global norm, repeat determinism, strict
  shard ownership, communication counters, and per-rank memory records.

Every distributed result retains only `total_amplitudes / world_size`
amplitudes.  Python full-state reconstruction is not used.  The machine profile
records rank placement, local state and memory bytes, peak scratch space,
communication counts and bytes, dtype tolerances, reference scope, source
revision, Torch-FL revision, and environment identity. On a CUDA-backed
development reference, missing public FlagOS device names fall back to the
public CUDA device properties and are labeled
`device_identity_source="torch_cuda_reference_fallback"`; this does not infer
the provider's inner communication route.

Each case labels its memory evidence. `provider_peak_allocator_bytes` is used
only when the public FlagOS allocator exposes a positive peak measurement;
otherwise `runtime_accounted_state_scratch_workspace` reports the sum of owned
state, executor peak scratch, and reserved exchange workspace. The latter is
runtime accounting, not a claim about physical allocator peak usage.

The global norm check gathers one FP64 scalar from each rank and sums those at
CPU precision. This is at most eight scalar values, never amplitudes. It keeps
the statevector check independent of a provider implementation that may narrow
real-valued `all_reduce(SUM)` while leaving complex128 state communication
accurate; the selected reduction method is recorded as `norm_reduction`.

## Fail-closed meaning

The schema is `flagquantum_flagos_statevector_scale_profile_v1`.
`scale_ladder_accepted=true` means only that this exact single-node forward
matrix passed at the public FlagOS boundary.  Even a complete result retains:

```text
flagcx_route_verified = false
host_staging_observed = null
communication_claim_allowed = false
scalability_claim_allowed = false
production_support_claim_allowed = false
release_gate_allowed = false
```

F2 does not inspect provider-private objects, infer a FlagCX route, prove the
absence of provider-internal host staging, measure a single-device capacity
failure, or cover sharded backward and optimizer ownership.  It is therefore
accelerator-backed development evidence, not a production or release
certification.  F3 is responsible for the sharded gradient and optimizer loop.

The pure report contract is covered by
`tests/unit/test_flagos_statevector_scale_profile.py`.  The hardware integration
test is explicitly opt-in with `FLAGQUANTUM_TEST_FLAGOS_SCALE=1`.
