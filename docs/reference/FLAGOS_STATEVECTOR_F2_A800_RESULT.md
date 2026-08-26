# FlagOS statevector F2 result on A800

The FlagQuantum F2 single-node scale profile passed on 26 August 2026 for the
exact development snapshot below. This is accelerator-backed development
evidence for the public FlagOS statevector forward route, not FlagCX,
production, release, or general scalability certification.

## Frozen identity

- FlagQuantum source: `21e220960e9756882ece2932614f3cc4d368a49a`;
- Torch-FL source: `1b19383c24d9aa54637dd23557a5a20ef6d52ec5`;
- PyTorch: `2.11.0+cu130`;
- CUDA runtime: `13.0`;
- physical reference: NVIDIA A800-SXM4-80GB;
- topology: one node, independent 2-, 4-, and 8-rank runs;
- schema: `flagquantum_flagos_statevector_scale_profile_v1`;
- raw artifact:
  `artifacts/flagos_statevector_scale_f2_a800_20260826.json`;
- compact artifact SHA-256:
  `6e52215287bbbd15198bec9d8558935e19c9d21c07923e5306dbb7af9b93b899`.

Torch-FL was built only inside an isolated temporary validation environment
with its optional FlagGems C++ and Python kernels disabled. It was not added to
FlagQuantum's core dependencies, and neither Torch-FL nor FlagCX was modified.

## Result

All 18 combinations passed: three world sizes, two complex dtypes, and three
workload cases. Every execution retained a strict rank-local amplitude shard,
performed positive communication, reported `sharded_across_ranks`, and kept
`full_state_materialization=false`.

| Ranks | complex64 max reference error | complex128 max reference error | complex128 max norm error | Repeat error |
| ---: | ---: | ---: | ---: | ---: |
| 2 | `3.73e-9` | `3.47e-18` | `0.00` | `0.00` |
| 4 | `3.00e-8` | `5.72e-17` | `2.22e-16` | `0.00` |
| 8 | `6.01e-8` | `1.67e-16` | `3.33e-16` | `0.00` |

The 24-wire capacity invariant retained 16,777,216 total amplitudes without a
CPU reference or full-state gather. For complex128, local owned state decreased
from 128 MiB at two ranks to 64 MiB at four ranks and 32 MiB at eight ranks.
The corresponding per-rank runtime-accounted state/scratch/workspace values
were 1,152 MiB, 448 MiB, and 224 MiB. These are executor accounting values, not
provider allocator peak measurements and not performance claims.

An initial diagnostic exposed `5.96e-8` error from the CUDA-backed FlagOS
real-FP64 `all_reduce(SUM)` used only by the validation norm. State shards still
matched the complex128 CPU reference at approximately `1e-16`. The final
harness therefore gathers at most one FP64 norm scalar per rank and sums those
scalars on the CPU. It never gathers amplitudes, and records the method as
`fp64_scalar_all_gather_then_cpu_sum`.

## Claim boundary

The accepted profile deliberately retains:

```text
flagcx_route_verified = false
host_staging_observed = null
communication_claim_allowed = false
scalability_claim_allowed = false
production_support_claim_allowed = false
release_gate_allowed = false
```

The blockers remain provider identity, host-staging visibility, single-node
forward-only scope, absent single-device capacity-failure evidence, and absent
sharded backward/optimizer measurement. F3 should address the last boundary by
validating the sharded gradient and optimizer loop without reconstructing the
state.

That follow-up initially failed closed, then F3.1 isolated a broken FlagOS
complex-conjugation primitive and removed it from FlagQuantum's reverse path.
The final bounded F3 training profile passed. See
`docs/reference/FLAGOS_STATEVECTOR_F3_A800_RESULT.md`. This does not change the
remaining FlagCX, multi-node, scalability, production, or release boundaries.
