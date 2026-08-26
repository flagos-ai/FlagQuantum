# FlagOS statevector F3 result on A800

The final F3 profile passed on 26 August 2026 after the bounded F3.1 reverse
differential isolated and removed one invalid provider primitive from the
FlagQuantum reverse path. This is accelerator-backed development evidence for
single-node sharded gradients and optimizer trajectories. It is not FlagCX,
production, release, or general scalability certification.

## Frozen identity

- FlagQuantum source: `fa38063534edaaa5044c4cdcc8ade58013f55d0b`;
- Torch-FL source: `1b19383c24d9aa54637dd23557a5a20ef6d52ec5`;
- PyTorch: `2.11.0+cu130`;
- CUDA runtime: `13.0`;
- physical reference: NVIDIA A800-SXM4-80GB;
- topology: one node, independent 2-, 4-, and 8-rank runs;
- schema: `flagquantum_flagos_statevector_training_profile_v1`;
- raw artifact:
  `artifacts/flagos_statevector_training_f3_a800_20260826.json`;
- artifact SHA-256:
  `1aa945b2bc78dd5306e3234d5826028b3e7442fb91672c16c5b29ff16654460d`.

Torch-FL was built only in an isolated temporary validation directory with its
optional FlagGems C++ and Python kernels disabled. FlagQuantum did not modify
Torch-FL or FlagCX and did not add either project as a core dependency.

## Result

All 18 combinations passed: three world sizes, two complex dtypes, and direct
gradient, SGD, and Adam cases. Every run retained rank-local amplitude shards,
used a replicated gradient only after all-reduce, and reported both
`full_state_materialization=false` and
`backward_uses_full_state_replay=false`.

| Ranks | complex64 max gradient error | complex64 max parameter error | complex128 max gradient error | complex128 max parameter error |
| ---: | ---: | ---: | ---: | ---: |
| 2 | `6.67e-8` | `1.04e-8` | `0.0` | `0.0` |
| 4 | `6.67e-8` | `1.04e-8` | `0.0` | `0.0` |
| 8 | `6.67e-8` | `1.04e-8` | `0.0` | `0.0` |

The complex64 numbers are normal single-precision rounding, not evidence of a
hidden FP32 complex128 path. The complex128 direct-gradient error was exactly
zero for these circuits, while the largest complex128 trajectory value error
was `2.22e-16` and the parameter error was zero.

The rank-consistency diagnostic gathers one complex128 scalar per rank and
compares the real component on the CPU. It never gathers amplitudes and records
the method as `complex128_scalar_all_gather_then_cpu_compare`.

## F3.1 root cause and repair

The initial F3 snapshot failed before any distributed collective. The F3.1
differential reduced the error to `torch.conj` on FlagOS complex tensors: both
complex64 and complex128 returned an incorrect conjugate, with maximum error
approximately `1.2`. Matrix Hermitian transpose, complex matrix multiplication,
and absolute square remained correct.

FlagQuantum's reverse implementation needs only the real conjugate inner
product. It now computes that scalar with real arithmetic,
`sum(left.real * right.real + left.imag * right.imag)`, rather than dispatching
the broken provider conjugation. Unit tests prove mathematical equivalence for
both supported complex dtypes. The repaired differential passes RY, RZ
interference, RXX, CRX, and composite reverse circuits, including complex128 at
`1.11e-16` maximum gradient error.

The machine-readable differential remains intentionally failed at the provider
primitive level while reporting no FlagQuantum reverse failures:

- artifact: `artifacts/flagos_statevector_reverse_f31_a800_20260826.json`;
- SHA-256:
  `8ca955246f69b8d14289e2c2fa5f94cd358c8ad04bbc3673a25b990696703ae1`;
- `primitive_failures = [conj, conjugate_inner_real, conjugate_inner_sum]`;
- `reverse_failures = []`.

This distinction prevents FlagQuantum from silently certifying the wider
provider surface. No external repository change or pull request is required
for the bounded FlagQuantum repair.

## Independent regression evidence

The exact validation source passed 49 focused unit tests covering the F3
profile, reverse path, and optimizer path. A two-rank native CUDA/NCCL reverse
run also passed and produced gradients consistent with the CPU reference,
confirming that the provider-specific repair did not regress CUDA behavior.

## Claim boundary

The accepted F3 profile records:

```text
training_ladder_accepted = true
sharded_backward_profile_accepted = true
sharded_optimizer_profile_accepted = true
flagcx_route_verified = false
scalability_claim_allowed = false
release_gate_allowed = false
```

F3 therefore closes the bounded single-node FlagOS training gate only. Provider
identity, host-staging visibility, actual FlagCX routing, multi-node behavior,
capacity-failure evidence, performance, and release readiness remain separate
fail-closed gates.
