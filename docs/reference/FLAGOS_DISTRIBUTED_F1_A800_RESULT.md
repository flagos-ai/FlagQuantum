# FlagOS distributed F1 result on A800

This record captures the bounded single-node, two-rank measurement performed
on `a800-node-1` on 2026-08-25. It is a failed conformance result, not a release
certificate or a FlagCX capability claim.

## Revisions and environment

- FlagQuantum: `c92f4347b329161701973d2956835d188f2fb30d`;
- Torch-FL: `1b19383c24d9aa54637dd23557a5a20ef6d52ec5`;
- PyTorch: `2.11.0+cu130`;
- CUDA runtime: `13.0`;
- driver: `580.126.20`;
- ranks: 2 on one node;
- devices: two NVIDIA A800-SXM4-80GB cards.

The Torch-FL revision is the public 2026-08-10 revision whose compatibility
contract names PyTorch 2.11.0. It was built only in an isolated disposable
container because current Torch-FL main pins PyTorch 2.10.x. This measurement
must not be represented as validation of current Torch-FL main.

The machine-readable payload is
[`artifacts/flagos_distributed_conformance_a800_20260825.json`](../../artifacts/flagos_distributed_conformance_a800_20260825.json).
Its SHA-256 digest is
`fbb49fb45ec1b22a39f9c58d676917212f4753c381b4319ca4833dbcfcc70ea4`.

## Result

The public `backend="flagos"` boundary initialized two ranks with one logical
device per rank. Python object collectives were forbidden throughout the run.

Eight of the ten required collective/dtype combinations passed with zero
maximum absolute error:

- `all_gather_into_tensor`, `all_reduce`, `broadcast`, and `isend`/`irecv` for
  `complex64`;
- the same four routes for `complex128`.

Both `reduce_scatter_tensor` checks failed before communication because the
active provider rejected `ComplexFloat` and `ComplexDouble`. The public errors
name the FlagCX process group, but an exception string is diagnostic evidence,
not provider-owned route identity. The report therefore correctly retains
`flagcx_route_verified=false` and `communication_claim_allowed=false`.

The true rank-sharded statevector completed after FlagQuantum stopped applying
the native-CUDA `block_current_stream()` optimization to an unverified FlagOS
work request and used the provider-neutral `wait()` contract instead. Each rank
retained four of eight amplitudes, executed three distributed gates, performed
two communications, and never materialized the distributed full state.

- `complex64`: passed, maximum absolute error `0.0`;
- `complex128`: failed, maximum absolute error
  `1.3427031042567705e-08` against a reference constructed natively in
  `complex128` (required tolerance `1e-11`).

## Decision

F1 remains **not accepted**. No capability-maturity, scalability, release, or
verified FlagCX claim may be promoted from this result. The measured blockers
inside FlagQuantum's scope are:

1. complex reduce-scatter is unavailable on this route;
2. the measured distributed complex128 workload does not preserve FP64-level
   accuracy;
3. provider-owned inner-backend and no-host-staging identity remain absent;
4. the measurement is single-node correctness evidence, not scaling evidence.

No Torch-FL or FlagCX source was changed, and no external pull request was
opened.

The subsequent FlagQuantum-only precision remediation is recorded in
[FlagOS distributed F1.1 result on A800](FLAGOS_DISTRIBUTED_F11_A800_RESULT.md).
