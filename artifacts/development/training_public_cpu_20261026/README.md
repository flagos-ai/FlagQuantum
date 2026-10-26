# Public CPU training: installed baseline and optimized paths

This is a matched, non-release development comparison supporting #918 and
the CPU training portion of #900. It is not a universal performance claim.
Both phases use the same protocol, dataset, initial vectors, seeds, precision,
native extension binary and installed distribution version. Each phase has
30 completed fresh-process cases (five routes, two tasks, three seeds).

- Baseline producer: `c8a4eb038dbd2c9624a5af3a55d62f197bd07a51`.
- Optimized producer: `c06cabd1ef3edb4453433e31f93e67de4af65945`.
- Both source trees were clean during every case.
- Distribution version: 0.4.0, editable source installation.
- Native CPU adjoint availability: true. This does not mean each route uses native adjoint; default Module still uses Torch reverse mode.
- Native extension SHA-256: `a7b5f80e48a1c387e6020d8b8f0e8d6951659ab4079a456963c3b5a32d8f551c`.
- [Baseline report](baseline/REPORT.md), [optimized report](optimized/REPORT.md).

## Measured training cost

Median seconds for 60 optimizer updates including validation after each update:

| Task | Route | Installed baseline | Optimized |
| --- | --- | ---: | ---: |
| classifier | flagquantum | 2.4391 | 2.6349 |
| classifier | flagquantum_torch_module | 2.4319 | 2.6675 |
| classifier | flagquantum_jax_module | 0.6325 | 0.1989 |
| classifier | flagquantum_jax_kernel | 0.2693 | 0.1528 |
| classifier | tensorcircuit_jax | 0.1300 | 0.1524 |
| heisenberg | flagquantum | 0.2894 | 0.3520 |
| heisenberg | flagquantum_torch_module | 0.3091 | 0.3622 |
| heisenberg | flagquantum_jax_module | 0.0479 | 0.0324 |
| heisenberg | flagquantum_jax_kernel | 0.0256 | 0.0247 |
| heisenberg | tensorcircuit_jax | 0.0245 | 0.0251 |

The public JAX Module classifier route improved from 0.6325 to 0.1989 seconds
(3.18x between these campaigns), but remained slower than the same-campaign
TensorCircuit-NG JAX route at 0.1524 seconds. Public Module Heisenberg cost
was 0.0324 versus 0.0251 seconds. The internal kernel was close to TensorCircuit
JAX on both small workloads; it is a diagnostic, not the public API recommendation.

The default PyTorch route remains substantially slower in this corpus. These
changes do not select JAX automatically, do not make every public path optimal
and do not establish each framework's best batched throughput. Untouched
controls varied between campaigns; sub-millisecond differences are not evidence
of a reliable framework win. These are single local campaigns, not independent
repeated timing trials.

All 60 cases passed independent initial value/gradient checks and reproduced
final parameters, gradients and Adam slots exactly after checkpoint reload.
Maximum initial finite-difference gradient error was 2.81e-10. Each route
reached the same final task score per seed. The maximum cross-phase final
parameter difference was 0. The held-out classifier split remains only 32 samples.

## Production changes and limits

The existing public call is unchanged: reuse `fq.Module` with the existing
JAX policy inside the normal Torch optimizer loop. Static, pure small-input
binding graphs defer IR materialization to checkpoint/diagnostic reads using
detached consumed-input copies. Stateful/lifted programs, matrices, mutable
constants and large retained inputs use the previous extraction path. Snapshot
tests cover in-place updates, checkpoint hashes, policy/precision changes,
no-grad and inference-mode validation, and avoiding large unused-input copies.

The autograd bridge class is reused rather than defined per call. Input
derivatives are computed only for trainable input tensors; parameter gradients
remain available, and switching input-gradient requirements is tested against
analytic VJPs. CPU NumPy conversion is restricted to CPU-default placement;
DLPack remains the route when needed to preserve CPU placement under an
accelerator-default JAX configuration or for other devices/dtypes.

First-call compilation is still material and is retained in both reports.
Raw JSON retains setup, warmup, training, final assessment and diagnostic phase
totals. Asymmetric import boundaries mean these totals are not fair cold-process
startup rankings. Configured Torch intra/inter-op and BLAS limits are one;
JAX/XLA runtime thread pools are not claimed to contain only one OS thread.

## Verification and retained blockers

- 306 focused runtime/API/JAX/native-adjoint/scientific, public-evidence and
  dependency-lane tests passed. Black, Ruff, changed-file mypy, architecture,
  dependency policy, public API baseline and repository hygiene checks passed.
- The installed extension loaded successfully and the previously missing
  distribution entry point was discovered. The earlier source-only metadata
  failure is resolved by installation.
- The full local default suite exceeded its 600-second budget and was stopped
  with its process group. Six existing historical-artifact provenance tests
  failed because their assumed unavailable pins resolve in this local object
  database; the source artifacts and validators were not changed. A new test
  dependency-lane classification failure was corrected and the lane audit passed.
- Remote CI was cancelled in source preparation on the earlier pushed revision;
  current required checks still need to complete before merge.
- The pre-existing non-finite compiled binding refusal gap is tracked in #925.
  Frozen baseline Module and kernel sources reproduced the same NaN behavior;
  this optimization does not certify invalid-input handling.
- Default/backend selection, realistic batching, startup cost and repeated
  timing trials remain in #918. QNG algorithm/fusion ablation remains in #900.

## Reproduce

See `docs/development/TRAINING_COMPARISON_PROTOCOL.md` and
`docs/guides/PERFORMANCE_ENGINEERING.md`. With the pinned optional dependencies
and build prerequisites installed:

```bash
python -m pip install -e . --no-deps --no-build-isolation
python -m benchmarks.training_comparison --installed-cpu-diagnostic --output /tmp/public-training-new
python -m benchmarks.training_comparison_report /tmp/public-training-new
```

Published JSON removes infrastructure identifiers only. `publication.json`
records original and published file hashes; local originals remain unchanged.
Binary checkpoints stay outside the source repository. Fresh runs produce new
checkpoints and repeat the recovery checks.
