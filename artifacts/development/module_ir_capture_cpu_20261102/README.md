# Reusable Module capture: CPU comparison and CUDA conformance

This development packet supports #928's approved replacement boundary in #918.
It does **not** satisfy #918's performance closure criteria. Both unchanged
proposed budgets pass **0/50** before and after, in both retained rounds.
These are single-device results, with no scalability or universal ranking claim.

## Producers and reproduction

- Before: `abde02fd05100b3ae2a65d68128d459cfd437dbe` (merged snapshot optimization).
- Initial candidate: `053adcfbae359c8204c3617d9e1dbf8ac524e248`.
- Final candidate: `97b315ea02863583e88d13a4127be65133485cf6`.
- All measured worktrees were clean. CPU Torch 2.14.1, JAX/jaxlib 0.10.2,
  Python 3.12.14, Apple M5/macOS; exact complex128/float64.
- The same dependency environment and identical loadable native CPU extension
  were used in all checkouts. Binary SHA-256:
  `a7b5f80e48a1c387e6020d8b8f0e8d6951659ab4079a456963c3b5a32d8f551c`.
  This is a source-checkout comparison, not an installed-wheel comparison.
  Native availability does not mean Module uses native whole-circuit adjoint.

From the repository root, with optional JAX and the benchmark dependencies:

```bash
PYTHONPATH=. python artifacts/development/module_ir_capture_cpu_20261102/validation/evidence.py
python -m benchmarks.module_execution_matrix --output /tmp/module-matrix-new --repeats 5 --iterations 30 --timeout 120
python -m benchmarks.training_comparison --installed-cpu-diagnostic --output /tmp/module-training-new
PYTHONPATH=. python artifacts/development/module_ir_capture_cpu_20261102/validation/reproduce_builder.py --output /tmp/module-builder-new.json
```

For matched measurements, use a separate checkout of each full producer pin,
point `PYTHONPATH` at that checkout, set `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `JAX_PLATFORMS=cpu`, and run
serially without concurrent builds/tests. Matrix order was before/after,
after/before, before/after; each shape gets a fresh process, three warmups,
five timing blocks of 30 calls and a 120-second worker timeout. XLA's thread
count is not constrained. Each round retains 60 shape processes and 900
value/VJP comparisons. Context-first timings reuse earlier context caches;
they are not fresh-process cold-start timings. Raw samples, first-use times,
process dispersion and every shape remain in the capsules and aggregate.

The separate training protocol is unchanged: two tasks, five engines, three
seeds, 30 fresh cases per phase, serial examples and fixed-step Adam with
per-update validation and checkpoint recovery. Its phase order was after/before
in each round, not counterbalanced. Three seeds are not timing repetitions of
one fixed seed. Per-phase reports include cold-call costs, per-update and total
cost, task quality, time to target, peak process RSS and recovery. Imports,
oracle checks and replay are outside the training timer; process RSS includes
them. This small protocol does not establish fastest batching or GPU performance.

## Results and limitations

The initial candidate showed warm regressions in several contexts; that entire
round is retained. The final candidate removes redundant warmed signature work
while preserving current binding and finite-angle checks. In the final matrix,
median before/after ratios across shapes are:

| Derivative context | Default Module | Public JAX Module | Internal JAX control |
| --- | ---: | ---: | ---: |
| Parameters | 1.007 | 1.105 | 1.016 |
| Inputs | 0.985 | 1.020 | 0.997 |
| Both | 1.018 | 1.065 | 1.011 |
| no_grad | 1.006 | 1.087 | 1.026 |
| inference_mode | 1.003 | 1.088 | 1.032 |

Ratios greater than one favor the candidate. Individual shapes regress; the
retained control drift and three-process sample size preclude a broad causal
speedup claim. The proposed public overhead budget is public JAX <= 1.15 times
internal kernel + 5 microseconds; the default accessibility budget is default
<= 1.25 times best public route + 5 microseconds. Neither budget was relaxed.
Even the smallest final public/internal median ratio is about 1.82.

All 120 matched training cases across both rounds completed, with passing oracle
checks, zero checkpoint parameter/gradient/optimizer replay error and identical
paired final parameters. In the final round, training before/after median ratios
across six task/seed pairs are 1.023 for default Module, 1.084 for public JAX,
0.992 for native adjoint, 1.001 for the internal JAX kernel and 1.037 for
TensorCircuit JAX. Inspect the per-seed reports rather than treating this as a
universal ranking or attributing all wall-time change to the implementation.

The independent public-builder reproducer passes 4/8 cases before and 8/8 after.
The old JAX route cannot replay Torch `.square()`, `torch.sin` or `.detach()`;
the old default Torch capture loses detach's parameter gradient cut. The
replacement preserves values and nonuniform parameter/input VJPs across three
changing-input steps and captures the original builder once for each case.
Unsupported binding operations retain an explicit compatibility reason;
unimplemented JAX explicit matrices and captured trainable constants are refused.
Torch remains the default with higher derivatives; JAX remains explicit and
first-order. This packet does not implement automatic selection or #936.

## Hardware and publication

[GPU CI run 41443862759](https://github.com/flagos-ai/FlagQuantum/actions/runs/41443862759)
tested the exact final producer on NVIDIA A800. All four new Torch/JAX x batch
1/32 training/recovery cases passed; actual JAX bridge arrays report GPU, with
no backend fallback. The one-GPU suite passed 19 tests, and the existing required
two-GPU semantic gate also passed. Torch 2.13.0+cu130/CUDA 13.0 and JAX 0.10.2
are retained in the receipts. These GPU tests establish conformance, not timing,
memory or capacity benefits. The older queued run 41438841962 was cancelled
before executing GPU work when superseded by the final source.

Capsules contain sanitized original JSON and text logs, with deterministic tar
metadata and gzip timestamps. `publication.json` records raw/published member
hashes and capsule hashes. Raw checkpoint binaries remain outside Git; their
hashes and numerical replay receipts are retained, not binary checkpoint files.
The verifier checks producer files against Git objects and regenerates both
aggregates and all training reports/comparisons. `report-producer.json` pins
the reporting tools separately from the measured production source.
