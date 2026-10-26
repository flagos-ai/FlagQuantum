# CPU JAX training: public interface and bridge overhead

This development artifact investigates the execution-strategy gap in #914. It
retains 24 baseline cases and 24 optimized cases under exactly the same frozen
protocol, initial vectors and dataset. Both producers were clean. Published
JSON copies redact infrastructure identity using the repository sanitizer;
`publication.json` records the original and published file hashes. Numerical
measurements and producing identities are unchanged. Local originals and binary
checkpoints remain outside the source repository; logs are copied unchanged.

- Baseline producer: `3b18d75fc9e11464575249a65ee5708392b9866c`.
- Optimized producer: `cdbf4ecf513596ec8338468ed5ba1ca84c5b5cd8`.
- Protocol: `benchmarks/manifests/training_comparison_jax_cpu_v1.json`.
- [Baseline report](baseline/REPORT.md), [optimized report](optimized/REPORT.md).

## Results

Median elapsed seconds for 60 Adam updates, including per-update validation:

| Task | Route | Before | After |
| --- | --- | ---: | ---: |
| classifier | flagquantum | 2.5200 | 2.7593 |
| classifier | flagquantum_jax_module | 1.4463 | 0.6738 |
| classifier | flagquantum_jax_kernel | 0.6820 | 0.2807 |
| classifier | tensorcircuit_jax | 0.1332 | 0.1360 |
| heisenberg | flagquantum | 0.2969 | 0.3632 |
| heisenberg | flagquantum_jax_module | 0.0720 | 0.0563 |
| heisenberg | flagquantum_jax_kernel | 0.0339 | 0.0350 |
| heisenberg | tensorcircuit_jax | 0.0261 | 0.0370 |

The public JAX Module classifier time fell from 1.4463 to 0.6738 seconds
(2.15x in these campaigns); its lower-level kernel fell from 0.6820 to
0.2807 seconds. The public Module remains substantially slower than the
TensorCircuit-NG JAX classifier route. The original default eager route was
not made compiled or switched to another backend. These fixes do not make
FlagQuantum universally fastest.

Each phase is one sequential local campaign, with three model seeds rather
than three independent timing campaigns. Untouched controls also varied
between campaigns; short Heisenberg durations are especially sensitive. Do
not infer a reliable Heisenberg win from its small timing differences or a
global framework ranking. Classifier examples are evaluated serially in all
quantum routes. Larger, batched and installed-native-adjoint comparisons
remain necessary.

All 48 cases passed their initial independent oracle checks; maximum gradient
error was 2.81e-10. Every checkpoint replay reproduced final parameters,
gradients and Adam slots exactly. All quantum routes reached identical task
scores per seed. The classifier split remains only 32 held-out samples.

## Changes that benefit the existing interface

1. CPU float16/float32/float64 inputs enter the JAX JIT through their NumPy
   buffers instead of repeating DLPack device discovery for every small tensor.
   Accelerator and other dtype handling continues through DLPack.
2. JAX `fq.Module.forward` shares validated kernel execution with `execute`
   while avoiding unused result assembly and IR hash serialization. It retains
   precision checks, detached IR snapshots, topology counters and binding
   cleanup. Detailed `execute`, explicit fallback and debug behavior retain
   their existing paths.

The public call remains `module(inputs, parameters=theta)`, with the existing
JAX runtime policy. No new public API, implicit backend change, precision
downgrade or competitor-specific training shortcut is introduced. The
low-level kernel route is a diagnostic, not the recommended public interface.

## First-call and total-cost limits

Both phase reports retain per-seed first value/gradient and warmup times. Raw
JSON also retains backend setup, training, final assessment and diagnostic
phase totals. The optimized Module classifier first-call median was about
0.650 seconds, versus 0.117 seconds for TensorCircuit JAX. Compilation and
first-use cost are still material for short user workflows. Common imports
precede these timers and optional backend imports occur during setup; phase
totals cannot be interpreted as a fair cold-process startup ranking.

## Profiling and reproduction

`profiles_before/` retains cProfile data for 200 warmed classifier expectation
and parameter-gradient calls per route (five calls warmup, float64 Torch
defaults and one configured Torch thread). It is localization
evidence with profiler overhead, not a performance ranking. Each profile
records its source and package identity. The underlying loop was:

```python
fn = build_expectation(engine, "classifier")
theta = initial_parameters("classifier", 11)
features = dataset()[0].features
for i in range(200):
    value = fn(theta, features[i % 64])
    torch.autograd.grad(value, theta)
```

The baseline kernel profile attributed about 0.020 seconds of 0.038 seconds
to Torch-to-JAX conversion. The Module profile also showed repeated snapshot,
IR encoding and result work. The two fixes were measured together; this
artifact does not isolate their individual speedups.

From either producing commit and the pinned comparison environment:

```bash
python -m benchmarks.training_comparison --jax-diagnostic --output /tmp/training-jax-new
python -m benchmarks.training_comparison_report /tmp/training-jax-new
```

Verification: 118 focused runtime/API/JAX/scientific tests passed after the
implementation change, followed by two new optimizer-trajectory tests checking
forward/execute value and parameter/input-gradient agreement plus IR snapshot
hashes across updates, with debug on and off. The two new CPU bridge cases
are included in the 118 tests and check strided float32/float64 tensors and
changing parameter/input values against analytic VJPs. Architecture,
dependency, lint and changed-production-file type checks passed.

The full default suite remains unverified: the earlier 300-second run was
terminated and its source-only entry-point metadata failure was reproduced.
No green-default-suite claim is made. QNG ablation remains outside this
diagnostic and open in #900.

Remaining public-workflow/default-path performance work is tracked in
[#918](https://github.com/flagos-ai/FlagQuantum/issues/918).
