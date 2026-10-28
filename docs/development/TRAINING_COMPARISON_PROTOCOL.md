# CPU training comparison protocol

Protocol: `cpu_training_20261026`, producer entry
`python -m benchmarks.training_comparison`. This extends the existing local
classifier and Heisenberg/HVA examples with fixed splits, independent gradient
checks, optimizer recovery, and isolated-process measurements. It does not
change a production API, execution mode or capability maturity.

## Run the workflow

Install the optional comparison dependencies into an isolated environment;
the resolved producing versions are recorded with each result. Then run:

```bash
python -m benchmarks.training_comparison --smoke --output /tmp/training-smoke
python -m benchmarks.training_comparison --output /tmp/training-cpu
python -m benchmarks.training_comparison_report /tmp/training-cpu
```

Output directories must be new. The first command runs eleven three-step
cases on reduced classifier splits at seed 11. The second runs all 33 cases:
three seeds, five quantum engines for two tasks, and a classical classifier
control. Both commands write a freeze record before case execution, a summary,
per-case logs/results, and checkpoints. They refuse an existing output path
rather than overwriting an earlier result. A failed or timed-out case remains
in the summary and is never scored as infinite speedup. Each case has a
240-second wall deadline and runs in a fresh process, with a heartbeat every
ten updates. CPU defaults require no credentials or accelerator.

The report command checks that completed cases share the frozen producer,
protocol and dataset before writing `REPORT.md`. Raw measurements remain
unchanged. The resolved optional environment is recorded in
`benchmarks/manifests/training_comparison_cpu_requirements.txt`; the source
commit, rather than a separately installed FlagQuantum wheel, identifies the
implementation under test.

## Frozen tasks

The complete definitions are in
`benchmarks/manifests/training_comparison_cpu_v1.json`. The manifest specifies
the fixed version-1 workloads; it is not a general runtime configuration API.
The code, manifest and dependency versions are bound in the result. A changed
task, target, seed, data split, optimizer, precision or stopping rule needs a
new protocol before inspecting its test results.

### Hybrid quantum classifier

There are 128 independently generated points in `[-1,1]^2`, using a fixed
data seed. Labels are the classical rule `x0*x1 > 0`, with no simulator used
as the label producer. The disjoint 64/32/32 train/validation/test split is
fixed before training. Encoding and gate order follow the existing two-qubit
classifier example. Four circuit parameters and a two-parameter classical
affine readout are jointly trained through a binary-cross-entropy objective.

A six-parameter quadratic logistic model uses the features
`1, x0, x1, x0^2, x0*x1, x1^2` as a parameter-count-matched classical control.
Its inductive bias is different and particularly well suited to the XOR
label rule. This small task checks trainability and workflow composition; it
does not establish a quantum advantage, a generalization advantage over
classical learning, or large-dataset performance.

Both models use Adam at learning rate 0.12 for exactly 60 updates. Validation
accuracy is measured after every update and included in elapsed training
time. The prospective target is 0.8 validation accuracy. Test accuracy is
evaluated once on the final model after the fixed update budget. Validation
does not select a checkpoint, and the test set does not change the training
or stopping rule. Every seed (11, 29, 47) is retained, including failures to
reach the target.

### Heisenberg optimization

Use the existing four-site, open-boundary XX+YY+ZZ Hamiltonian and the
depth-two `bond_resolved_phase` HVA with its dimer-singlet initialization.
The gate recipe is derived once from the canonical HVA builder's direct
parameter occurrences; each backend then constructs its native circuit from
that fixed recipe. There are 26 circuit parameters. No FlagQuantum execution
is inserted into a competitor's evaluation path.

Adam uses learning rate 0.03 for 60 updates from each frozen initial vector.
The target is 0.01 relative energy error to an independent dense Hamiltonian
eigensolution. All optimizers and seeds use the same initial vector and
objective definition. The zero-angle dimer energy is -6; the exact four-site
ground energy is `-3 - 2*sqrt(3)`. These analytic checks supplement cross-engine
comparisons. This stage compares Adam training; it does not complete the
separate QNG algorithm-versus-fusion ablation in issue #900.

## Methods and precision

| Engine | Backend and derivative | JIT |
| --- | --- | --- |
| FlagQuantum | complex128 local statevector, Torch reverse mode | Off |
| PennyLane backprop | default.qubit, Torch backprop | Off |
| PennyLane adjoint | lightning.qubit, adjoint through the Torch interface | Off |
| TensorCircuit-NG Torch | Circuit, PyTorch backend, Torch reverse mode | Off |
| TensorCircuit-NG JAX | Circuit, x64 JAX value-and-gradient, explicit CPU Torch bridge | On |

All parameters, outputs and Torch's default dtype are float64; amplitudes are
complex128. An output narrowed to float32 is refused, not reclassified as a
double-precision result. The initial adjoint preflight caught this distinction
when Torch's default dtype was float32; that failed exploratory run is retained
separately rather than presented as a completed case.

All quantum classifier engines evaluate examples serially. The JAX bridge
includes host-array conversion, compiled value-and-gradient execution and the
Torch callback in the measured work. Its derivative work occurs during value
evaluation. Lightning may also prepare derivatives during value evaluation.
Consequently, combined optimizer-step time is comparable within this task,
but forward/backward phase rankings across methods are not asserted.

This corpus does not measure batched/vmapped classifier throughput or every
framework's fastest available implementation. The classical control uses its
natural matrix-vector operation. Do not describe these results as a global
simulator ranking. GPU, noisy-circuit, MPS, distributed, finite-shot and hardware
behavior are outside the measured domain.

## Correctness, recovery and timing

Before timed updates, each quantum engine is checked against a separate NumPy
dense-matrix simulator. Value tolerance is 1e-9. Gradients are checked against
centered finite differences at step 1e-5 with absolute tolerance 2e-6. The
classical control uses an independent analytic logistic gradient. The oracle
is validation only and is never used as the training backend.

One warmup value/gradient call occurs without an optimizer update. The report
separates backend setup, first value/gradient, warmup, training, checkpoint
write and final assessment. The primary training timer includes the optimizer
updates and per-update validation. Independent oracle computation and the
subsequent checkpoint replay are excluded from that timer. Imports of common
runner dependencies occur before the measured phases; optional backend imports
may occur during setup. Therefore setup and `total_measured_seconds` are
diagnostics, not a fair cold-process startup ranking. Compare matched
`training_seconds` and training time to the fixed target, with the phase scope
stated explicitly. End-to-end application startup needs a separate experiment.

The checkpoint at update 30 is serialized and reloaded after the primary run.
Replaying the remaining updates must reproduce final parameters, gradients
and Adam slots within 1e-10. Replay is never used to pick a better model.
The report records peak process RSS, including imports, warmup, oracle checks
and replay. It is not tensor-only memory and is not exclusive accelerator
memory. All cases are single-process CPU comparisons with
`benchmark_evidence_class=comparison_non_release` and
`scalability_claim_allowed=false`.

## Verification and remaining scope

```bash
python -m pytest tests/benchmark_contract/test_training_comparison.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

The focused tests use a known dimer/ground-energy oracle, independent numerical
derivatives, full checkpoint/replay scenarios, disjoint data splits and a
deliberately wrong simulator. Optional backend cases skip only when their
package is unavailable, without substituting another engine.

Remaining extensions in #900 include QNG reverse/unfused-forward/fused-forward
ablation, independently varied parameter count, larger workloads, vectorized
training, repeated timing campaigns and optional accelerator measurements.
This bounded CPU stage does not close those requirements or promote results
to release evidence.

## JAX execution-strategy diagnostic

The original CPU corpus remains immutable. A separate frozen protocol keeps
the same tasks, initial vectors, splits, seeds, 60 updates and precision, and
compares FlagQuantum eager, `fq.Module` with its JAX backend, the existing
lower-level FlagQuantum JAX kernel, and TensorCircuit-NG JAX:

```bash
python -m benchmarks.training_comparison --jax-diagnostic --smoke --output /tmp/training-jax-smoke
python -m benchmarks.training_comparison --jax-diagnostic --output /tmp/training-jax-cpu
python -m benchmarks.training_comparison_report /tmp/training-jax-cpu
```

This produces eight smoke cases or 24 full cases and uses
`benchmarks/manifests/training_comparison_jax_cpu_v1.json`. It does not repeat
the classical control, whose original artifact remains available. JAX is JIT
compiled for all three JAX routes. Fallback is disabled for `fq.Module`.
All routes retain independent gradient and optimizer-recovery checks.

The lower-level kernel is a diagnostic of the quantum execution boundary;
it omits the Module's builder, IR, policy and result-processing work, and is
not a replacement public API. The baseline FlagQuantum JAX routes computed
input derivatives as well as parameter derivatives for input-bearing builders;
TensorCircuit's benchmark bridge computes parameter derivatives only because
the features are fixed. The optimized bridge skips input derivatives when
inputs do not require gradients and preserves them for trainable upstream
features. Backend identity, producing commit and phase boundaries must
accompany timing comparisons.

This diagnostic investigates the original eager-versus-JIT gap. It does not
retroactively replace the producing commit or measurements of the first
protocol, and does not establish that any route is globally optimal.

## Installed CPU diagnostic

An additional frozen protocol requires a loadable native CPU extension and
includes the public Module's default PyTorch policy alongside eager Circuit,
public JAX Module, internal JAX kernel and TensorCircuit-NG JAX. The workloads,
splits, precision and update rules remain fixed. It runs ten smoke cases or
30 full cases and uses
`benchmarks/manifests/training_comparison_installed_cpu_v1.json`.

```bash
python -m pip install -e . --no-deps --no-build-isolation
python -m benchmarks.training_comparison --installed-cpu-diagnostic --output /tmp/training-installed-cpu
python -m benchmarks.training_comparison_report /tmp/training-installed-cpu
```

The installation command assumes the declared build prerequisites are already
present. Result source identity records the installed distribution version,
native-adjoint availability and native-extension binary hash. The campaign
fails preflight if its extension is unavailable. An installed extension does
not imply every gate or derivative uses native code: the default Module route
still declares Torch reverse mode. This diagnostic does not substitute a
private adjoint kernel for the normal public call or change backend defaults.
Source-only and installed campaigns remain separate artifacts.

The current public JAX tensor path can defer IR materialization for pure static
binding graphs with small retained inputs. Checkpoint/diagnostic reads rebuild
the snapshot from detached copies of the consumed inputs, independently of
later in-place parameter changes. Stateful/lifted graphs, matrices, mutable
constants and large input buffers retain the existing extraction path. This
optimization does not change execution policy or backend defaults.

The configured thread limit refers to Torch intra/inter-op threads and the
OMP/OpenBLAS/MKL environment, not the total number of JAX/XLA runtime threads.
CPU campaign workers explicitly select the CPU JAX platform. The historical
campaigns also ran on a CPU-only JAX host.

The pre-existing refusal gap for non-finite compiled bindings is tracked in
[#925](https://github.com/flagos-ai/FlagQuantum/issues/925). It was reproduced
with frozen baseline Module and kernel sources as well as the candidate, and
is outside this performance change. The comparison's independent parity and
finite-output checks remain mandatory; this diagnostic does not certify all
invalid-input behavior.
