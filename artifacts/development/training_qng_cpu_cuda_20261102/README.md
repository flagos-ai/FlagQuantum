# Matched training and QNG evidence for issue #900

This development comparison extends the existing training report and HVA
benchmark entry point. It does not change a public API, execution policy,
mandatory dependency, or capability maturity.

## Inventory and retained workloads

The earlier `training_cpu_20261026`, `training_public_cpu_20261026` and
`training_jax_cpu_20261026` artifacts already define the small classifier,
classical control, independent parity and checkpoint recovery. They remain
immutable. This campaign reruns the original CPU task manifest with explicit
first-call-inclusive total time to target and process wall accounting.

The earlier `qng_hva_time_to_accuracy.py`, `qng_hva_scaling.py`,
`qng_phase_breakdown.py` and `qng_hva_forward_tangent_triton.py` compare Adam,
reverse Jacobians or fused tangents, but do not isolate unfused forward
propagation. The extended HVA entry point adds that missing control and
reuses the canonical HVA recipe and layer-block metric definition.

## Frozen CPU results

- [Matched training report](training-cpu/REPORT.md): 33 completed cases,
  five quantum routes for two tasks and one classical classifier control,
  each at seeds 11, 29 and 47. All independent gradient checks and complete
  Adam checkpoint/replay checks passed. Each case records 60 updates.
- [QNG CPU report](qng-cpu/REPORT.md): 27 completed cases and nine explicit
  unsupported Triton-on-CPU cases. Three geometry routes, three seeds and
  widths/depths (4,1), (4,2), (6,1) retain all 30 optimizer updates.
- The QNG four-site, depth-two workload reached 0.01 relative energy error
  for every seed and applicable method. The depth-one four- and six-site
  workloads did not reach the target within 30 steps. Their target times
  remain null, rather than being discarded or scored as infinite speedup.

The classification task freezes a disjoint 64/32/32 split, seed, label rule,
six trainable parameters and a six-parameter classical control. The physics
objective freezes the open XX+YY+ZZ Hamiltonian, singlet initialization,
parameterization, optimizer, damping, target and stop rules. QNG uses 13 and
26 parameters at the same width, and 21 parameters at width six.

CPU raw evidence was produced by committed source
`d76c095afba994972d4a0dfce9a924a2f7143673`, with `dirty=false` in every freeze
and case. Frozen manifests, producer hashes, installed package versions,
platform, Python and host identity accompany the raw results. The later
`770e24b4f43290f0e1a7d9ca212a8a4e4a396da5` commit adds GPU device/cache
provenance, bounded compiler-process cleanup and report inventory checks;
it does not change the CPU numerical algorithm, target or optimizer.
The CPU QNG Markdown was regenerated with the complete-trajectory renderer
`bd8819e22e48c9826341ae64eaf514aa10cb5ff1`; raw JSON is unchanged.

## Numerical checks and timing interpretation

An independently assembled NumPy dense circuit and centered finite differences
check state tangents, the projected Fubini-Study metric, energy gradients and
damped update directions. Analytic dimer and ground energies provide additional
checks. The PennyLane control uses its actual `adjoint_metric_tensor` transform
on `default.qubit` with a Torch interface. Its objective gradient is held equal
to the other geometry routes, so the ablation isolates metric construction.
Complete Lightning and TensorCircuit-NG training are compared in the separate
matched training report, not inferred from the metric control.

Reverse and unfused geometry share the same eager Pauli rotation primitive,
static layout and batched tangent lanes. Their paired metric-time ratio
measures derivative-algorithm differences in this adapter. Unfused/Triton
metric time measures the fusion implementation difference. Ratios below one
are losses. No CPU/CUDA cross-precision speedup is computed.

Measured total includes setup, first call, warmup, initial assessment and
training; the Adam report also includes final assessment. Training includes
validation and checkpoint writing. Independent oracles and checkpoint replay
are outside that timer. Fresh-process wall includes imports, interpreter
startup, oracle checks and replay. Each worker has a 240-second deadline.
RSS includes the entire worker; the CUDA allocated peak covers training.
These are separate memory scopes, not an estimate of exclusive system usage.
All step energies, parameters, QNG directions and phase times remain in JSON.
No quantum advantage, best-backend ranking or scalability is established.

Binary Adam checkpoints are deliberately excluded from this text evidence
corpus. Each result retains its checkpoint SHA-256 and reload errors; rerunning
writes fresh checkpoints and verifies the full remaining optimizer trajectory.

## Reproduce

Use the optional environment in
`benchmarks/manifests/training_comparison_cpu_requirements.txt`; the actual
producing versions are in each source record. From the producing checkout:

```bash
python -m benchmarks.training_comparison --output /tmp/training-900
python -m benchmarks.training_comparison_report /tmp/training-900
python -m benchmarks.qng_hva_time_to_accuracy --ablation --output /tmp/qng-900-cpu
python -m benchmarks.qng_hva_time_to_accuracy --ablation --device cuda --output /tmp/qng-900-cuda
python -m benchmarks.training_comparison_report /tmp/qng-900-cpu
python -m benchmarks.training_comparison_report /tmp/qng-900-cuda
python -m pytest tests/benchmark_contract/test_training_comparison.py tests/benchmark_contract/test_qng_hva_ablation.py -q
```

Output directories must be new and should be outside the source checkout.
`--smoke` uses three updates and seed 11 for workflow checks only; it is not
the full acceptance experiment. Report regeneration validates the frozen
producer/protocol and, for QNG, requires the complete workload/method/seed
inventory including unsupported and failed routes.

The existing Local GPU Gate's `qng_ablation=true` dispatch runs the frozen
CUDA protocol on its A800 lane with accelerator concurrency and a watchdog.
Every case has a fresh private Triton compilation cache. First compile/load
is included in total cost. Raw results include actual device name, CUDA
runtime, compute capability and Triton version, rather than inferring them
from the runner label. The workflow uploads results and logs even on failure.

## Retained validation

The local focused selection passed 99 tests, with the real CUDA-only check
explicitly skipped on this CPU host. The subsequently added rejection test
for a truncated completed trajectory passed separately, together with the matched-training missing-seed rejection test, bringing
the focused coverage to 101 passed and one GPU-only skip. Black, Ruff, mypy, architecture,
dependency policy and ownership checks passed. Logs are under `validation/`.
This is a focused verification statement, not a claim that the entire default
suite ran locally.

`validation/verify_sources.py` compares every recorded producer and manifest
hash with the bytes at its recorded Git commit, checks every retained case,
and verifies that completed runs contain the full fixed-step trace. It also
checks all Adam reload errors and the CUDA device/cache/memory records. Run:

```bash
python artifacts/development/training_qng_cpu_cuda_20261102/validation/verify_sources.py
```

The audit output is retained in
[validation/source-and-case-audit.json](validation/source-and-case-audit.json).

## CUDA results and acceptance decision

The [CUDA report](qng-cuda/REPORT.md) contains 27 completed QNG cases and nine
explicit unsupported `default.qubit`-on-CUDA cases. The framework metric is
available and independently checked in the CPU campaign. The real fused-kernel
parity test passed on an **NVIDIA A800-SXM4-80GB**, CUDA runtime **13.0**,
Triton **3.7.1**. The producing commit is
`770e24b4f43290f0e1a7d9ca212a8a4e4a396da5`, with `dirty=false` throughout.
The [successful GPU run](https://github.com/flagos-ai/FlagQuantum/actions/runs/41394198690)
and its complete job log and metadata are retained under `validation/`.

CUDA paired reverse/unfused metric-time ratios range from 10.85 to 78.49;
unfused/fused metric-time ratios range from 3.73 to 4.80. These are small-state
adapter measurements, not a general complexity or simulator ranking. Both
algorithm and fusion measurements include the same metric projection. Complete
updates still include the identical objective gradient and solve, so these
phase ratios must not be presented as end-to-end training speedups.

Cold compilation changes the practical outcome. For (4 wires, depth 2), median
first-call cost is 5.80 seconds for fused tangents versus 0.11 seconds unfused;
median measured total for 30 steps is 7.59 versus 1.93 seconds. Thus fusion
loses on total cost for this short campaign even though its warm metric is
faster. The report retains that loss. As on CPU, depth-one workloads do not
reach the target; every depth-two seed reaches it. No non-convergent case is
assigned a time-to-target speedup.

| Issue #900 acceptance criterion | Retained evidence | Decision |
| --- | --- | --- |
| CPU-default hybrid and physics tasks with frozen input, split, seed and workload; checkpoint reload and classical control | Original task manifest and `training-cpu/freeze.json`; 33 cases with independent gradients, full Adam reload checks and classical control | Met |
| Applicable FlagQuantum, Lightning and TensorCircuit-NG routes with actual execution configuration | Five quantum engines per task; per-case backend, interface, dtype, derivative, thread and JIT metadata; separate CUDA device/runtime/cache records | Met |
| Fixed-budget quality, time-to-target, per-step/total time, memory, all seeds, retained failures and first-call cost | Both report families and complete raw results; measured total includes first call; process wall and oracle scopes stated; null target times and unsupported statuses retained | Met |
| Reverse, unfused-forward, fused-forward and framework metric; independent tangent/metric/update checks; independent parameter-count variation | CPU and CUDA QNG reports, NumPy oracle checks, actual fused-kernel CI test; 13/26 parameters at width four and 21 at width six | Met |
| Separate algorithm/fusion savings and complete optimizer trajectories without advantage claims | Paired metric ratios, full step tables and parameter/direction JSON; compile-inclusive loss explicitly reported | Met |
| Machine-readable, reproducible report with source/environment provenance | 105 retained cases: 87 completed, 18 explicitly unsupported; exact committed producer/manifest hash audit, runtime versions, commands, logs and workflow record | Met |

All six acceptance criteria are supported by this bounded evidence. Issue #900
can close when the fix and this evidence are merged after repository checks.
Larger-task performance, vectorized throughput, repeated host timing campaigns,
other accelerators and quantum advantage remain outside these claims.
