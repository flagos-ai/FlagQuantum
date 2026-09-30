# CPU phase 1 scorecard — Apple arm64, 30 September 2026

## Headline: FlagQuantum wins all 20 cross-framework cases

> On this exact-statevector corpus and recorded Apple arm64 environment,
> FlagQuantum is faster than Qiskit Aer, Cirq Simulator, and PennyLane
> Lightning in every workload/width case.

| Compared with | FlagQuantum wins | Observed speedup range |
| --- | ---: | ---: |
| Qiskit Aer | **20 / 20** | **1.38x–71.12x** |
| Cirq Simulator | **20 / 20** | **1.93x–9.11x** |
| PennyLane Lightning | **20 / 20** | **1.38x–13.12x** |

FlagQuantum was freshly measured on 30 September. External measurements are
the unchanged 24 September samples from the same host, environment, workload
IR, and methodology; this is explicitly recorded in `refresh_history`. This is
a finite-corpus result, not a claim that FlagQuantum is faster for every
possible circuit.

## Decision

**Pass.** The final native CPU measurements pass the profile-aware regression
gate with a maximum allowed slowdown of 20%. The gate covers 20 exact
statevector cases and eight differentiable VQE/QAOA cases. All current samples
were stable, all cross-framework state checks passed, and the focused benchmark
contract tests are the independent correctness backstop.

This is a local Apple arm64 scorecard, not a universal simulator ranking or
release/scalability evidence. The threshold is 20%, rather than 15%, because
the maintained end-to-end corpus already defines 20% relative median absolute
deviation as its stability boundary. A tighter threshold incorrectly rejected
two stable dense-nonlocal cases in the calibration run.

## What is measured and why

- The forward corpus measures one complete user-facing exact-statevector call,
  including conversion, preparation, execution, and result retrieval.
- Hardware-efficient, truncated-QFT, Random Clifford, local-brickwork, and
  dense-nonlocal circuits cover materially different gate and connectivity
  structures. Widths 10/14 expose fixed overhead; 18/22 expose kernel cost.
- The differentiable corpus measures expectation forward, native adjoint
  backward, and combined value-and-gradient for VQE and QAOA.
- Stable IR content hashes make each current case identical to its baseline.

## Measured FlagQuantum medians

Times are milliseconds. Forward FlagQuantum cells are the median of nine fresh
samples after warmup; adjoint cells use seven fresh samples.

| Forward workload | 10q | 14q | 18q | 22q |
| --- | ---: | ---: | ---: | ---: |
| Hardware efficient | 0.772 | 1.432 | 11.228 | 282.610 |
| Truncated QFT | 0.462 | 1.118 | 3.898 | 79.349 |
| Random Clifford | 0.739 | 1.816 | 8.190 | 112.162 |
| Local brickwork | 0.800 | 2.030 | 17.924 | 520.445 |
| Dense nonlocal | 0.416 | 0.866 | 8.846 | 254.773 |

## Detailed 22-qubit cross-framework comparison

The table below makes the 22-qubit comparison explicit. FlagQuantum was
remeasured on 30 September. Qiskit Aer, Cirq Simulator, and PennyLane Lightning
are the unchanged 24 September measurements from the same host, environment,
workload IR, and methodology; ratios were recomputed against the fresh
FlagQuantum medians. A ratio above 1 means FlagQuantum was faster.

| 22q workload | FlagQuantum (ms) | Qiskit Aer (ms / ratio) | Cirq (ms / ratio) | PennyLane Lightning (ms / ratio) |
| --- | ---: | ---: | ---: | ---: |
| Hardware efficient | 282.610 | 718.561 / 2.54x | 1043.863 / 3.69x | 476.849 / 1.69x |
| Truncated QFT | 79.349 | 940.726 / 11.86x | 154.574 / 1.95x | 1040.753 / 13.12x |
| Random Clifford | 112.162 | 654.365 / 5.83x | 261.208 / 2.33x | 298.529 / 2.66x |
| Local brickwork | 520.445 | 717.676 / 1.38x | 1221.049 / 2.35x | 717.974 / 1.38x |
| Dense nonlocal | 254.773 | 1951.721 / 7.66x | 967.858 / 3.80x | 413.626 / 1.62x |

The complete 10/14/18/22-qubit matrix and every raw timing sample are retained
in the linked forward artifact below.

| Differentiable workload | Width | Forward (ms) | Backward (ms) | Value + gradient (ms) |
| --- | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 10 | 1.109 | 0.591 | 1.698 |
| Hardware-efficient VQE | 14 | 1.662 | 1.268 | 2.931 |
| Hardware-efficient VQE | 18 | 7.062 | 8.814 | 15.817 |
| Hardware-efficient VQE | 22 | 107.922 | 167.143 | 275.683 |
| QAOA path MaxCut | 10 | 1.035 | 0.284 | 1.334 |
| QAOA path MaxCut | 14 | 1.503 | 0.958 | 2.496 |
| QAOA path MaxCut | 18 | 10.373 | 11.058 | 21.562 |
| QAOA path MaxCut | 22 | 174.805 | 188.073 | 363.555 |

Against the checked-in FlagQuantum baseline, 17/20 forward cases are faster and
three are slower. The largest observed slowdown is hardware-efficient 22q at
1.076x; the largest forward improvement is dense-nonlocal 14q at 4.85x. All eight combined
adjoint measurements improved: current time is 0.249x–0.870x baseline time
(1.15x–4.01x faster).

The raw artifacts preserve every timing sample, correctness value, stability
statistic, environment field, and per-case ratio:

- [`cpu_phase1_forward_cpu_arm64_20260930.json`](cpu_phase1_forward_cpu_arm64_20260930.json)
- [`cpu_phase1_forward_gate_cpu_arm64_20260930.json`](cpu_phase1_forward_gate_cpu_arm64_20260930.json)
- [`cpu_phase1_adjoint_cpu_arm64_20260930.json`](cpu_phase1_adjoint_cpu_arm64_20260930.json)
- [`cpu_phase1_adjoint_gate_cpu_arm64_20260930.json`](cpu_phase1_adjoint_gate_cpu_arm64_20260930.json)

The forward artifact retains the external raw samples and state agreement while
recording its native-only refresh history. The adjoint historical baseline
retains the PennyLane Lightning gradient agreement; the current refresh times
FlagQuantum only and is backed by the project's gradient correctness tests.

## Reproduce

Use Python 3.12, PyTorch 2.13, complex128, and one CPU thread on Apple arm64:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

flagquantum-benchmark run simulator_workload_corpus \
  --workloads hardware_efficient_statevector truncated_qft_statevector \
    random_clifford_statevector local_brickwork_statevector \
    dense_nonlocal_statevector \
  --n-wires 10 14 18 22 --engines flagquantum_native \
  --threads 1 --warmup 1 --iterations 9 --calls-per-sample 1 \
  --refresh-from \
    benchmarks/results/comparison/simulator_workload_corpus_cpu_arm64_20260924.json \
  --json-output current-forward.json --markdown-output current-forward.md

flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 10 14 18 22 --layers 1 --engines flagquantum_adjoint \
  --threads 1 --warmup 2 --iterations 7 --calls-per-sample 1 \
  --skip-memory-probe --json-output current-adjoint.json

flagquantum-benchmark run cpu_performance_gate \
  benchmarks/results/comparison/simulator_workload_corpus_cpu_arm64_20260924.json \
  current-forward.json --json-output forward-gate.json

flagquantum-benchmark run cpu_performance_gate \
  benchmarks/results/comparison/adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json \
  current-adjoint.json --json-output adjoint-gate.json
```

## Stop conditions for CPU phase 1

CPU phase 1 is complete when all of the following remain true:

1. The deterministic workload/IR matrix has not shrunk.
2. State, expectation, and full parameter-gradient correctness tests pass for
   complex64 and complex128.
3. Native, rollback, cache-isolation, non-CPU routing, and memory-tier tests pass.
4. Each gated timing has at least five samples and passes the corpus stability
   rule on a matching platform/runtime/thread profile.
5. No comparable median is more than 20% slower than its accepted baseline.
6. Static checks introduce no new errors and no concrete correctness defect is
   open.

Different hardware or runtime profiles return `incomparable` (exit status 2),
not pass. After these conditions are met, further CPU work needs a new measured
bottleneck or user-visible capability; it should not continue as unbounded
micro-optimization.
