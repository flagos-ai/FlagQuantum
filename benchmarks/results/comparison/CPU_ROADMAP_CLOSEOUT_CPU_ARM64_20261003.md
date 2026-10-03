# CPU simulator roadmap closeout — Apple arm64

## Decision

**Complete for the defined local CPU roadmap.** The accepted evidence covers
single-state forward execution, exact adjoint value-and-gradient, native CPU
threading, memory-bounded adjoint execution, parameter-batched forward
execution, and regression governance. The forward and adjoint gates pass, no
measured batch workload is slower than the matched PennyLane Lightning native
broadcast, and the last three named batch-memory targets have met their stop
conditions.

This is a bounded engineering decision, not a claim that CPU simulation is
universally optimal. New CPU work should begin only when a reproducible user
workload exposes a distinct correctness, latency, throughput, memory, or
capacity gap outside the accepted matrix below.

## What was completed

| Track | Accepted evidence | Result | Status |
| --- | --- | --- | --- |
| Single-state exact forward | 5 workloads at 10/14/18/22 qubits, complex128, one thread | FlagQuantum won all 20 cases against Qiskit Aer, Cirq, and PennyLane Lightning; observed ratios span **1.38x–71.12x** | Complete |
| Exact adjoint differentiation | Hardware-efficient VQE and QAOA at 10/14/18/22 qubits | All eight current value-and-full-gradient cases pass the profile-aware regression gate; the method-matched corpus agrees within `1e-9` | Complete |
| Native CPU threading | 22-qubit VQE and QAOA, 1/2/4/8 threads | Four threads give **2.30x** and **2.41x** total speedups; eight threads give **2.57x** and **2.84x** on this host | Complete with documented sublinear scaling |
| Adjoint memory tiers | 24-qubit, complex128 hardware-efficient VQE | Automatic zero-auxiliary, compact-cycle, and fused-gather tiers complete in **846.828/665.454/648.038 ms** with explicit modeled budgets | Complete |
| Parameter-batched forward | 5 workloads, 18 qubits, batch 32, complex128, one thread | FlagQuantum takes **321.046–960.035 ms** and is **1.323x–2.571x** faster than PennyLane Lightning native broadcast | Complete |
| Batch working set | Hardware Efficient, Dense Nonlocal, and Truncated QFT | Execution RSS falls **41.5%**, **23.8%**, and **44.9%** against the fixed-64-MiB rollback while preserving bounded or improved time | Complete |
| Regression governance | 20 forward and 8 adjoint cases | Both checked-in profile-aware gates report `pass`; incompatible hardware/runtime profiles report `incomparable` rather than a false pass | Complete |

The source artifacts retain the absolute timings, every sample, correctness
errors, runtime versions, workload hashes, and measurement methodology:

- [`CPU_PHASE1_SCORECARD_CPU_ARM64_20260930.md`](CPU_PHASE1_SCORECARD_CPU_ARM64_20260930.md)
- [`cpu_phase1_forward_cpu_arm64_20260930.json`](cpu_phase1_forward_cpu_arm64_20260930.json)
- [`cpu_phase1_forward_gate_cpu_arm64_20260930.json`](cpu_phase1_forward_gate_cpu_arm64_20260930.json)
- [`cpu_phase1_adjoint_cpu_arm64_20260930.json`](cpu_phase1_adjoint_cpu_arm64_20260930.json)
- [`cpu_phase1_adjoint_gate_cpu_arm64_20260930.json`](cpu_phase1_adjoint_gate_cpu_arm64_20260930.json)
- [`NATIVE_CPU_ADJOINT_THREAD_SCALING_CPU_ARM64_20260928.md`](NATIVE_CPU_ADJOINT_THREAD_SCALING_CPU_ARM64_20260928.md)
- [`NATIVE_CPU_ADJOINT_MEMORY_TIERS_CPU_ARM64_20260930.md`](NATIVE_CPU_ADJOINT_MEMORY_TIERS_CPU_ARM64_20260930.md)
- [`native_cpu_adjoint_memory_tiers_cpu_arm64_20260930.json`](native_cpu_adjoint_memory_tiers_cpu_arm64_20260930.json)
- [`BATCHED_STATEVECTOR_CPU_PHASE_CLOSEOUT_CPU_ARM64_20261003_SCORECARD.md`](BATCHED_STATEVECTOR_CPU_PHASE_CLOSEOUT_CPU_ARM64_20261003_SCORECARD.md)
- [`batched_statevector_cpu_phase_closeout_cpu_arm64_20261003.json`](batched_statevector_cpu_phase_closeout_cpu_arm64_20261003.json)
- [`BATCHED_STATEVECTOR_ADAPTIVE_MEMORY_CPU_ARM64_20261003_SCORECARD.md`](BATCHED_STATEVECTOR_ADAPTIVE_MEMORY_CPU_ARM64_20261003_SCORECARD.md)
- [`batched_statevector_adaptive_memory_cpu_arm64_20261003.json`](batched_statevector_adaptive_memory_cpu_arm64_20261003.json)
- [`BATCHED_STATEVECTOR_QFT_ADAPTIVE_MEMORY_CPU_ARM64_20261003_SCORECARD.md`](BATCHED_STATEVECTOR_QFT_ADAPTIVE_MEMORY_CPU_ARM64_20261003_SCORECARD.md)
- [`batched_statevector_qft_adaptive_memory_cpu_arm64_20261003.json`](batched_statevector_qft_adaptive_memory_cpu_arm64_20261003.json)

## User path

The optimized paths require no backend-specific API:

```python
import torch
import flagquantum as fq
from flagquantum import algorithms as fqa

torch.set_num_threads(4)
angles = torch.linspace(-0.4, 0.4, 32, dtype=torch.float64)
circuit = fq.Circuit(18, bsz=32, dtype=torch.complex128)
for wire in range(18):
    circuit.ry(wire, angles + 0.01 * wire)
for left in range(0, 17, 2):
    circuit.cx(left, left + 1)

states = circuit.state()
assert states.shape == (32, 2**18)

theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
training_circuit = fq.Circuit(3, dtype=torch.complex128)
training_circuit.ry(0, theta).cx(0, 1)
hamiltonian = fqa.Hamiltonian((fqa.pauli_term(0.7, "ZZ", (0, 1)),))
energy = hamiltonian.expectation(training_circuit, differentiation="adjoint")
energy.backward()
```

## Reproduce the accepted gates

Use Python 3.12, PyTorch 2.13, complex128, and the recorded Apple-arm64 host
profile. External framework packages are required only for matched comparisons.

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
  --json-output current-forward.json

flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 10 14 18 22 --layers 1 --engines flagquantum_adjoint \
  --threads 1 --warmup 2 --iterations 7 --calls-per-sample 1 \
  --skip-memory-probe --json-output current-adjoint.json

flagquantum-benchmark run cpu_performance_gate \
  benchmarks/results/comparison/cpu_phase1_forward_cpu_arm64_20260930.json \
  current-forward.json --json-output forward-gate.json

flagquantum-benchmark run cpu_performance_gate \
  benchmarks/results/comparison/cpu_phase1_adjoint_cpu_arm64_20260930.json \
  current-adjoint.json --json-output adjoint-gate.json
```

The batch and memory reproduction commands remain beside their raw artifacts
in the linked scorecards. They require 11 retained calls and three fresh-process
RSS probes, so they are intentionally separate from routine CI.

## Why this roadmap stops

The roadmap stops because all of its declared checks are satisfied:

1. the forward and adjoint workload matrices have not shrunk;
2. state, expectation, and full-gradient correctness have independent tests;
3. rollback, cache isolation, non-CPU routing, and memory tiers are covered;
4. timing series meet their minimum sample and stability rules;
5. the forward and adjoint performance gates have no accepted-profile failure;
6. the batch corpus leads its matched external comparison in every case;
7. the final named batch-memory targets met their measured thresholds; and
8. rejected optimizations were removed when end-to-end A/B results did not win.

Future CPU work must identify its own workload, reference implementation,
correctness tolerance, measurement profile, rollback, success threshold, and
stop condition before implementation. This prevents an already accepted finite
corpus from turning into unbounded micro-optimization.

## Boundaries

- All headline evidence is local Apple-arm64 data. It is not a universal framework
  ranking, release claim, or cross-platform scalability result.
- The single-state corpus reaches 22 qubits; the 24-qubit result is one bounded
  adjoint-memory workload. Neither proves arbitrary 22- or 26-qubit capacity.
- The batch result is an 18-qubit throughput and working-set test whose logical
  output alone is 128 MiB.
- Thread scaling is sublinear because Python/planning work, serial regions,
  memory bandwidth, synchronization, and a finite task count remain. The table
  reports measured speedup rather than promising thread-count-proportional gain.
- GPU, distributed statevector, MPS, tensor-network, noise, and shot-based
  simulation are owned by separate evidence tracks.
