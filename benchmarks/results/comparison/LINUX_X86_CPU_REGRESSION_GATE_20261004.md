# Linux x86 CPU regression gate

This report establishes a recurring single-socket Linux x86 gate for the
optimized 22-qubit Random Clifford path. The raw measured baseline is
[`linux_x86_cpu_regression_baseline_20261004.json`](linux_x86_cpu_regression_baseline_20261004.json)
and the checked gate result is
[`linux_x86_cpu_regression_gate_20261004.json`](linux_x86_cpu_regression_gate_20261004.json).
The independent Lightning floor is recorded in
[`linux_x86_cpu_regression_gate_pennylane_20261004.json`](linux_x86_cpu_regression_gate_pennylane_20261004.json).

## Measured baseline

The baseline was recorded on an Intel Xeon Platinum 8358 host with the process
pinned to physical CPUs `0-31` of one socket. It uses exact complex128
statevectors, two warmups and seven retained end-to-end samples. Conversion,
backend preparation, execution and result retrieval are included.

| Engine | Median | RMAD | FlagQuantum advantage |
| --- | ---: | ---: | ---: |
| **FlagQuantum native** | **46.257 ms** | 2.67% | **1.00×** |
| Qiskit Aer 0.17.2 | 135.248 ms | 0.94% | **2.92× faster** |
| PennyLane Lightning 0.45.0 | 436.830 ms | 0.76% | **9.44× faster** |

All three statevectors passed at absolute tolerance `1e-10`. Aer and Lightning
maximum absolute errors versus FlagQuantum were `8.67e-18` and `9.11e-18`.

## Gate contract

The scheduled gate fails closed unless all of these conditions hold:

- CPU model, process affinity, platform, Python/PyTorch family, thread settings,
  measurement scope and workload hash match the checked baseline;
- exact-statevector correctness passes for every engine;
- FlagQuantum and the selected comparison engine each have at least seven
  retained samples and pass the 20% RMAD stability rule;
- FlagQuantum is no more than 20% slower than its internal baseline;
- FlagQuantum remains no slower than Qiskit Aer and PennyLane Lightning;
- the selected external engine versions match the baseline when version data is
  present.

An environment or methodology mismatch returns `incomparable` with exit status
2 instead of being treated as a performance pass or regression. A correctness,
stability, coverage or performance failure returns exit status 1.

The checked reports pass with an internal current/baseline ratio of `1.0`, a
FlagQuantum/Aer ratio of `0.342`, and a FlagQuantum/Lightning ratio of `0.106`.
The workflow evaluates both external floors against the same baseline.

## User path

The gate exercises the normal public CPU path; users do not select a benchmark
backend or special kernel:

```python
import random

import flagquantum as fq
import torch

n_qubits = 22
generator = random.Random(7319 + 10_007 * n_qubits)
circuit = fq.Circuit(n_qubits, dtype=torch.complex128)
for _ in range(4):
    for qubit in range(n_qubits):
        getattr(circuit, generator.choice(("h", "s", "x")))(qubit)
    qubits = list(range(n_qubits))
    generator.shuffle(qubits)
    for index in range(0, n_qubits - 1, 2):
        getattr(circuit, generator.choice(("cx", "cz")))(
            qubits[index], qubits[index + 1]
        )

state = circuit.state()
```

## Automation and reproduction

`.github/workflows/linux-x86-cpu-performance.yml` runs weekly and by manual
dispatch on the fixed `a800-172` self-hosted runner. It joins the existing
host-level benchmark concurrency group so repository GPU and CPU measurements
do not run against each other.

Reproduce the measurement from the repository root:

```bash
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 \
taskset -c 0-31 flagquantum-benchmark run simulator_workload_corpus \
  --workloads random_clifford_statevector --n-wires 22 \
  --engines flagquantum_native qiskit_aer pennylane_lightning_qubit \
  --threads 32 --warmup 2 --iterations 7 --calls-per-sample 1 \
  --json-output current.json

flagquantum-benchmark run cpu_performance_gate \
  benchmarks/results/comparison/linux_x86_cpu_regression_baseline_20261004.json \
  current.json --max-slowdown 1.20 --minimum-samples 7 \
  --comparison-engine qiskit_aer --max-native-over-comparison 1.0 \
  --json-output gate-qiskit-aer.json

flagquantum-benchmark run cpu_performance_gate \
  benchmarks/results/comparison/linux_x86_cpu_regression_baseline_20261004.json \
  current.json --max-slowdown 1.20 --minimum-samples 7 \
  --comparison-engine pennylane_lightning_qubit \
  --max-native-over-comparison 1.0 \
  --json-output gate-pennylane-lightning.json
```

## Boundaries and stop condition

This is local single-process, single-device comparison evidence, not distributed
scalability or a universal framework ranking. The maintained gate covers the
measured 22-qubit complex128 Random Clifford workload on the named CPU and
affinity. New workloads and hardware require their own measured baselines.

This CPU phase is ready to close when the workflow contract tests and repository
CI pass, the self-hosted workflow reproduces a passing result, and no maintained
CPU scorecard case remains slower than its same-semantics external comparison.
