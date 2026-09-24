# Cross-framework simulator workload corpus (Apple arm64 CPU)

This report is generated from [`simulator_workload_corpus_cpu_arm64_20260924.json`](simulator_workload_corpus_cpu_arm64_20260924.json).
All workloads use exact complex128 statevectors on one CPU thread. Timings
include conversion, backend preparation, execution, and result retrieval.
Each median uses 9 retained samples after 1 warmup run, with 1 measured call per sample.
77 of 80 timing groups meet the declared 20% RMAD threshold; unstable measurements
remain visible in the raw artifact.
Environment: macOS-27.0-arm64-arm-64bit; Python 3.12.14; flagquantum 0.2.0, qiskit
2.5.2, qiskit-aer 0.17.2, cirq-core 1.7.0, pennylane 0.45.1, pennylane-lightning 0.45.0.

The latest refresh remeasured only FlagQuantum native. Existing Qiskit Aer, Cirq, and
PennyLane measurements were preserved unchanged; comparison ratios were recomputed from
the refreshed native medians.

| Workload | Qubits | Gates | Depth | FlagQuantum (s) | Qiskit Aer (s) | Cirq (s) | PennyLane (s) | Qiskit Aer / FQ | Cirq / FQ | PennyLane / FQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient | 10 | 82 | 26 | 0.000999 | 0.024817 | 0.003965 | 0.002063 | 24.83x | 3.97x | 2.06x |
| Hardware-efficient | 14 | 114 | 34 | 0.001760 | 0.025235 | 0.006532 | 0.003500 | 14.34x | 3.71x | 1.99x |
| Hardware-efficient | 18 | 146 | 42 | 0.016362 | 0.057187 | 0.056968 | 0.021387 | 3.50x | 3.48x | 1.31x |
| Hardware-efficient | 22 | 178 | 50 | 0.401835 | 0.718561 | 1.043863 | 0.476849 | 1.79x | 2.60x | 1.19x |
| Truncated QFT | 10 | 135 | 71 | 0.000714 | 0.024871 | 0.004211 | 0.003298 | 34.82x | 5.89x | 4.62x |
| Truncated QFT | 14 | 201 | 103 | 0.003037 | 0.030667 | 0.008379 | 0.007009 | 10.10x | 2.76x | 2.31x |
| Truncated QFT | 18 | 267 | 135 | 0.008044 | 0.075421 | 0.015606 | 0.050151 | 9.38x | 1.94x | 6.23x |
| Truncated QFT | 22 | 333 | 167 | 0.151352 | 0.940726 | 0.154574 | 1.040753 | 6.22x | 1.02x | 6.88x |
| Random Clifford | 10 | 60 | 8 | 0.000955 | 0.024117 | 0.002348 | 0.001865 | 25.24x | 2.46x | 1.95x |
| Random Clifford | 14 | 84 | 8 | 0.003643 | 0.028257 | 0.004249 | 0.002884 | 7.76x | 1.17x | 0.79x |
| Random Clifford | 18 | 108 | 8 | 0.020029 | 0.056273 | 0.015780 | 0.013661 | 2.81x | 0.79x | 0.68x |
| Random Clifford | 22 | 132 | 8 | 0.314945 | 0.654365 | 0.261208 | 0.298529 | 2.08x | 0.83x | 0.95x |
| Local brickwork | 10 | 98 | 12 | 0.001087 | 0.031712 | 0.005074 | 0.004219 | 29.16x | 4.67x | 3.88x |
| Local brickwork | 14 | 138 | 12 | 0.002814 | 0.029838 | 0.008650 | 0.004785 | 10.61x | 3.07x | 1.70x |
| Local brickwork | 18 | 178 | 12 | 0.032028 | 0.060526 | 0.061983 | 0.031604 | 1.89x | 1.94x | 0.99x |
| Local brickwork | 22 | 218 | 12 | 0.672586 | 0.717676 | 1.221049 | 0.717974 | 1.07x | 1.82x | 1.07x |
| Dense nonlocal | 10 | 65 | 19 | 0.001115 | 0.029552 | 0.003509 | 0.003722 | 26.50x | 3.15x | 3.34x |
| Dense nonlocal | 14 | 119 | 27 | 0.004203 | 0.029009 | 0.007501 | 0.004954 | 6.90x | 1.78x | 1.18x |
| Dense nonlocal | 18 | 189 | 35 | 0.022016 | 0.104685 | 0.043925 | 0.016151 | 4.75x | 2.00x | 0.73x |
| Dense nonlocal | 22 | 275 | 43 | 0.374537 | 1.951721 | 0.967858 | 0.413626 | 5.21x | 2.58x | 1.10x |

Ratios above one mean FlagQuantum was faster for that measured case.
These results are local comparison evidence, not a universal framework
ranking or release/scalability evidence.

## Reproduction

Refresh only FlagQuantum while preserving the checked-in external data:

```bash
flagquantum-benchmark run simulator_workload_corpus \
  --workloads random_clifford_statevector dense_nonlocal_statevector \
  --n-wires 18 22 \
  --engines flagquantum_native --threads 1 \
  --warmup 1 \
  --iterations 9 \
  --calls-per-sample 1 \
  --refresh-from simulator_workload_corpus_cpu_arm64_20260924.json \
  --json-output simulator_workload_corpus_cpu_arm64_20260924.json --markdown-output REPORT.md
```
