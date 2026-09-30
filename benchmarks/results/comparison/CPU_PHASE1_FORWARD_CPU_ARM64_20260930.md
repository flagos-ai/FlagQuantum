# Cross-framework simulator workload corpus (Apple arm64 CPU)

This report is generated from [`cpu_phase1_forward_cpu_arm64_20260930.json`](cpu_phase1_forward_cpu_arm64_20260930.json).
All workloads use exact complex128 statevectors on one CPU thread. Timings
include conversion, backend preparation, execution, and result retrieval.
Each median uses 9 retained samples after 1 warmup run, with 1 measured call per sample.
78 of 80 timing groups meet the declared 20% RMAD threshold; unstable measurements
remain visible in the raw artifact.
Environment: macOS-27.0-arm64-arm-64bit; Python 3.12.14; flagquantum 0.2.0, qiskit
2.5.2, qiskit-aer 0.17.2, cirq-core 1.7.0, pennylane 0.45.1, pennylane-lightning 0.45.0.

The latest refresh remeasured only FlagQuantum native. Existing Qiskit Aer, Cirq, and
PennyLane measurements were preserved unchanged; comparison ratios were recomputed from
the refreshed native medians.

| Workload | Qubits | Gates | Depth | FlagQuantum (s) | Qiskit Aer (s) | Cirq (s) | PennyLane (s) | Qiskit Aer / FQ | Cirq / FQ | PennyLane / FQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient | 10 | 82 | 26 | 0.000772 | 0.024817 | 0.003965 | 0.002063 | 32.16x | 5.14x | 2.67x |
| Hardware-efficient | 14 | 114 | 34 | 0.001432 | 0.025235 | 0.006532 | 0.003500 | 17.62x | 4.56x | 2.44x |
| Hardware-efficient | 18 | 146 | 42 | 0.011228 | 0.057187 | 0.056968 | 0.021387 | 5.09x | 5.07x | 1.90x |
| Hardware-efficient | 22 | 178 | 50 | 0.282610 | 0.718561 | 1.043863 | 0.476849 | 2.54x | 3.69x | 1.69x |
| Truncated QFT | 10 | 135 | 71 | 0.000462 | 0.024871 | 0.004211 | 0.003298 | 53.82x | 9.11x | 7.14x |
| Truncated QFT | 14 | 201 | 103 | 0.001118 | 0.030667 | 0.008379 | 0.007009 | 27.42x | 7.49x | 6.27x |
| Truncated QFT | 18 | 267 | 135 | 0.003898 | 0.075421 | 0.015606 | 0.050151 | 19.35x | 4.00x | 12.86x |
| Truncated QFT | 22 | 333 | 167 | 0.079349 | 0.940726 | 0.154574 | 1.040753 | 11.86x | 1.95x | 13.12x |
| Random Clifford | 10 | 60 | 8 | 0.000739 | 0.024117 | 0.002348 | 0.001865 | 32.62x | 3.18x | 2.52x |
| Random Clifford | 14 | 84 | 8 | 0.001816 | 0.028257 | 0.004249 | 0.002884 | 15.56x | 2.34x | 1.59x |
| Random Clifford | 18 | 108 | 8 | 0.008190 | 0.056273 | 0.015780 | 0.013661 | 6.87x | 1.93x | 1.67x |
| Random Clifford | 22 | 132 | 8 | 0.112162 | 0.654365 | 0.261208 | 0.298529 | 5.83x | 2.33x | 2.66x |
| Local brickwork | 10 | 98 | 12 | 0.000800 | 0.031712 | 0.005074 | 0.004219 | 39.65x | 6.34x | 5.27x |
| Local brickwork | 14 | 138 | 12 | 0.002030 | 0.029838 | 0.008650 | 0.004785 | 14.70x | 4.26x | 2.36x |
| Local brickwork | 18 | 178 | 12 | 0.017924 | 0.060526 | 0.061983 | 0.031604 | 3.38x | 3.46x | 1.76x |
| Local brickwork | 22 | 218 | 12 | 0.520445 | 0.717676 | 1.221049 | 0.717974 | 1.38x | 2.35x | 1.38x |
| Dense nonlocal | 10 | 65 | 19 | 0.000416 | 0.029552 | 0.003509 | 0.003722 | 71.12x | 8.44x | 8.96x |
| Dense nonlocal | 14 | 119 | 27 | 0.000866 | 0.029009 | 0.007501 | 0.004954 | 33.48x | 8.66x | 5.72x |
| Dense nonlocal | 18 | 189 | 35 | 0.008846 | 0.104685 | 0.043925 | 0.016151 | 11.83x | 4.97x | 1.83x |
| Dense nonlocal | 22 | 275 | 43 | 0.254773 | 1.951721 | 0.967858 | 0.413626 | 7.66x | 3.80x | 1.62x |

Ratios above one mean FlagQuantum was faster for that measured case.
These results are local comparison evidence, not a universal framework
ranking or release/scalability evidence.

## Reproduction

Refresh only FlagQuantum while preserving the checked-in external data:

```bash
flagquantum-benchmark run simulator_workload_corpus \
  --workloads hardware_efficient_statevector truncated_qft_statevector random_clifford_statevector local_brickwork_statevector dense_nonlocal_statevector \
  --n-wires 10 14 18 22 \
  --engines flagquantum_native --threads 1 \
  --warmup 1 \
  --iterations 9 \
  --calls-per-sample 1 \
  --refresh-from cpu_phase1_forward_cpu_arm64_20260930.json \
  --json-output cpu_phase1_forward_cpu_arm64_20260930.json --markdown-output REPORT.md
```
