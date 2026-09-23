# Cross-framework CPU simulator comparison

This file is generated deterministically from measured JSON artifacts. It is local comparison evidence, not a universal ranking, scalability claim, or release gate.

## Scope

- Device: `cpu` on `arm64`
- Precision and workload: exact statevector rows listed below
- Worker threads: `1`
- Warmups / samples / calls per sample: `3` / `9` / `10`
- Ratios are external-engine median divided by FlagQuantum median; values above `1.00x` mean FlagQuantum was faster in that row.

## Results

| Qubits | Gates | FlagQuantum median | Qiskit Aer median | Qiskit Aer / FQ | Cirq Simulator median | Cirq Simulator / FQ | PennyLane Lightning median | PennyLane Lightning / FQ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 82 | 0.577 ms | 0.668 ms | 1.16x | 6.519 ms | 11.29x | 0.554 ms | 0.96x |
| 14 | 114 | 1.307 ms | 2.481 ms | 1.90x | 17.391 ms | 13.31x | 1.488 ms | 1.14x |
| 18 | 146 | 13.125 ms | 26.872 ms | 2.05x | 69.178 ms | 5.27x | 16.484 ms | 1.26x |
| 22 | 178 | 300.297 ms | 534.415 ms | 1.78x | 859.003 ms | 2.86x | 341.139 ms | 1.14x |
| 24 | 194 | 1164.636 ms | 2280.374 ms | 1.96x | 3923.549 ms | 3.37x | 1610.489 ms | 1.38x |

## Correctness and stability

| Qubits | Engine | Maximum error | R-MAD | Verdict |
| ---: | --- | ---: | ---: | --- |
| 10 | FlagQuantum | 6.280e-16 | 0.020 | correct, stable |
| 10 | Qiskit Aer | 6.280e-16 | 0.024 | correct, stable |
| 10 | Cirq Simulator | 1.618e-16 | 0.296 | correct, noisy |
| 10 | PennyLane Lightning | 1.328e-16 | 0.036 | correct, stable |
| 14 | FlagQuantum | 7.065e-16 | 0.048 | correct, stable |
| 14 | Qiskit Aer | 7.065e-16 | 0.020 | correct, stable |
| 14 | Cirq Simulator | 7.238e-16 | 0.106 | correct, noisy |
| 14 | PennyLane Lightning | 5.715e-16 | 0.005 | correct, stable |
| 18 | FlagQuantum | 4.743e-16 | 0.006 | correct, stable |
| 18 | Qiskit Aer | 4.743e-16 | 0.002 | correct, stable |
| 18 | Cirq Simulator | 8.951e-16 | 0.244 | correct, noisy |
| 18 | PennyLane Lightning | 4.003e-16 | 0.020 | correct, stable |
| 22 | FlagQuantum | 1.683e-16 | 0.034 | correct, stable |
| 22 | Qiskit Aer | 1.683e-16 | 0.027 | correct, stable |
| 22 | Cirq Simulator | 1.714e-13 | 0.022 | correct, stable |
| 22 | PennyLane Lightning | 7.109e-16 | 0.026 | correct, stable |
| 24 | FlagQuantum | 4.905e-16 | 0.033 | correct, stable |
| 24 | Qiskit Aer | 4.905e-16 | 0.021 | correct, stable |
| 24 | Cirq Simulator | 1.456e-12 | 0.052 | correct, stable |
| 24 | PennyLane Lightning | 1.555e-15 | 0.028 | correct, stable |

## Historical comparison

No historical baseline was supplied. This report makes no performance regression verdict.

## Sources and reproduction

- `benchmarks/results/comparison/flagquantum_qiskit_aer_cpu_arm64_20260923.json` (`sha256:fe0b50d8303209e1a383641de6986f8001079a1f87e42bfdb4d9c60306735f91`)
- `benchmarks/results/comparison/cirq_cpu_arm64_20260923.json` (`sha256:0bb8bf4a56fea2cf731f56a6362c78e7760f8d4a337ad54ee9084a67894dda6f`)
- `benchmarks/results/comparison/pennylane_lightning_cpu_arm64_20260923.json` (`sha256:26c3fb1861a1f5400f79e9d4b37b301960a4c58d088ccb7b17751bb09d463087`)

Regenerate the report from the repository root without rerunning any simulator:

```bash
flagquantum-benchmark run simulator_comparison_report \
  benchmarks/results/comparison/flagquantum_qiskit_aer_cpu_arm64_20260923.json \
  benchmarks/results/comparison/cirq_cpu_arm64_20260923.json \
  benchmarks/results/comparison/pennylane_lightning_cpu_arm64_20260923.json \
  --json-output comparison.json \
  --markdown-output comparison.md
```
