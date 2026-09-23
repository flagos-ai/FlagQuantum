# Cirq and PennyLane Lightning CPU comparison

This measured comparison extends the existing FlagQuantum and Qiskit Aer result
with Cirq Simulator 1.7.0 and PennyLane 0.45.1 using Lightning 0.45.0. All four
columns use the same deterministic FlagQuantum IR, exact complex128 statevector
semantics, Apple arm64 host, and one worker thread. The FlagQuantum and Qiskit
Aer values below are reused unchanged from the adjacent measured artifact; the
new runners measured only their selected external engine and executed
FlagQuantum once per case as an untimed correctness reference.

Each new steady-state median contains nine samples of ten executions after
three warmups. Conversion, cold execution, steady-state execution, every raw
sample, and correctness error are retained in the adjacent JSON files.

| Qubits | Gates | FlagQuantum | Qiskit Aer | Aer / FQ | Cirq Simulator | Cirq / FQ | PennyLane Lightning | Lightning / FQ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 82 | 0.577 ms | 0.668 ms | 1.16x | 6.519 ms | 11.29x | 0.554 ms | 0.96x |
| 14 | 114 | 1.307 ms | 2.481 ms | 1.90x | 17.391 ms | 13.31x | 1.488 ms | 1.14x |
| 18 | 146 | 13.125 ms | 26.872 ms | 2.05x | 69.178 ms | 5.27x | 16.484 ms | 1.26x |
| 22 | 178 | 300.297 ms | 534.415 ms | 1.78x | 859.003 ms | 2.86x | 341.139 ms | 1.14x |
| 24 | 194 | 1164.636 ms | 2280.374 ms | 1.96x | 3923.549 ms | 3.37x | 1610.489 ms | 1.38x |

A ratio above one means FlagQuantum was faster in that row. FlagQuantum was
1.16x to 2.05x faster than Qiskit Aer. PennyLane Lightning was 1.04x faster at
10 qubits; FlagQuantum was 1.14x to 1.38x faster from 14 to 24 qubits.
FlagQuantum was 2.86x to 13.31x faster than Cirq on this workload.
These observations are specific to this host, workload, dtype, and thread count;
they are comparison evidence, not universal rankings or scalability evidence.

All ten new correctness checks passed the absolute tolerance of `1e-10`.
PennyLane's largest maximum absolute error was `1.55e-15`; Cirq's was `1.46e-12`
at 24 qubits. Every PennyLane steady-state row passed the declared relative
median absolute deviation threshold. Cirq passed it at 22 and 24 qubits, while
10, 14, and 18 qubits were marked unstable rather than rerun or omitted. This
does not change their correctness verdict, but those small-width Cirq medians
should be treated as noisy local observations.

Reproduce only the newly measured external-engine artifacts with:

The complete benchmark implementation is checked in as
[`flagquantum/benchmarking/external_simulator_compare.py`](../../../flagquantum/benchmarking/external_simulator_compare.py).
The commands below invoke that code through the installed CLI and regenerate
the two raw JSON artifacts; the Markdown table is not the benchmark source.

```bash
pip install -e '.[cirq,pennylane]'
pip install 'cirq-core==1.7.0' 'pennylane==0.45.1' \
  'pennylane-lightning==0.45.0'
flagquantum-benchmark run simulator_compare_cirq \
  --n-wires 10 14 18 22 24 --layers 2 --threads 1 \
  --warmup 3 --iterations 9 --setup-iterations 3 --calls-per-sample 10 \
  --json-output benchmarks/results/comparison/cirq_cpu_arm64_20260923.json
flagquantum-benchmark run simulator_compare_pennylane \
  --n-wires 10 14 18 22 24 --layers 2 --threads 1 \
  --warmup 3 --iterations 9 --setup-iterations 3 --calls-per-sample 10 \
  --json-output \
  benchmarks/results/comparison/pennylane_lightning_cpu_arm64_20260923.json
```

The new artifacts explicitly record `flagquantum_performance_measured=false`.
The FlagQuantum and Aer values in this report come from
`flagquantum_qiskit_aer_cpu_arm64_20260923.json`; no Aer samples were rerun or
modified.
