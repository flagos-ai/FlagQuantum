# CPU batched exact-statevector scorecard

## Conclusion

This benchmark evaluates one complete user task: run the same circuit structure
with 32 independent parameter bindings and return 32 exact complex128
statevectors. On this Apple arm64 host, FlagQuantum's native parameter batch is
strong at 10 qubits and remains useful at 14 qubits, but reaches a clear
wide-state batching cliff at 18 qubits. The result identifies an optimization
target; it is not evidence that FlagQuantum is universally the fastest engine.

The external columns measure the current FlagQuantum ecosystem bridge paths.
Each bridge is called once per binding because the public bridges do not yet
accept a parameter batch. These values must not be presented as the external
frameworks' best raw native-batch performance.

## Measured batch-32 result

`vs FQ` is external or serial total time divided by FlagQuantum native-batch
total time. A value above `1.00x` means FlagQuantum completed the identical
32-statevector task faster; a value below `1.00x` means it was slower.

| Workload | Qubits | FQ total (ms) | FQ states/s | FQ serial vs FQ | Qiskit Aer bridge vs FQ | Cirq bridge vs FQ | PennyLane Lightning bridge vs FQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware efficient | 10 | 2.857 | 11,200.07 | 9.13x | 245.43x | 38.46x | 19.06x |
| Hardware efficient | 14 | 33.491 | 955.48 | 1.76x | 24.19x | 7.45x | 3.13x |
| Hardware efficient | 18 | 719.982 | 44.45 | 0.87x | 2.57x | 2.79x | 1.04x |
| Truncated QFT | 10 | 2.300 | 13,912.04 | 8.12x | 342.80x | 67.67x | 44.26x |
| Truncated QFT | 14 | 37.973 | 842.70 | 1.36x | 22.16x | 6.58x | 5.30x |
| Truncated QFT | 18 | 789.476 | 40.53 | 0.93x | 2.93x | 1.22x | 1.76x |
| Random Clifford | 10 | 4.592 | 6,968.14 | 6.36x | 163.45x | 17.93x | 11.60x |
| Random Clifford | 14 | 52.560 | 608.83 | 1.47x | 14.75x | 2.99x | 1.57x |
| Random Clifford | 18 | 1,077.809 | 29.69 | 0.91x | 1.59x | 0.90x | 0.40x |
| Local brickwork | 10 | 4.240 | 7,547.47 | 6.85x | 161.42x | 28.06x | 13.28x |
| Local brickwork | 14 | 59.245 | 540.13 | 1.49x | 13.34x | 4.59x | 1.98x |
| Local brickwork | 18 | 922.191 | 34.70 | 0.90x | 1.73x | 2.15x | 0.94x |
| Dense nonlocal | 10 | 2.080 | 15,384.92 | 5.10x | 357.24x | 49.82x | 33.08x |
| Dense nonlocal | 14 | 20.603 | 1,553.21 | 1.35x | 42.24x | 11.35x | 6.93x |
| Dense nonlocal | 18 | 369.459 | 86.61 | 0.89x | 7.91x | 4.24x | 1.42x |

The 10-qubit batch wins over FlagQuantum serial by `5.10x` to `9.13x`. At
18 qubits, however, all five native batches are `8%` to `15%` slower than
repeating the native scalar path. Random Clifford also loses to the current
Cirq and PennyLane Lightning bridge paths, and local brickwork narrowly loses
to PennyLane Lightning. This is consistent with a memory-layout or working-set
cliff and makes budgeted/chunked native CPU batching the next implementation
target.

All 45 cases passed exact-statevector comparison with absolute tolerance
`1e-10`. The main run marked 43 of 45 cases stable. The two noisy measurements
were repeated with two warmups and nine measured iterations; all focused rerun
measurements passed the `0.20` relative-median-absolute-deviation threshold.

## Reproduce

The complete raw samples and environment metadata are in
[`batched_statevector_corpus_cpu_arm64_20260930.json`](batched_statevector_corpus_cpu_arm64_20260930.json).
The generated full table is in
[`BATCHED_STATEVECTOR_CORPUS_CPU_ARM64_20260930.md`](https://github.com/FlagQuantum/FlagQuantum-evidence/releases/tag/evidence-2026-10-09.2).
The stability rerun is recorded in
[`batched_statevector_corpus_stability_cpu_arm64_20260930.json`](batched_statevector_corpus_stability_cpu_arm64_20260930.json).

```bash
pip install -e '.[qiskit,cirq,pennylane]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
flagquantum-benchmark run batched_statevector_corpus \
  --workloads hardware_efficient_statevector truncated_qft_statevector \
    random_clifford_statevector local_brickwork_statevector \
    dense_nonlocal_statevector \
  --n-wires 10 14 18 --batch-sizes 1 8 32 \
  --threads 1 --warmup 1 --iterations 5 \
  --json-output batched-statevectors.json \
  --markdown-output batched-statevectors.md
```

Environment: Apple arm64, CPU, one PyTorch/OpenMP/BLAS thread, Python and package
versions recorded in the JSON. Circuit construction is excluded; conversion,
backend preparation, execution, and result retrieval are included. The result
is local non-release comparison evidence with
`scalability_claim_allowed=false`. Logical statevector bytes are recorded, but
process peak RSS is not measured.
