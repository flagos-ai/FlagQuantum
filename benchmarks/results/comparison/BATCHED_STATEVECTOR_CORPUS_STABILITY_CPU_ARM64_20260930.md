# Cross-framework CPU batched statevector throughput

Generated from [`batched_statevector_corpus_stability_cpu_arm64_20260930.json`](batched_statevector_corpus_stability_cpu_arm64_20260930.json). Each task returns N
exact statevectors for N independent parameter bindings. FlagQuantum uses
its native parameter batch; current external bridges accept one item and
are therefore invoked repeatedly. This measures the FlagQuantum user-facing
interop surface, not each external framework's best raw batching API.

| Workload | Qubits | Batch | Engine | Total (ms) | Per state (ms) | States/s | vs FQ batch |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| random_clifford_statevector | 10 | 1 | FlagQuantum native batch | 1.043 | 1.043 | 958.77 | 1.00x |
| random_clifford_statevector | 10 | 1 | FlagQuantum serial | 0.991 | 0.991 | 1008.78 | 0.95x |
| random_clifford_statevector | 10 | 1 | Qiskit Aer bridge | 22.595 | 22.595 | 44.26 | 21.66x |
| random_clifford_statevector | 10 | 1 | Cirq bridge | 2.646 | 2.646 | 377.86 | 2.54x |
| random_clifford_statevector | 10 | 1 | PennyLane Lightning bridge | 1.874 | 1.874 | 533.48 | 1.80x |
| random_clifford_statevector | 10 | 32 | FlagQuantum native batch | 4.121 | 0.129 | 7765.26 | 1.00x |
| random_clifford_statevector | 10 | 32 | FlagQuantum serial | 26.412 | 0.825 | 1211.59 | 6.41x |
| random_clifford_statevector | 10 | 32 | Qiskit Aer bridge | 724.847 | 22.651 | 44.15 | 175.89x |
| random_clifford_statevector | 10 | 32 | Cirq bridge | 80.916 | 2.529 | 395.47 | 19.64x |
| random_clifford_statevector | 10 | 32 | PennyLane Lightning bridge | 51.529 | 1.610 | 621.01 | 12.50x |
| dense_nonlocal_statevector | 10 | 1 | FlagQuantum native batch | 0.559 | 0.559 | 1788.24 | 1.00x |
| dense_nonlocal_statevector | 10 | 1 | FlagQuantum serial | 0.819 | 0.819 | 1221.68 | 1.46x |
| dense_nonlocal_statevector | 10 | 1 | Qiskit Aer bridge | 24.330 | 24.330 | 41.10 | 43.51x |
| dense_nonlocal_statevector | 10 | 1 | Cirq bridge | 2.905 | 2.905 | 344.26 | 5.19x |
| dense_nonlocal_statevector | 10 | 1 | PennyLane Lightning bridge | 2.396 | 2.396 | 417.44 | 4.28x |
| dense_nonlocal_statevector | 10 | 32 | FlagQuantum native batch | 1.531 | 0.048 | 20896.25 | 1.00x |
| dense_nonlocal_statevector | 10 | 32 | FlagQuantum serial | 11.147 | 0.348 | 2870.72 | 7.28x |
| dense_nonlocal_statevector | 10 | 32 | Qiskit Aer bridge | 755.018 | 23.594 | 42.38 | 493.03x |
| dense_nonlocal_statevector | 10 | 32 | Cirq bridge | 90.534 | 2.829 | 353.46 | 59.12x |
| dense_nonlocal_statevector | 10 | 32 | PennyLane Lightning bridge | 66.342 | 2.073 | 482.35 | 43.32x |

Ratios above one mean FlagQuantum native batch completed the identical
N-statevector task faster. Results are local comparison evidence, not a
universal framework ranking or release/scalability evidence.
