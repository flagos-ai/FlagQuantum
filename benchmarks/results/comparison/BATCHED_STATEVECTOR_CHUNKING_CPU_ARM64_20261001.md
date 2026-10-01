# Cross-framework CPU batched statevector throughput

Generated from [`batched_statevector_chunking_cpu_arm64_20261001.json`](batched_statevector_chunking_cpu_arm64_20261001.json). Each task returns N
exact statevectors for N independent parameter bindings. FlagQuantum uses
its native parameter batch; current external bridges accept one item and
are therefore invoked repeatedly. This measures the FlagQuantum user-facing
interop surface, not each external framework's best raw batching API.

| Workload | Qubits | Batch | Engine | Total (ms) | Per state (ms) | States/s | vs FQ batch |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| hardware_efficient_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 705.031 | 22.032 | 45.39 | 1.00x |
| hardware_efficient_statevector | 18 | 32 | FlagQuantum native monolithic batch | 776.374 | 24.262 | 41.22 | 1.10x |
| hardware_efficient_statevector | 18 | 32 | FlagQuantum serial | 593.294 | 18.540 | 53.94 | 0.84x |
| hardware_efficient_statevector | 18 | 32 | Qiskit Aer bridge | 1590.864 | 49.715 | 20.11 | 2.26x |
| hardware_efficient_statevector | 18 | 32 | Cirq bridge | 1834.387 | 57.325 | 17.44 | 2.60x |
| hardware_efficient_statevector | 18 | 32 | PennyLane Lightning bridge | 641.079 | 20.034 | 49.92 | 0.91x |
| truncated_qft_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 1290.929 | 40.342 | 24.79 | 1.00x |
| truncated_qft_statevector | 18 | 32 | FlagQuantum native monolithic batch | 1356.132 | 42.379 | 23.60 | 1.05x |
| truncated_qft_statevector | 18 | 32 | FlagQuantum serial | 1100.812 | 34.400 | 29.07 | 0.85x |
| truncated_qft_statevector | 18 | 32 | Qiskit Aer bridge | 3101.784 | 96.931 | 10.32 | 2.40x |
| truncated_qft_statevector | 18 | 32 | Cirq bridge | 1250.088 | 39.065 | 25.60 | 0.97x |
| truncated_qft_statevector | 18 | 32 | PennyLane Lightning bridge | 1813.558 | 56.674 | 17.64 | 1.40x |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 1338.473 | 41.827 | 23.91 | 1.00x |
| random_clifford_statevector | 18 | 32 | FlagQuantum native monolithic batch | 1516.169 | 47.380 | 21.11 | 1.13x |
| random_clifford_statevector | 18 | 32 | FlagQuantum serial | 1255.735 | 39.242 | 25.48 | 0.94x |
| random_clifford_statevector | 18 | 32 | Qiskit Aer bridge | 1930.114 | 60.316 | 16.58 | 1.44x |
| random_clifford_statevector | 18 | 32 | Cirq bridge | 1166.824 | 36.463 | 27.42 | 0.87x |
| random_clifford_statevector | 18 | 32 | PennyLane Lightning bridge | 505.141 | 15.786 | 63.35 | 0.38x |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 1281.584 | 40.049 | 24.97 | 1.00x |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native monolithic batch | 1344.053 | 42.002 | 23.81 | 1.05x |
| local_brickwork_statevector | 18 | 32 | FlagQuantum serial | 1091.893 | 34.122 | 29.31 | 0.85x |
| local_brickwork_statevector | 18 | 32 | Qiskit Aer bridge | 1718.701 | 53.709 | 18.62 | 1.34x |
| local_brickwork_statevector | 18 | 32 | Cirq bridge | 2235.310 | 69.853 | 14.32 | 1.74x |
| local_brickwork_statevector | 18 | 32 | PennyLane Lightning bridge | 984.076 | 30.752 | 32.52 | 0.77x |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 349.150 | 10.911 | 91.65 | 1.00x |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum native monolithic batch | 442.847 | 13.839 | 72.26 | 1.27x |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum serial | 291.855 | 9.120 | 109.64 | 0.84x |
| dense_nonlocal_statevector | 18 | 32 | Qiskit Aer bridge | 2716.033 | 84.876 | 11.78 | 7.78x |
| dense_nonlocal_statevector | 18 | 32 | Cirq bridge | 1411.837 | 44.120 | 22.67 | 4.04x |
| dense_nonlocal_statevector | 18 | 32 | PennyLane Lightning bridge | 458.637 | 14.332 | 69.77 | 1.31x |

Ratios above one mean FlagQuantum native batch completed the identical
N-statevector task faster. Results are local comparison evidence, not a
universal framework ranking or release/scalability evidence.
