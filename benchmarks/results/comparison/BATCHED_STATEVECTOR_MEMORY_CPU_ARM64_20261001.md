# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_memory_cpu_arm64_20261001.json`](batched_statevector_memory_cpu_arm64_20261001.json). Timings are warm
same-process medians. Each peak RSS value comes from one fresh process
running one complete cold task, so one engine cannot contaminate another's
high-water mark. The process baseline and circuit construction are included.

| Workload | Qubits | Batch | Engine | Median time (ms) | vs FQ batch | Peak RSS (MiB) | Execution RSS growth (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| hardware_efficient_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 813.310 | 1.00x | 1042.5 | 848.8 |
| hardware_efficient_statevector | 18 | 32 | FlagQuantum native monolithic batch | 930.222 | 1.14x | 1363.3 | 1169.2 |
| hardware_efficient_statevector | 18 | 32 | FlagQuantum serial | 668.579 | 0.82x | 607.4 | 414.0 |
| hardware_efficient_statevector | 18 | 32 | Qiskit Aer bridge | 1923.195 | 2.36x | 496.1 | 301.8 |
| hardware_efficient_statevector | 18 | 32 | Cirq bridge | 2114.284 | 2.60x | 630.4 | 435.7 |
| hardware_efficient_statevector | 18 | 32 | PennyLane Lightning bridge | 727.017 | 0.89x | 637.4 | 442.3 |
| truncated_qft_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 863.867 | 1.00x | 1166.1 | 972.2 |
| truncated_qft_statevector | 18 | 32 | FlagQuantum native monolithic batch | 986.409 | 1.14x | 1416.3 | 1222.5 |
| truncated_qft_statevector | 18 | 32 | FlagQuantum serial | 700.964 | 0.81x | 618.4 | 424.1 |
| truncated_qft_statevector | 18 | 32 | Qiskit Aer bridge | 2222.748 | 2.57x | 496.6 | 302.7 |
| truncated_qft_statevector | 18 | 32 | Cirq bridge | 915.210 | 1.06x | 627.4 | 433.5 |
| truncated_qft_statevector | 18 | 32 | PennyLane Lightning bridge | 1303.141 | 1.51x | 671.5 | 477.6 |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 1074.345 | 1.00x | 1167.6 | 973.8 |
| random_clifford_statevector | 18 | 32 | FlagQuantum native monolithic batch | 1213.435 | 1.13x | 1615.5 | 1421.8 |
| random_clifford_statevector | 18 | 32 | FlagQuantum serial | 969.539 | 0.90x | 616.1 | 422.2 |
| random_clifford_statevector | 18 | 32 | Qiskit Aer bridge | 1609.796 | 1.50x | 496.0 | 300.9 |
| random_clifford_statevector | 18 | 32 | Cirq bridge | 945.414 | 0.88x | 623.1 | 429.2 |
| random_clifford_statevector | 18 | 32 | PennyLane Lightning bridge | 404.360 | 0.38x | 671.2 | 477.3 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 934.087 | 1.00x | 1042.8 | 849.5 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native monolithic batch | 1107.615 | 1.19x | 1363.4 | 1169.8 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum serial | 857.850 | 0.92x | 612.2 | 419.0 |
| local_brickwork_statevector | 18 | 32 | Qiskit Aer bridge | 1671.102 | 1.79x | 497.4 | 302.3 |
| local_brickwork_statevector | 18 | 32 | Cirq bridge | 2016.297 | 2.16x | 571.4 | 376.5 |
| local_brickwork_statevector | 18 | 32 | PennyLane Lightning bridge | 863.757 | 0.92x | 476.1 | 281.5 |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 439.069 | 1.00x | 1042.0 | 848.1 |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum native monolithic batch | 585.853 | 1.33x | 1362.1 | 1168.2 |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum serial | 407.270 | 0.93x | 612.7 | 419.0 |
| dense_nonlocal_statevector | 18 | 32 | Qiskit Aer bridge | 3081.385 | 7.02x | 496.5 | 302.1 |
| dense_nonlocal_statevector | 18 | 32 | Cirq bridge | 1692.038 | 3.85x | 626.3 | 431.7 |
| dense_nonlocal_statevector | 18 | 32 | PennyLane Lightning bridge | 530.593 | 1.21x | 671.7 | 477.6 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
