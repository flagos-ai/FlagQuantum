# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_preallocated_assembly_cpu_arm64_20261002.json`](batched_statevector_preallocated_assembly_cpu_arm64_20261002.json). Timings are warm
same-process medians. Each peak RSS sample comes from one fresh process
running one complete cold task, so one engine cannot
contaminate another's high-water mark. Each displayed value is the median
of those probes; raw observations remain in JSON. The process baseline and
circuit construction are included.
PennyLane Lightning native batch uses public broadcast expansion with
one-time device preprocessing outside warm timing. The separately named
bridge engine executes one public FlagQuantum bridge call per batch item.

| Workload | Qubits | Batch | Engine | Median time (ms) | vs FQ batch | Peak RSS (MiB) | Execution RSS growth (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| hardware_efficient_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 779.080 | 1.00x | 541.0 | 359.4 |
| hardware_efficient_statevector | 18 | 32 | FlagQuantum native batch (functional assembly rollback) | 758.675 | 0.97x | 658.8 | 465.4 |
| hardware_efficient_statevector | 18 | 32 | PennyLane Lightning native batch | 983.908 | 1.26x | 681.4 | 388.7 |
| truncated_qft_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 787.166 | 1.00x | 556.5 | 363.3 |
| truncated_qft_statevector | 18 | 32 | FlagQuantum native batch (functional assembly rollback) | 794.837 | 1.01x | 547.4 | 378.1 |
| truncated_qft_statevector | 18 | 32 | PennyLane Lightning native batch | 1896.252 | 2.41x | 466.7 | 214.2 |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 477.311 | 1.00x | 507.6 | 334.7 |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (functional assembly rollback) | 501.866 | 1.05x | 569.8 | 394.2 |
| random_clifford_statevector | 18 | 32 | PennyLane Lightning native batch | 774.880 | 1.62x | 568.6 | 325.8 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 1269.561 | 1.00x | 560.4 | 380.0 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (functional assembly rollback) | 1235.449 | 0.97x | 524.1 | 355.7 |
| local_brickwork_statevector | 18 | 32 | PennyLane Lightning native batch | 1396.672 | 1.10x | 504.0 | 254.8 |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 365.814 | 1.00x | 534.5 | 354.6 |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum native batch (functional assembly rollback) | 358.731 | 0.98x | 553.9 | 386.3 |
| dense_nonlocal_statevector | 18 | 32 | PennyLane Lightning native batch | 883.755 | 2.42x | 564.9 | 324.3 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
