# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_inplace_diagonal_graphs_cpu_arm64_20261002.json`](batched_statevector_inplace_diagonal_graphs_cpu_arm64_20261002.json). Timings are warm
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
| truncated_qft_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 759.850 | 1.00x | 501.0 | 321.0 |
| truncated_qft_statevector | 18 | 32 | FlagQuantum native batch (functional diagonal-graph rollback) | 773.674 | 1.02x | 517.7 | 324.8 |
| truncated_qft_statevector | 18 | 32 | PennyLane Lightning native batch | 2032.140 | 2.67x | 514.4 | 261.4 |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 293.167 | 1.00x | 504.2 | 311.0 |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum native batch (functional diagonal-graph rollback) | 326.862 | 1.11x | 585.4 | 392.2 |
| dense_nonlocal_statevector | 18 | 32 | PennyLane Lightning native batch | 676.256 | 2.31x | 573.0 | 317.9 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
