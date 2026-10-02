# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_dense_width_cpu_arm64_20261002.json`](batched_statevector_dense_width_cpu_arm64_20261002.json). Timings are warm
same-process medians. Each peak RSS sample comes from one fresh process
running one complete cold task, so one engine cannot
contaminate another's high-water mark. Each displayed value is the median
of those probes; raw observations remain in JSON. The process baseline and
circuit construction are included.

| Workload | Qubits | Batch | Engine | Median time (ms) | vs FQ batch | Peak RSS (MiB) | Execution RSS growth (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 1088.262 | 1.00x | 655.9 | 461.3 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (4-wire dense rollback) | 1500.801 | 1.38x | 916.6 | 722.5 |
| local_brickwork_statevector | 18 | 32 | PennyLane Lightning bridge | 1350.228 | 1.24x | 612.7 | 417.5 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
