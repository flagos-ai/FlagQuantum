# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_layout_lifetime_cpu_arm64_20261001.json`](batched_statevector_layout_lifetime_cpu_arm64_20261001.json). Timings are warm
same-process medians. Each peak RSS sample comes from one fresh process
running one complete cold task, so one engine cannot
contaminate another's high-water mark. Each displayed value is the median
of those probes; raw observations remain in JSON. The process baseline and
circuit construction are included.

| Workload | Qubits | Batch | Engine | Median time (ms) | vs FQ batch | Peak RSS (MiB) | Execution RSS growth (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 1049.385 | 1.00x | 941.9 | 749.9 |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (legacy layout retention) | 1053.283 | 1.00x | 1005.8 | 813.9 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 1061.349 | 1.00x | 894.5 | 702.5 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (legacy layout retention) | 1057.404 | 1.00x | 977.6 | 785.5 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
