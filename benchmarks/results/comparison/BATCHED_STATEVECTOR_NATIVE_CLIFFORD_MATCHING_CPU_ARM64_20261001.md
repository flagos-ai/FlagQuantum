# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_native_clifford_matching_cpu_arm64_20261001.json`](batched_statevector_native_clifford_matching_cpu_arm64_20261001.json). Timings are warm
same-process medians. Each peak RSS sample comes from one fresh process
running one complete cold task, so one engine cannot
contaminate another's high-water mark. Each displayed value is the median
of those probes; raw observations remain in JSON. The process baseline and
circuit construction are included.

| Workload | Qubits | Batch | Engine | Median time (ms) | vs FQ batch | Peak RSS (MiB) | Execution RSS growth (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 754.687 | 1.00x | 588.7 | 395.9 |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (Clifford-matching rollback) | 989.075 | 1.31x | 740.1 | 548.1 |
| random_clifford_statevector | 18 | 32 | PennyLane Lightning bridge | 519.045 | 0.69x | 557.6 | 364.5 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 972.286 | 1.00x | 866.2 | 673.1 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (Clifford-matching rollback) | 1165.631 | 1.20x | 914.0 | 722.1 |
| local_brickwork_statevector | 18 | 32 | PennyLane Lightning bridge | 934.724 | 0.96x | 506.9 | 313.7 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
