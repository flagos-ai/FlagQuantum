# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_native_clifford_cpu_arm64_20261001.json`](batched_statevector_native_clifford_cpu_arm64_20261001.json). Timings are warm
same-process medians. Each peak RSS sample comes from one fresh process
running one complete cold task, so one engine cannot
contaminate another's high-water mark. Each displayed value is the median
of those probes; raw observations remain in JSON. The process baseline and
circuit construction are included.

| Workload | Qubits | Batch | Engine | Median time (ms) | vs FQ batch | Peak RSS (MiB) | Execution RSS growth (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 989.682 | 1.00x | 1004.1 | 812.3 |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (fixed-layer rollback) | 1150.921 | 1.16x | 942.7 | 750.4 |
| random_clifford_statevector | 18 | 32 | PennyLane Lightning bridge | 453.091 | 0.46x | 557.5 | 364.4 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 889.736 | 1.00x | 913.8 | 721.8 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (fixed-layer rollback) | 894.631 | 1.01x | 913.9 | 721.9 |
| local_brickwork_statevector | 18 | 32 | PennyLane Lightning bridge | 866.504 | 0.97x | 557.0 | 364.3 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
