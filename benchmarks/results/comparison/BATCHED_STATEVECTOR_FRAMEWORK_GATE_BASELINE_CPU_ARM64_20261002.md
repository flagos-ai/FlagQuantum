# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_framework_gate_baseline_cpu_arm64_20261002.json`](batched_statevector_framework_gate_baseline_cpu_arm64_20261002.json). Timings are warm
same-process medians. Each peak RSS sample comes from one fresh process
running one complete cold task, so one engine cannot
contaminate another's high-water mark. Each displayed value is the median
of those probes; raw observations remain in JSON. The process baseline and
circuit construction are included.

| Workload | Qubits | Batch | Engine | Median time (ms) | vs FQ batch | Peak RSS (MiB) | Execution RSS growth (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 544.188 | 1.00x | 575.7 | 391.7 |
| random_clifford_statevector | 18 | 32 | PennyLane Lightning bridge | 941.428 | 1.73x | 600.1 | 403.9 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 1084.115 | 1.00x | 540.1 | 353.1 |
| local_brickwork_statevector | 18 | 32 | PennyLane Lightning bridge | 1334.363 | 1.23x | 582.4 | 388.0 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
