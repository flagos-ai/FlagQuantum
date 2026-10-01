# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_static_clifford_layer_cpu_arm64_20261001.json`](batched_statevector_static_clifford_layer_cpu_arm64_20261001.json). Timings are warm
same-process medians. Each peak RSS sample comes from one fresh process
running one complete cold task, so one engine cannot
contaminate another's high-water mark. Each displayed value is the median
of those probes; raw observations remain in JSON. The process baseline and
circuit construction are included.

| Workload | Qubits | Batch | Engine | Median time (ms) | vs FQ batch | Peak RSS (MiB) | Execution RSS growth (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 289.961 | 1.00x | 592.7 | 400.2 |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (static-Clifford-layer rollback) | 572.322 | 1.97x | 594.4 | 401.5 |
| random_clifford_statevector | 18 | 32 | PennyLane Lightning bridge | 425.728 | 1.47x | 535.4 | 341.9 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 976.444 | 1.00x | 840.9 | 647.5 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (static-Clifford-layer rollback) | 890.106 | 0.91x | 793.5 | 599.2 |
| local_brickwork_statevector | 18 | 32 | PennyLane Lightning bridge | 995.795 | 1.02x | 520.9 | 334.3 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
