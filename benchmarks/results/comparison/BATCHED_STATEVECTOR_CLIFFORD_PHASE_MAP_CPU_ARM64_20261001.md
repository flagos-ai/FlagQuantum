# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_clifford_phase_map_cpu_arm64_20261001.json`](batched_statevector_clifford_phase_map_cpu_arm64_20261001.json). Timings are warm
same-process medians. Each peak RSS sample comes from one fresh process
running one complete cold task, so one engine cannot
contaminate another's high-water mark. Each displayed value is the median
of those probes; raw observations remain in JSON. The process baseline and
circuit construction are included.

| Workload | Qubits | Batch | Engine | Median time (ms) | vs FQ batch | Peak RSS (MiB) | Execution RSS growth (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 605.827 | 1.00x | 591.4 | 398.9 |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (Clifford phase-map rollback) | 673.693 | 1.11x | 589.0 | 396.6 |
| random_clifford_statevector | 18 | 32 | PennyLane Lightning bridge | 437.393 | 0.72x | 529.1 | 335.7 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 1110.186 | 1.00x | 861.9 | 674.7 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (Clifford phase-map rollback) | 1197.026 | 1.08x | 868.7 | 676.7 |
| local_brickwork_statevector | 18 | 32 | PennyLane Lightning bridge | 1107.697 | 1.00x | 557.4 | 364.4 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
