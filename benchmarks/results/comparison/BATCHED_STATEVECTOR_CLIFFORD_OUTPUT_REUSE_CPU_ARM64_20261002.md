# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_clifford_output_reuse_cpu_arm64_20261002.json`](batched_statevector_clifford_output_reuse_cpu_arm64_20261002.json). Timings are warm
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
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 275.685 | 1.00x | 569.2 | 389.1 |
| random_clifford_statevector | 18 | 32 | PennyLane Lightning native batch | 365.846 | 1.33x | 543.6 | 288.7 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
