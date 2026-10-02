# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_terminal_fused_rotation_cpu_arm64_20261002.json`](batched_statevector_terminal_fused_rotation_cpu_arm64_20261002.json). Timings are warm
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
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 373.666 | 1.00x | 567.8 | 389.0 |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum native batch (fused-rotation-layer rollback) | 421.472 | 1.13x | 623.9 | 458.0 |
| dense_nonlocal_statevector | 18 | 32 | PennyLane Lightning native batch | 807.468 | 2.16x | 549.0 | 326.2 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
