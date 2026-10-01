# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_memory_preallocation_cpu_arm64_20261001.json`](batched_statevector_memory_preallocation_cpu_arm64_20261001.json). Timings are warm
same-process medians. Each peak RSS sample comes from one fresh process
running one complete cold task, so one engine cannot
contaminate another's high-water mark. Each displayed value is the median
of those probes; raw observations remain in JSON. The process baseline and
circuit construction are included.

| Workload | Qubits | Batch | Engine | Median time (ms) | vs FQ batch | Peak RSS (MiB) | Execution RSS growth (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| hardware_efficient_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 898.942 | 1.00x | 655.2 | 518.3 |
| hardware_efficient_statevector | 18 | 32 | FlagQuantum native batch (legacy functional windows) | 884.052 | 0.98x | 762.1 | 579.7 |
| truncated_qft_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 876.608 | 1.00x | 654.8 | 476.8 |
| truncated_qft_statevector | 18 | 32 | FlagQuantum native batch (legacy functional windows) | 1084.592 | 1.24x | 774.9 | 590.6 |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 1394.936 | 1.00x | 824.1 | 651.1 |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (legacy functional windows) | 1448.064 | 1.04x | 859.1 | 677.3 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 1182.160 | 1.00x | 859.2 | 698.2 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (legacy functional windows) | 1144.981 | 0.97x | 882.5 | 701.4 |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 660.327 | 1.00x | 578.9 | 398.4 |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum native batch (legacy functional windows) | 653.355 | 0.99x | 587.8 | 446.9 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
