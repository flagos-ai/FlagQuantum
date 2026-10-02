# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_pennylane_native_batch_cpu_arm64_20261002.json`](batched_statevector_pennylane_native_batch_cpu_arm64_20261002.json). Timings are warm
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
| hardware_efficient_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 582.761 | 1.00x | 636.3 | 442.2 |
| hardware_efficient_statevector | 18 | 32 | PennyLane Lightning native batch | 748.519 | 1.28x | 749.9 | 388.2 |
| hardware_efficient_statevector | 18 | 32 | PennyLane Lightning bridge | 808.712 | 1.39x | 666.5 | 470.7 |
| truncated_qft_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 592.609 | 1.00x | 582.5 | 388.8 |
| truncated_qft_statevector | 18 | 32 | PennyLane Lightning native batch | 1346.926 | 2.27x | 775.0 | 385.3 |
| truncated_qft_statevector | 18 | 32 | PennyLane Lightning bridge | 1406.096 | 2.37x | 622.5 | 427.5 |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 268.050 | 1.00x | 583.4 | 390.3 |
| random_clifford_statevector | 18 | 32 | PennyLane Lightning native batch | 370.231 | 1.38x | 785.4 | 379.4 |
| random_clifford_statevector | 18 | 32 | PennyLane Lightning bridge | 400.739 | 1.50x | 670.3 | 475.8 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 682.513 | 1.00x | 630.7 | 437.2 |
| local_brickwork_statevector | 18 | 32 | PennyLane Lightning native batch | 838.924 | 1.23x | 740.6 | 357.7 |
| local_brickwork_statevector | 18 | 32 | PennyLane Lightning bridge | 928.575 | 1.36x | 648.6 | 465.5 |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 231.709 | 1.00x | 639.3 | 462.4 |
| dense_nonlocal_statevector | 18 | 32 | PennyLane Lightning native batch | 431.296 | 1.86x | 726.6 | 342.4 |
| dense_nonlocal_statevector | 18 | 32 | PennyLane Lightning bridge | 499.661 | 2.16x | 586.9 | 392.5 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
