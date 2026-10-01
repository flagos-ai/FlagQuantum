# Cross-framework CPU batched statevector time and peak RSS

Generated from [`batched_statevector_native_parameterized_cpu_arm64_20261001.json`](batched_statevector_native_parameterized_cpu_arm64_20261001.json). Timings are warm
same-process medians. Each peak RSS sample comes from one fresh process
running one complete cold task, so one engine cannot
contaminate another's high-water mark. Each displayed value is the median
of those probes; raw observations remain in JSON. The process baseline and
circuit construction are included.

| Workload | Qubits | Batch | Engine | Median time (ms) | vs FQ batch | Peak RSS (MiB) | Execution RSS growth (MiB) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 801.939 | 1.00x | 710.5 | 533.6 |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch (parameterized-layer rollback) | 908.163 | 1.13x | 913.8 | 720.5 |
| random_clifford_statevector | 18 | 32 | PennyLane Lightning bridge | 454.045 | 0.57x | 495.9 | 303.5 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (budgeted) | 941.103 | 1.00x | 914.2 | 721.9 |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch (parameterized-layer rollback) | 1078.723 | 1.15x | 914.1 | 721.9 |
| local_brickwork_statevector | 18 | 32 | PennyLane Lightning bridge | 1067.262 | 1.13x | 558.0 | 364.8 |

Ratios above one mean FlagQuantum native batch was faster. Peak RSS
includes framework/interpreter baseline and is local comparison evidence,
not a universal framework ranking or a release/scalability claim.
