# Socket-local independent-simulation throughput

## Conclusion

On one 32-core Intel Xeon Platinum 8358 socket, the best FlagQuantum
configuration has higher exact complex128 statevector throughput than the best
same-run Qiskit Aer and PennyLane Lightning configuration on all six 22-qubit
workloads. FlagQuantum's peak advantage over the fastest external engine ranges
from **1.228x to 8.119x**. Correctness and stability pass for every measured
cell.

| Workload | FlagQuantum peak | Best external peak | FlagQuantum advantage |
| --- | ---: | ---: | ---: |
| Hardware-efficient | 25.496 tasks/s | Qiskit Aer, 16.042 tasks/s | **1.589x** |
| Truncated QFT | 17.295 tasks/s | Qiskit Aer, 14.088 tasks/s | **1.228x** |
| Random Clifford | 25.126 tasks/s | Qiskit Aer, 17.887 tasks/s | **1.405x** |
| Local brickwork | 32.405 tasks/s | Qiskit Aer, 16.341 tasks/s | **1.983x** |
| Dense nonlocal | 34.162 tasks/s | Qiskit Aer, 7.437 tasks/s | **4.594x** |
| SWAP routing | 206.736 tasks/s | Qiskit Aer, 25.463 tasks/s | **8.119x** |

All peak values above use `8 workers x 4 threads`. This is a multitask
throughput result, not a universal single-circuit latency claim. At `1x32`, Aer
is still faster on Hardware-efficient, Truncated QFT, and Random Clifford; the
full table keeps those losses visible.

## Measurement contract

Generated from
[`socket_local_multitask_throughput_linux_x86_20261003.json`](socket_local_multitask_throughput_linux_x86_20261003.json).
Process startup and circuit construction are excluded; every timed task is one
user-facing exact-statevector run including conversion and result retrieval.
All configurations use physical CPUs `0-31` with disjoint `taskset` affinity,
no SMT, synchronized worker starts, one warmup per worker, and 16 timed tasks per
cell. Every worker receives its thread environment before Python imports
PyTorch or an external simulator.

The run used Python 3.12.13, PyTorch 2.13.0, Qiskit 2.5.1, Qiskit Aer 0.17.2,
PennyLane 0.45.1, and PennyLane Lightning 0.45.0 on a Linux x86-64 host. The
public artifact redacts the infrastructure hostname but retains all absolute
samples, p50/p95 latency, wall time, affinity, package versions, correctness
error, and RMAD values.

## Complete results

| Workload | Workers x threads | Engine | Tasks/s | p50 | p95 | vs 1xsocket |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| hardware_efficient_statevector | 1x32 | flagquantum_native | 6.992 | 142.454 ms | 146.105 ms | 1.000x |
| hardware_efficient_statevector | 1x32 | qiskit_aer | 7.115 | 140.098 ms | 146.007 ms | 1.000x |
| hardware_efficient_statevector | 1x32 | pennylane_lightning_qubit | 1.466 | 681.652 ms | 695.912 ms | 1.000x |
| hardware_efficient_statevector | 2x16 | flagquantum_native | 11.779 | 166.501 ms | 174.515 ms | 1.685x |
| hardware_efficient_statevector | 2x16 | qiskit_aer | 10.804 | 183.174 ms | 193.114 ms | 1.518x |
| hardware_efficient_statevector | 2x16 | pennylane_lightning_qubit | 2.283 | 834.970 ms | 885.286 ms | 1.558x |
| hardware_efficient_statevector | 4x8 | flagquantum_native | 18.441 | 214.448 ms | 218.859 ms | 2.637x |
| hardware_efficient_statevector | 4x8 | qiskit_aer | 13.848 | 286.977 ms | 293.061 ms | 1.946x |
| hardware_efficient_statevector | 4x8 | pennylane_lightning_qubit | 3.989 | 979.368 ms | 1005.181 ms | 2.722x |
| hardware_efficient_statevector | 8x4 | flagquantum_native | 25.496 | 297.020 ms | 316.263 ms | 3.646x |
| hardware_efficient_statevector | 8x4 | qiskit_aer | 16.042 | 495.296 ms | 503.864 ms | 2.255x |
| hardware_efficient_statevector | 8x4 | pennylane_lightning_qubit | 5.768 | 1370.307 ms | 1387.289 ms | 3.936x |
| truncated_qft_statevector | 1x32 | flagquantum_native | 4.626 | 220.687 ms | 235.200 ms | 1.000x |
| truncated_qft_statevector | 1x32 | qiskit_aer | 6.911 | 144.214 ms | 150.839 ms | 1.000x |
| truncated_qft_statevector | 1x32 | pennylane_lightning_qubit | 0.877 | 1138.979 ms | 1150.597 ms | 1.000x |
| truncated_qft_statevector | 2x16 | flagquantum_native | 7.766 | 255.523 ms | 265.305 ms | 1.679x |
| truncated_qft_statevector | 2x16 | qiskit_aer | 9.755 | 202.028 ms | 212.298 ms | 1.411x |
| truncated_qft_statevector | 2x16 | pennylane_lightning_qubit | 1.291 | 1497.053 ms | 1582.331 ms | 1.471x |
| truncated_qft_statevector | 4x8 | flagquantum_native | 12.303 | 322.996 ms | 333.608 ms | 2.660x |
| truncated_qft_statevector | 4x8 | qiskit_aer | 12.250 | 324.706 ms | 329.407 ms | 1.772x |
| truncated_qft_statevector | 4x8 | pennylane_lightning_qubit | 2.295 | 1699.610 ms | 1743.865 ms | 2.616x |
| truncated_qft_statevector | 8x4 | flagquantum_native | 17.295 | 456.096 ms | 465.366 ms | 3.739x |
| truncated_qft_statevector | 8x4 | qiskit_aer | 14.088 | 561.969 ms | 571.225 ms | 2.038x |
| truncated_qft_statevector | 8x4 | pennylane_lightning_qubit | 3.333 | 2369.946 ms | 2403.344 ms | 3.798x |
| random_clifford_statevector | 1x32 | flagquantum_native | 5.545 | 179.333 ms | 185.329 ms | 1.000x |
| random_clifford_statevector | 1x32 | qiskit_aer | 7.506 | 128.677 ms | 165.499 ms | 1.000x |
| random_clifford_statevector | 1x32 | pennylane_lightning_qubit | 2.112 | 469.909 ms | 486.141 ms | 1.000x |
| random_clifford_statevector | 2x16 | flagquantum_native | 11.165 | 178.071 ms | 182.787 ms | 2.014x |
| random_clifford_statevector | 2x16 | qiskit_aer | 11.799 | 168.812 ms | 172.189 ms | 1.572x |
| random_clifford_statevector | 2x16 | pennylane_lightning_qubit | 3.535 | 553.604 ms | 570.411 ms | 1.674x |
| random_clifford_statevector | 4x8 | flagquantum_native | 17.756 | 218.567 ms | 226.780 ms | 3.202x |
| random_clifford_statevector | 4x8 | qiskit_aer | 15.467 | 258.083 ms | 262.951 ms | 2.061x |
| random_clifford_statevector | 4x8 | pennylane_lightning_qubit | 6.148 | 637.249 ms | 652.880 ms | 2.911x |
| random_clifford_statevector | 8x4 | flagquantum_native | 25.126 | 305.390 ms | 318.140 ms | 4.532x |
| random_clifford_statevector | 8x4 | qiskit_aer | 17.887 | 443.072 ms | 449.615 ms | 2.383x |
| random_clifford_statevector | 8x4 | pennylane_lightning_qubit | 9.069 | 874.532 ms | 884.469 ms | 4.294x |
| local_brickwork_statevector | 1x32 | flagquantum_native | 24.644 | 40.334 ms | 41.679 ms | 1.000x |
| local_brickwork_statevector | 1x32 | qiskit_aer | 7.351 | 135.688 ms | 138.753 ms | 1.000x |
| local_brickwork_statevector | 1x32 | pennylane_lightning_qubit | 1.146 | 872.829 ms | 882.546 ms | 1.000x |
| local_brickwork_statevector | 2x16 | flagquantum_native | 28.646 | 68.598 ms | 78.597 ms | 1.162x |
| local_brickwork_statevector | 2x16 | qiskit_aer | 10.928 | 181.395 ms | 196.057 ms | 1.487x |
| local_brickwork_statevector | 2x16 | pennylane_lightning_qubit | 1.843 | 1041.668 ms | 1092.160 ms | 1.609x |
| local_brickwork_statevector | 4x8 | flagquantum_native | 31.119 | 126.097 ms | 132.672 ms | 1.263x |
| local_brickwork_statevector | 4x8 | qiskit_aer | 13.260 | 285.606 ms | 318.809 ms | 1.804x |
| local_brickwork_statevector | 4x8 | pennylane_lightning_qubit | 3.224 | 1225.330 ms | 1241.970 ms | 2.814x |
| local_brickwork_statevector | 8x4 | flagquantum_native | 32.405 | 241.766 ms | 246.990 ms | 1.315x |
| local_brickwork_statevector | 8x4 | qiskit_aer | 16.341 | 476.718 ms | 495.343 ms | 2.223x |
| local_brickwork_statevector | 8x4 | pennylane_lightning_qubit | 4.678 | 1691.642 ms | 1712.536 ms | 4.083x |
| dense_nonlocal_statevector | 1x32 | flagquantum_native | 14.007 | 70.897 ms | 73.616 ms | 1.000x |
| dense_nonlocal_statevector | 1x32 | qiskit_aer | 4.549 | 218.162 ms | 232.115 ms | 1.000x |
| dense_nonlocal_statevector | 1x32 | pennylane_lightning_qubit | 1.865 | 535.621 ms | 541.297 ms | 1.000x |
| dense_nonlocal_statevector | 2x16 | flagquantum_native | 23.807 | 83.553 ms | 88.576 ms | 1.700x |
| dense_nonlocal_statevector | 2x16 | qiskit_aer | 6.027 | 332.385 ms | 341.422 ms | 1.325x |
| dense_nonlocal_statevector | 2x16 | pennylane_lightning_qubit | 3.007 | 649.108 ms | 683.241 ms | 1.613x |
| dense_nonlocal_statevector | 4x8 | flagquantum_native | 30.732 | 128.860 ms | 135.410 ms | 2.194x |
| dense_nonlocal_statevector | 4x8 | qiskit_aer | 6.966 | 571.871 ms | 581.732 ms | 1.531x |
| dense_nonlocal_statevector | 4x8 | pennylane_lightning_qubit | 4.707 | 831.589 ms | 852.035 ms | 2.524x |
| dense_nonlocal_statevector | 8x4 | flagquantum_native | 34.162 | 232.567 ms | 235.616 ms | 2.439x |
| dense_nonlocal_statevector | 8x4 | qiskit_aer | 7.437 | 1069.130 ms | 1094.909 ms | 1.635x |
| dense_nonlocal_statevector | 8x4 | pennylane_lightning_qubit | 7.121 | 1112.228 ms | 1124.709 ms | 3.819x |
| swap_routing_statevector | 1x32 | flagquantum_native | 41.915 | 23.214 ms | 28.676 ms | 1.000x |
| swap_routing_statevector | 1x32 | qiskit_aer | 8.729 | 111.400 ms | 128.858 ms | 1.000x |
| swap_routing_statevector | 1x32 | pennylane_lightning_qubit | 1.707 | 583.418 ms | 600.901 ms | 1.000x |
| swap_routing_statevector | 2x16 | flagquantum_native | 82.262 | 23.574 ms | 29.271 ms | 1.963x |
| swap_routing_statevector | 2x16 | qiskit_aer | 14.426 | 137.391 ms | 142.666 ms | 1.653x |
| swap_routing_statevector | 2x16 | pennylane_lightning_qubit | 2.809 | 683.732 ms | 720.387 ms | 1.645x |
| swap_routing_statevector | 4x8 | flagquantum_native | 132.383 | 28.545 ms | 36.683 ms | 3.158x |
| swap_routing_statevector | 4x8 | qiskit_aer | 20.672 | 192.539 ms | 199.276 ms | 2.368x |
| swap_routing_statevector | 4x8 | pennylane_lightning_qubit | 4.898 | 796.411 ms | 817.756 ms | 2.869x |
| swap_routing_statevector | 8x4 | flagquantum_native | 206.736 | 35.648 ms | 39.697 ms | 4.932x |
| swap_routing_statevector | 8x4 | qiskit_aer | 25.463 | 308.452 ms | 315.523 ms | 2.917x |
| swap_routing_statevector | 8x4 | pennylane_lightning_qubit | 7.111 | 1113.406 ms | 1128.354 ms | 4.165x |

Ratios describe replicated independent work on one socket, not single-circuit
latency scaling, multi-socket scaling, distributed statevectors, or cold start.

## Why FlagQuantum improved

Profiling identified three production-path gaps rather than a benchmark-only
shortcut:

- Random Clifford spent about half of its four-thread runtime applying one
  full-width static Clifford layer through repeated PyTorch tensor operations.
  Wide product-state components now reuse the existing native fused static
  Clifford kernel. A focused 22-qubit run decreased from 443.081 ms with
  `FQ_CPU_PRODUCT_STATE_NATIVE_STATIC_CLIFFORD=0` to 254.058 ms.
- Local brickwork compiled each rotation layer into dense `bmm` regions and
  then traversed the state again for a disjoint CX matching. Wide scalar
  inference can now select the existing native rotation-plus-CX tiles. Its
  focused median decreased from 724.626 ms with
  `FQ_CPU_NATIVE_SCALAR_CLIFFORD_MATCHING=0` to 239.590 ms, a 3.024x speedup.
- Hardware-efficient circuits contain overlapping CX chains, so they cannot
  use the disjoint matching optimization. Their wide scalar RX/RY/RZ regions
  now use native fused rotation tiles before the unchanged CX chain. The
  focused median decreased from 451.601 ms with
  `FQ_CPU_NATIVE_SCALAR_FUSED_ROTATION_LAYER=0` to 325.360 ms, a 1.388x
  speedup.

The new scalar selections require CPU inference, batch size one, at least 16
qubits, a loadable native extension, and no autograd dependency. Smaller
states, gradients, unsupported operations, non-CPU devices, and any disabled
feature retain the established paths. Each optimization has the independent
rollback shown above.

## Reproduction

Build the native CPU extension and install the Qiskit Aer and PennyLane
Lightning benchmark extras, then run:

```bash
python -m flagquantum.benchmarking.socket_local_throughput \
  --workloads hardware_efficient_statevector \
    truncated_qft_statevector random_clifford_statevector \
    local_brickwork_statevector dense_nonlocal_statevector \
    swap_routing_statevector \
  --n-wires 22 \
  --engines flagquantum_native qiskit_aer pennylane_lightning_qubit \
  --cpu-list 0-31 --workers 1 2 4 8 \
  --total-tasks 16 --warmup 1 --socket-id 0 \
  --json-output socket-local.json \
  --markdown-output SOCKET_LOCAL.md
```

To reproduce the focused rollback measurements, use the existing
`simulator_workload_corpus` runner with four pinned threads, 2 warmups, and 11
iterations, setting exactly one rollback variable to `0` per run.

## Boundaries and stopping condition

This result measures independent exact-statevector jobs sharing one socket. It
does not measure one circuit scaling across workers, multi-socket NUMA, GPU,
distributed statevectors, cold start, sampling-only output, or approximate
simulation. The report is development hardware evidence and is not a release
gate.

This stage stops because the diff and public API are bounded, all focused
complex64/complex128 and rollback tests pass, every formal cell passes
correctness and stability, and FlagQuantum's best throughput exceeds the best
external throughput on every maintained workload. Single-worker latency gaps
remain explicit follow-up targets rather than being represented as closed.
