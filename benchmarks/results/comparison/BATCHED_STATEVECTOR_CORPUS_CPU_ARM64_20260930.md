# Cross-framework CPU batched statevector throughput

Generated from [`batched_statevector_corpus_cpu_arm64_20260930.json`](batched_statevector_corpus_cpu_arm64_20260930.json). Each task returns N
exact statevectors for N independent parameter bindings. FlagQuantum uses
its native parameter batch; current external bridges accept one item and
are therefore invoked repeatedly. This measures the FlagQuantum user-facing
interop surface, not each external framework's best raw batching API.

| Workload | Qubits | Batch | Engine | Total (ms) | Per state (ms) | States/s | vs FQ batch |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| hardware_efficient_statevector | 10 | 1 | FlagQuantum native batch | 1.762 | 1.762 | 567.43 | 1.00x |
| hardware_efficient_statevector | 10 | 1 | FlagQuantum serial | 1.038 | 1.038 | 963.78 | 0.59x |
| hardware_efficient_statevector | 10 | 1 | Qiskit Aer bridge | 23.113 | 23.113 | 43.27 | 13.11x |
| hardware_efficient_statevector | 10 | 1 | Cirq bridge | 4.574 | 4.574 | 218.62 | 2.60x |
| hardware_efficient_statevector | 10 | 1 | PennyLane Lightning bridge | 2.108 | 2.108 | 474.41 | 1.20x |
| hardware_efficient_statevector | 10 | 8 | FlagQuantum native batch | 1.466 | 0.183 | 5457.34 | 1.00x |
| hardware_efficient_statevector | 10 | 8 | FlagQuantum serial | 7.071 | 0.884 | 1131.43 | 4.82x |
| hardware_efficient_statevector | 10 | 8 | Qiskit Aer bridge | 176.661 | 22.083 | 45.28 | 120.51x |
| hardware_efficient_statevector | 10 | 8 | Cirq bridge | 28.175 | 3.522 | 283.94 | 19.22x |
| hardware_efficient_statevector | 10 | 8 | PennyLane Lightning bridge | 14.448 | 1.806 | 553.71 | 9.86x |
| hardware_efficient_statevector | 10 | 32 | FlagQuantum native batch | 2.857 | 0.089 | 11200.07 | 1.00x |
| hardware_efficient_statevector | 10 | 32 | FlagQuantum serial | 26.074 | 0.815 | 1227.26 | 9.13x |
| hardware_efficient_statevector | 10 | 32 | Qiskit Aer bridge | 701.234 | 21.914 | 45.63 | 245.43x |
| hardware_efficient_statevector | 10 | 32 | Cirq bridge | 109.893 | 3.434 | 291.19 | 38.46x |
| hardware_efficient_statevector | 10 | 32 | PennyLane Lightning bridge | 54.460 | 1.702 | 587.59 | 19.06x |
| hardware_efficient_statevector | 14 | 1 | FlagQuantum native batch | 2.695 | 2.695 | 371.10 | 1.00x |
| hardware_efficient_statevector | 14 | 1 | FlagQuantum serial | 2.008 | 2.008 | 497.88 | 0.75x |
| hardware_efficient_statevector | 14 | 1 | Qiskit Aer bridge | 25.537 | 25.537 | 39.16 | 9.48x |
| hardware_efficient_statevector | 14 | 1 | Cirq bridge | 7.758 | 7.758 | 128.90 | 2.88x |
| hardware_efficient_statevector | 14 | 1 | PennyLane Lightning bridge | 3.407 | 3.407 | 293.55 | 1.26x |
| hardware_efficient_statevector | 14 | 8 | FlagQuantum native batch | 8.987 | 1.123 | 890.14 | 1.00x |
| hardware_efficient_statevector | 14 | 8 | FlagQuantum serial | 15.298 | 1.912 | 522.94 | 1.70x |
| hardware_efficient_statevector | 14 | 8 | Qiskit Aer bridge | 195.718 | 24.465 | 40.88 | 21.78x |
| hardware_efficient_statevector | 14 | 8 | Cirq bridge | 60.255 | 7.532 | 132.77 | 6.70x |
| hardware_efficient_statevector | 14 | 8 | PennyLane Lightning bridge | 25.723 | 3.215 | 311.00 | 2.86x |
| hardware_efficient_statevector | 14 | 32 | FlagQuantum native batch | 33.491 | 1.047 | 955.48 | 1.00x |
| hardware_efficient_statevector | 14 | 32 | FlagQuantum serial | 58.854 | 1.839 | 543.72 | 1.76x |
| hardware_efficient_statevector | 14 | 32 | Qiskit Aer bridge | 810.283 | 25.321 | 39.49 | 24.19x |
| hardware_efficient_statevector | 14 | 32 | Cirq bridge | 249.403 | 7.794 | 128.31 | 7.45x |
| hardware_efficient_statevector | 14 | 32 | PennyLane Lightning bridge | 104.873 | 3.277 | 305.13 | 3.13x |
| hardware_efficient_statevector | 18 | 1 | FlagQuantum native batch | 18.188 | 18.188 | 54.98 | 1.00x |
| hardware_efficient_statevector | 18 | 1 | FlagQuantum serial | 18.387 | 18.387 | 54.39 | 1.01x |
| hardware_efficient_statevector | 18 | 1 | Qiskit Aer bridge | 53.229 | 53.229 | 18.79 | 2.93x |
| hardware_efficient_statevector | 18 | 1 | Cirq bridge | 60.842 | 60.842 | 16.44 | 3.35x |
| hardware_efficient_statevector | 18 | 1 | PennyLane Lightning bridge | 20.991 | 20.991 | 47.64 | 1.15x |
| hardware_efficient_statevector | 18 | 8 | FlagQuantum native batch | 163.809 | 20.476 | 48.84 | 1.00x |
| hardware_efficient_statevector | 18 | 8 | FlagQuantum serial | 147.247 | 18.406 | 54.33 | 0.90x |
| hardware_efficient_statevector | 18 | 8 | Qiskit Aer bridge | 458.964 | 57.371 | 17.43 | 2.80x |
| hardware_efficient_statevector | 18 | 8 | Cirq bridge | 497.618 | 62.202 | 16.08 | 3.04x |
| hardware_efficient_statevector | 18 | 8 | PennyLane Lightning bridge | 176.672 | 22.084 | 45.28 | 1.08x |
| hardware_efficient_statevector | 18 | 32 | FlagQuantum native batch | 719.982 | 22.499 | 44.45 | 1.00x |
| hardware_efficient_statevector | 18 | 32 | FlagQuantum serial | 628.507 | 19.641 | 50.91 | 0.87x |
| hardware_efficient_statevector | 18 | 32 | Qiskit Aer bridge | 1848.244 | 57.758 | 17.31 | 2.57x |
| hardware_efficient_statevector | 18 | 32 | Cirq bridge | 2009.056 | 62.783 | 15.93 | 2.79x |
| hardware_efficient_statevector | 18 | 32 | PennyLane Lightning bridge | 749.049 | 23.408 | 42.72 | 1.04x |
| truncated_qft_statevector | 10 | 1 | FlagQuantum native batch | 0.767 | 0.767 | 1303.57 | 1.00x |
| truncated_qft_statevector | 10 | 1 | FlagQuantum serial | 0.792 | 0.792 | 1263.29 | 1.03x |
| truncated_qft_statevector | 10 | 1 | Qiskit Aer bridge | 23.750 | 23.750 | 42.11 | 30.96x |
| truncated_qft_statevector | 10 | 1 | Cirq bridge | 5.374 | 5.374 | 186.09 | 7.00x |
| truncated_qft_statevector | 10 | 1 | PennyLane Lightning bridge | 3.638 | 3.638 | 274.86 | 4.74x |
| truncated_qft_statevector | 10 | 8 | FlagQuantum native batch | 1.262 | 0.158 | 6340.40 | 1.00x |
| truncated_qft_statevector | 10 | 8 | FlagQuantum serial | 4.351 | 0.544 | 1838.76 | 3.45x |
| truncated_qft_statevector | 10 | 8 | Qiskit Aer bridge | 197.766 | 24.721 | 40.45 | 156.74x |
| truncated_qft_statevector | 10 | 8 | Cirq bridge | 37.688 | 4.711 | 212.27 | 29.87x |
| truncated_qft_statevector | 10 | 8 | PennyLane Lightning bridge | 25.018 | 3.127 | 319.77 | 19.83x |
| truncated_qft_statevector | 10 | 32 | FlagQuantum native batch | 2.300 | 0.072 | 13912.04 | 1.00x |
| truncated_qft_statevector | 10 | 32 | FlagQuantum serial | 18.688 | 0.584 | 1712.33 | 8.12x |
| truncated_qft_statevector | 10 | 32 | Qiskit Aer bridge | 788.488 | 24.640 | 40.58 | 342.80x |
| truncated_qft_statevector | 10 | 32 | Cirq bridge | 155.644 | 4.864 | 205.60 | 67.67x |
| truncated_qft_statevector | 10 | 32 | PennyLane Lightning bridge | 101.814 | 3.182 | 314.30 | 44.26x |
| truncated_qft_statevector | 14 | 1 | FlagQuantum native batch | 1.754 | 1.754 | 570.19 | 1.00x |
| truncated_qft_statevector | 14 | 1 | FlagQuantum serial | 2.363 | 2.363 | 423.16 | 1.35x |
| truncated_qft_statevector | 14 | 1 | Qiskit Aer bridge | 26.116 | 26.116 | 38.29 | 14.89x |
| truncated_qft_statevector | 14 | 1 | Cirq bridge | 8.042 | 8.042 | 124.34 | 4.59x |
| truncated_qft_statevector | 14 | 1 | PennyLane Lightning bridge | 6.151 | 6.151 | 162.57 | 3.51x |
| truncated_qft_statevector | 14 | 8 | FlagQuantum native batch | 8.669 | 1.084 | 922.82 | 1.00x |
| truncated_qft_statevector | 14 | 8 | FlagQuantum serial | 13.801 | 1.725 | 579.69 | 1.59x |
| truncated_qft_statevector | 14 | 8 | Qiskit Aer bridge | 214.536 | 26.817 | 37.29 | 24.75x |
| truncated_qft_statevector | 14 | 8 | Cirq bridge | 63.267 | 7.908 | 126.45 | 7.30x |
| truncated_qft_statevector | 14 | 8 | PennyLane Lightning bridge | 50.133 | 6.267 | 159.57 | 5.78x |
| truncated_qft_statevector | 14 | 32 | FlagQuantum native batch | 37.973 | 1.187 | 842.70 | 1.00x |
| truncated_qft_statevector | 14 | 32 | FlagQuantum serial | 51.590 | 1.612 | 620.27 | 1.36x |
| truncated_qft_statevector | 14 | 32 | Qiskit Aer bridge | 841.558 | 26.299 | 38.02 | 22.16x |
| truncated_qft_statevector | 14 | 32 | Cirq bridge | 249.744 | 7.805 | 128.13 | 6.58x |
| truncated_qft_statevector | 14 | 32 | PennyLane Lightning bridge | 201.255 | 6.289 | 159.00 | 5.30x |
| truncated_qft_statevector | 18 | 1 | FlagQuantum native batch | 19.184 | 19.184 | 52.13 | 1.00x |
| truncated_qft_statevector | 18 | 1 | FlagQuantum serial | 21.531 | 21.531 | 46.44 | 1.12x |
| truncated_qft_statevector | 18 | 1 | Qiskit Aer bridge | 68.116 | 68.116 | 14.68 | 3.55x |
| truncated_qft_statevector | 18 | 1 | Cirq bridge | 27.315 | 27.315 | 36.61 | 1.42x |
| truncated_qft_statevector | 18 | 1 | PennyLane Lightning bridge | 41.541 | 41.541 | 24.07 | 2.17x |
| truncated_qft_statevector | 18 | 8 | FlagQuantum native batch | 203.575 | 25.447 | 39.30 | 1.00x |
| truncated_qft_statevector | 18 | 8 | FlagQuantum serial | 175.996 | 22.000 | 45.46 | 0.86x |
| truncated_qft_statevector | 18 | 8 | Qiskit Aer bridge | 595.405 | 74.426 | 13.44 | 2.92x |
| truncated_qft_statevector | 18 | 8 | Cirq bridge | 254.222 | 31.778 | 31.47 | 1.25x |
| truncated_qft_statevector | 18 | 8 | PennyLane Lightning bridge | 347.693 | 43.462 | 23.01 | 1.71x |
| truncated_qft_statevector | 18 | 32 | FlagQuantum native batch | 789.476 | 24.671 | 40.53 | 1.00x |
| truncated_qft_statevector | 18 | 32 | FlagQuantum serial | 731.053 | 22.845 | 43.77 | 0.93x |
| truncated_qft_statevector | 18 | 32 | Qiskit Aer bridge | 2311.145 | 72.223 | 13.85 | 2.93x |
| truncated_qft_statevector | 18 | 32 | Cirq bridge | 961.626 | 30.051 | 33.28 | 1.22x |
| truncated_qft_statevector | 18 | 32 | PennyLane Lightning bridge | 1393.167 | 43.536 | 22.97 | 1.76x |
| random_clifford_statevector | 10 | 1 | FlagQuantum native batch | 1.554 | 1.554 | 643.66 | 1.00x |
| random_clifford_statevector | 10 | 1 | FlagQuantum serial | 1.179 | 1.179 | 847.85 | 0.76x |
| random_clifford_statevector | 10 | 1 | Qiskit Aer bridge | 23.151 | 23.151 | 43.19 | 14.90x |
| random_clifford_statevector | 10 | 1 | Cirq bridge | 3.613 | 3.613 | 276.78 | 2.33x |
| random_clifford_statevector | 10 | 1 | PennyLane Lightning bridge | 2.208 | 2.208 | 452.93 | 1.42x |
| random_clifford_statevector | 10 | 8 | FlagQuantum native batch | 1.742 | 0.218 | 4592.97 | 1.00x |
| random_clifford_statevector | 10 | 8 | FlagQuantum serial | 7.261 | 0.908 | 1101.71 | 4.17x |
| random_clifford_statevector | 10 | 8 | Qiskit Aer bridge | 195.041 | 24.380 | 41.02 | 111.98x |
| random_clifford_statevector | 10 | 8 | Cirq bridge | 20.336 | 2.542 | 393.38 | 11.68x |
| random_clifford_statevector | 10 | 8 | PennyLane Lightning bridge | 13.644 | 1.706 | 586.33 | 7.83x |
| random_clifford_statevector | 10 | 32 | FlagQuantum native batch | 4.592 | 0.144 | 6968.14 | 1.00x |
| random_clifford_statevector | 10 | 32 | FlagQuantum serial | 29.211 | 0.913 | 1095.47 | 6.36x |
| random_clifford_statevector | 10 | 32 | Qiskit Aer bridge | 750.599 | 23.456 | 42.63 | 163.45x |
| random_clifford_statevector | 10 | 32 | Cirq bridge | 82.343 | 2.573 | 388.62 | 17.93x |
| random_clifford_statevector | 10 | 32 | PennyLane Lightning bridge | 53.294 | 1.665 | 600.45 | 11.60x |
| random_clifford_statevector | 14 | 1 | FlagQuantum native batch | 3.109 | 3.109 | 321.66 | 1.00x |
| random_clifford_statevector | 14 | 1 | FlagQuantum serial | 2.667 | 2.667 | 375.01 | 0.86x |
| random_clifford_statevector | 14 | 1 | Qiskit Aer bridge | 24.429 | 24.429 | 40.94 | 7.86x |
| random_clifford_statevector | 14 | 1 | Cirq bridge | 4.847 | 4.847 | 206.31 | 1.56x |
| random_clifford_statevector | 14 | 1 | PennyLane Lightning bridge | 3.068 | 3.068 | 325.95 | 0.99x |
| random_clifford_statevector | 14 | 8 | FlagQuantum native batch | 13.203 | 1.650 | 605.91 | 1.00x |
| random_clifford_statevector | 14 | 8 | FlagQuantum serial | 19.266 | 2.408 | 415.25 | 1.46x |
| random_clifford_statevector | 14 | 8 | Qiskit Aer bridge | 187.420 | 23.428 | 42.68 | 14.19x |
| random_clifford_statevector | 14 | 8 | Cirq bridge | 36.152 | 4.519 | 221.29 | 2.74x |
| random_clifford_statevector | 14 | 8 | PennyLane Lightning bridge | 19.599 | 2.450 | 408.18 | 1.48x |
| random_clifford_statevector | 14 | 32 | FlagQuantum native batch | 52.560 | 1.642 | 608.83 | 1.00x |
| random_clifford_statevector | 14 | 32 | FlagQuantum serial | 77.098 | 2.409 | 415.06 | 1.47x |
| random_clifford_statevector | 14 | 32 | Qiskit Aer bridge | 775.371 | 24.230 | 41.27 | 14.75x |
| random_clifford_statevector | 14 | 32 | Cirq bridge | 157.358 | 4.917 | 203.36 | 2.99x |
| random_clifford_statevector | 14 | 32 | PennyLane Lightning bridge | 82.546 | 2.580 | 387.66 | 1.57x |
| random_clifford_statevector | 18 | 1 | FlagQuantum native batch | 30.706 | 30.706 | 32.57 | 1.00x |
| random_clifford_statevector | 18 | 1 | FlagQuantum serial | 29.277 | 29.277 | 34.16 | 0.95x |
| random_clifford_statevector | 18 | 1 | Qiskit Aer bridge | 48.505 | 48.505 | 20.62 | 1.58x |
| random_clifford_statevector | 18 | 1 | Cirq bridge | 26.624 | 26.624 | 37.56 | 0.87x |
| random_clifford_statevector | 18 | 1 | PennyLane Lightning bridge | 12.571 | 12.571 | 79.55 | 0.41x |
| random_clifford_statevector | 18 | 8 | FlagQuantum native batch | 258.147 | 32.268 | 30.99 | 1.00x |
| random_clifford_statevector | 18 | 8 | FlagQuantum serial | 231.758 | 28.970 | 34.52 | 0.90x |
| random_clifford_statevector | 18 | 8 | Qiskit Aer bridge | 409.282 | 51.160 | 19.55 | 1.59x |
| random_clifford_statevector | 18 | 8 | Cirq bridge | 236.315 | 29.539 | 33.85 | 0.92x |
| random_clifford_statevector | 18 | 8 | PennyLane Lightning bridge | 106.006 | 13.251 | 75.47 | 0.41x |
| random_clifford_statevector | 18 | 32 | FlagQuantum native batch | 1077.809 | 33.682 | 29.69 | 1.00x |
| random_clifford_statevector | 18 | 32 | FlagQuantum serial | 975.551 | 30.486 | 32.80 | 0.91x |
| random_clifford_statevector | 18 | 32 | Qiskit Aer bridge | 1711.426 | 53.482 | 18.70 | 1.59x |
| random_clifford_statevector | 18 | 32 | Cirq bridge | 974.426 | 30.451 | 32.84 | 0.90x |
| random_clifford_statevector | 18 | 32 | PennyLane Lightning bridge | 433.494 | 13.547 | 73.82 | 0.40x |
| local_brickwork_statevector | 10 | 1 | FlagQuantum native batch | 1.070 | 1.070 | 934.32 | 1.00x |
| local_brickwork_statevector | 10 | 1 | FlagQuantum serial | 1.195 | 1.195 | 836.50 | 1.12x |
| local_brickwork_statevector | 10 | 1 | Qiskit Aer bridge | 22.916 | 22.916 | 43.64 | 21.41x |
| local_brickwork_statevector | 10 | 1 | Cirq bridge | 4.539 | 4.539 | 220.32 | 4.24x |
| local_brickwork_statevector | 10 | 1 | PennyLane Lightning bridge | 2.518 | 2.518 | 397.07 | 2.35x |
| local_brickwork_statevector | 10 | 8 | FlagQuantum native batch | 1.713 | 0.214 | 4669.04 | 1.00x |
| local_brickwork_statevector | 10 | 8 | FlagQuantum serial | 7.493 | 0.937 | 1067.65 | 4.37x |
| local_brickwork_statevector | 10 | 8 | Qiskit Aer bridge | 186.233 | 23.279 | 42.96 | 108.69x |
| local_brickwork_statevector | 10 | 8 | Cirq bridge | 31.292 | 3.912 | 255.65 | 18.26x |
| local_brickwork_statevector | 10 | 8 | PennyLane Lightning bridge | 15.843 | 1.980 | 504.95 | 9.25x |
| local_brickwork_statevector | 10 | 32 | FlagQuantum native batch | 4.240 | 0.132 | 7547.47 | 1.00x |
| local_brickwork_statevector | 10 | 32 | FlagQuantum serial | 29.026 | 0.907 | 1102.45 | 6.85x |
| local_brickwork_statevector | 10 | 32 | Qiskit Aer bridge | 684.391 | 21.387 | 46.76 | 161.42x |
| local_brickwork_statevector | 10 | 32 | Cirq bridge | 118.982 | 3.718 | 268.95 | 28.06x |
| local_brickwork_statevector | 10 | 32 | PennyLane Lightning bridge | 56.314 | 1.760 | 568.24 | 13.28x |
| local_brickwork_statevector | 14 | 1 | FlagQuantum native batch | 3.111 | 3.111 | 321.46 | 1.00x |
| local_brickwork_statevector | 14 | 1 | FlagQuantum serial | 2.630 | 2.630 | 380.19 | 0.85x |
| local_brickwork_statevector | 14 | 1 | Qiskit Aer bridge | 25.545 | 25.545 | 39.15 | 8.21x |
| local_brickwork_statevector | 14 | 1 | Cirq bridge | 8.583 | 8.583 | 116.51 | 2.76x |
| local_brickwork_statevector | 14 | 1 | PennyLane Lightning bridge | 4.005 | 4.005 | 249.70 | 1.29x |
| local_brickwork_statevector | 14 | 8 | FlagQuantum native batch | 15.699 | 1.962 | 509.58 | 1.00x |
| local_brickwork_statevector | 14 | 8 | FlagQuantum serial | 22.232 | 2.779 | 359.85 | 1.42x |
| local_brickwork_statevector | 14 | 8 | Qiskit Aer bridge | 224.035 | 28.004 | 35.71 | 14.27x |
| local_brickwork_statevector | 14 | 8 | Cirq bridge | 72.046 | 9.006 | 111.04 | 4.59x |
| local_brickwork_statevector | 14 | 8 | PennyLane Lightning bridge | 33.242 | 4.155 | 240.66 | 2.12x |
| local_brickwork_statevector | 14 | 32 | FlagQuantum native batch | 59.245 | 1.851 | 540.13 | 1.00x |
| local_brickwork_statevector | 14 | 32 | FlagQuantum serial | 88.396 | 2.762 | 362.01 | 1.49x |
| local_brickwork_statevector | 14 | 32 | Qiskit Aer bridge | 790.250 | 24.695 | 40.49 | 13.34x |
| local_brickwork_statevector | 14 | 32 | Cirq bridge | 271.853 | 8.495 | 117.71 | 4.59x |
| local_brickwork_statevector | 14 | 32 | PennyLane Lightning bridge | 117.131 | 3.660 | 273.20 | 1.98x |
| local_brickwork_statevector | 18 | 1 | FlagQuantum native batch | 25.901 | 25.901 | 38.61 | 1.00x |
| local_brickwork_statevector | 18 | 1 | FlagQuantum serial | 26.106 | 26.106 | 38.31 | 1.01x |
| local_brickwork_statevector | 18 | 1 | Qiskit Aer bridge | 49.428 | 49.428 | 20.23 | 1.91x |
| local_brickwork_statevector | 18 | 1 | Cirq bridge | 57.259 | 57.259 | 17.46 | 2.21x |
| local_brickwork_statevector | 18 | 1 | PennyLane Lightning bridge | 26.890 | 26.890 | 37.19 | 1.04x |
| local_brickwork_statevector | 18 | 8 | FlagQuantum native batch | 235.321 | 29.415 | 34.00 | 1.00x |
| local_brickwork_statevector | 18 | 8 | FlagQuantum serial | 210.049 | 26.256 | 38.09 | 0.89x |
| local_brickwork_statevector | 18 | 8 | Qiskit Aer bridge | 423.212 | 52.901 | 18.90 | 1.80x |
| local_brickwork_statevector | 18 | 8 | Cirq bridge | 478.260 | 59.782 | 16.73 | 2.03x |
| local_brickwork_statevector | 18 | 8 | PennyLane Lightning bridge | 229.695 | 28.712 | 34.83 | 0.98x |
| local_brickwork_statevector | 18 | 32 | FlagQuantum native batch | 922.191 | 28.818 | 34.70 | 1.00x |
| local_brickwork_statevector | 18 | 32 | FlagQuantum serial | 827.830 | 25.870 | 38.66 | 0.90x |
| local_brickwork_statevector | 18 | 32 | Qiskit Aer bridge | 1599.532 | 49.985 | 20.01 | 1.73x |
| local_brickwork_statevector | 18 | 32 | Cirq bridge | 1978.588 | 61.831 | 16.17 | 2.15x |
| local_brickwork_statevector | 18 | 32 | PennyLane Lightning bridge | 865.807 | 27.056 | 36.96 | 0.94x |
| dense_nonlocal_statevector | 10 | 1 | FlagQuantum native batch | 0.523 | 0.523 | 1910.53 | 1.00x |
| dense_nonlocal_statevector | 10 | 1 | FlagQuantum serial | 0.517 | 0.517 | 1932.83 | 0.99x |
| dense_nonlocal_statevector | 10 | 1 | Qiskit Aer bridge | 21.614 | 21.614 | 46.27 | 41.29x |
| dense_nonlocal_statevector | 10 | 1 | Cirq bridge | 2.784 | 2.784 | 359.18 | 5.32x |
| dense_nonlocal_statevector | 10 | 1 | PennyLane Lightning bridge | 2.223 | 2.223 | 449.84 | 4.25x |
| dense_nonlocal_statevector | 10 | 8 | FlagQuantum native batch | 0.743 | 0.093 | 10767.75 | 1.00x |
| dense_nonlocal_statevector | 10 | 8 | FlagQuantum serial | 2.567 | 0.321 | 3115.87 | 3.46x |
| dense_nonlocal_statevector | 10 | 8 | Qiskit Aer bridge | 168.487 | 21.061 | 47.48 | 226.78x |
| dense_nonlocal_statevector | 10 | 8 | Cirq bridge | 21.286 | 2.661 | 375.84 | 28.65x |
| dense_nonlocal_statevector | 10 | 8 | PennyLane Lightning bridge | 16.256 | 2.032 | 492.12 | 21.88x |
| dense_nonlocal_statevector | 10 | 32 | FlagQuantum native batch | 2.080 | 0.065 | 15384.92 | 1.00x |
| dense_nonlocal_statevector | 10 | 32 | FlagQuantum serial | 10.617 | 0.332 | 3013.98 | 5.10x |
| dense_nonlocal_statevector | 10 | 32 | Qiskit Aer bridge | 743.044 | 23.220 | 43.07 | 357.24x |
| dense_nonlocal_statevector | 10 | 32 | Cirq bridge | 103.615 | 3.238 | 308.84 | 49.82x |
| dense_nonlocal_statevector | 10 | 32 | PennyLane Lightning bridge | 68.805 | 2.150 | 465.08 | 33.08x |
| dense_nonlocal_statevector | 14 | 1 | FlagQuantum native batch | 1.013 | 1.013 | 987.17 | 1.00x |
| dense_nonlocal_statevector | 14 | 1 | FlagQuantum serial | 1.050 | 1.050 | 952.27 | 1.04x |
| dense_nonlocal_statevector | 14 | 1 | Qiskit Aer bridge | 23.333 | 23.333 | 42.86 | 23.03x |
| dense_nonlocal_statevector | 14 | 1 | Cirq bridge | 6.815 | 6.815 | 146.73 | 6.73x |
| dense_nonlocal_statevector | 14 | 1 | PennyLane Lightning bridge | 3.844 | 3.844 | 260.11 | 3.80x |
| dense_nonlocal_statevector | 14 | 8 | FlagQuantum native batch | 4.734 | 0.592 | 1689.77 | 1.00x |
| dense_nonlocal_statevector | 14 | 8 | FlagQuantum serial | 7.153 | 0.894 | 1118.45 | 1.51x |
| dense_nonlocal_statevector | 14 | 8 | Qiskit Aer bridge | 183.084 | 22.886 | 43.70 | 38.67x |
| dense_nonlocal_statevector | 14 | 8 | Cirq bridge | 54.148 | 6.768 | 147.74 | 11.44x |
| dense_nonlocal_statevector | 14 | 8 | PennyLane Lightning bridge | 32.892 | 4.112 | 243.22 | 6.95x |
| dense_nonlocal_statevector | 14 | 32 | FlagQuantum native batch | 20.603 | 0.644 | 1553.21 | 1.00x |
| dense_nonlocal_statevector | 14 | 32 | FlagQuantum serial | 27.883 | 0.871 | 1147.64 | 1.35x |
| dense_nonlocal_statevector | 14 | 32 | Qiskit Aer bridge | 870.203 | 27.194 | 36.77 | 42.24x |
| dense_nonlocal_statevector | 14 | 32 | Cirq bridge | 233.887 | 7.309 | 136.82 | 11.35x |
| dense_nonlocal_statevector | 14 | 32 | PennyLane Lightning bridge | 142.752 | 4.461 | 224.17 | 6.93x |
| dense_nonlocal_statevector | 18 | 1 | FlagQuantum native batch | 13.639 | 13.639 | 73.32 | 1.00x |
| dense_nonlocal_statevector | 18 | 1 | FlagQuantum serial | 13.261 | 13.261 | 75.41 | 0.97x |
| dense_nonlocal_statevector | 18 | 1 | Qiskit Aer bridge | 106.884 | 106.884 | 9.36 | 7.84x |
| dense_nonlocal_statevector | 18 | 1 | Cirq bridge | 58.153 | 58.153 | 17.20 | 4.26x |
| dense_nonlocal_statevector | 18 | 1 | PennyLane Lightning bridge | 19.331 | 19.331 | 51.73 | 1.42x |
| dense_nonlocal_statevector | 18 | 8 | FlagQuantum native batch | 87.557 | 10.945 | 91.37 | 1.00x |
| dense_nonlocal_statevector | 18 | 8 | FlagQuantum serial | 82.992 | 10.374 | 96.40 | 0.95x |
| dense_nonlocal_statevector | 18 | 8 | Qiskit Aer bridge | 762.218 | 95.277 | 10.50 | 8.71x |
| dense_nonlocal_statevector | 18 | 8 | Cirq bridge | 393.483 | 49.185 | 20.33 | 4.49x |
| dense_nonlocal_statevector | 18 | 8 | PennyLane Lightning bridge | 130.703 | 16.338 | 61.21 | 1.49x |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum native batch | 369.459 | 11.546 | 86.61 | 1.00x |
| dense_nonlocal_statevector | 18 | 32 | FlagQuantum serial | 328.081 | 10.253 | 97.54 | 0.89x |
| dense_nonlocal_statevector | 18 | 32 | Qiskit Aer bridge | 2921.936 | 91.311 | 10.95 | 7.91x |
| dense_nonlocal_statevector | 18 | 32 | Cirq bridge | 1566.690 | 48.959 | 20.43 | 4.24x |
| dense_nonlocal_statevector | 18 | 32 | PennyLane Lightning bridge | 524.390 | 16.387 | 61.02 | 1.42x |

Ratios above one mean FlagQuantum native batch completed the identical
N-statevector task faster. Results are local comparison evidence, not a
universal framework ranking or release/scalability evidence.
