# CPU training comparison

Producer: `97b315ea02863583e88d13a4127be65133485cf6`.
Protocol SHA-256: `02f15449c3f5aaf8ebb34151aaaadb196ea8cffa01602bd6ee5d73882ba190a0`.
Smoke: `False`. This is CPU comparison evidence, not release or scalability evidence.

Training time includes optimizer updates and per-update validation; imports, oracle checks and replay are outside that timer.
Peak RSS includes imports, oracle checks, warmup and replay. These small, serial-example workloads are not a ranking of the fastest available configurations.

| Task | Engine | Completed | Target reached | Training seconds, median | Final task score, median [min, max] | Peak process RSS MiB, median |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 3 | 3 | 2.6786 | 1 [1, 1] | 298.2 |
| classifier | flagquantum_jax_kernel | 3 | 3 | 0.1442 | 1 [1, 1] | 471.8 |
| classifier | flagquantum_jax_module | 3 | 3 | 0.3151 | 1 [1, 1] | 483.9 |
| classifier | flagquantum_torch_module | 3 | 3 | 2.3687 | 1 [1, 1] | 303.3 |
| classifier | tensorcircuit_jax | 3 | 3 | 0.1489 | 1 [1, 1] | 520.3 |
| heisenberg | flagquantum | 3 | 3 | 0.3169 | 0.000133035 [0.00012318, 0.000192061] | 288.6 |
| heisenberg | flagquantum_jax_kernel | 3 | 3 | 0.0263 | 0.000133035 [0.00012318, 0.000192061] | 533.0 |
| heisenberg | flagquantum_jax_module | 3 | 3 | 0.0351 | 0.000133035 [0.00012318, 0.000192061] | 568.3 |
| heisenberg | flagquantum_torch_module | 3 | 3 | 0.3308 | 0.000133035 [0.00012318, 0.000192061] | 291.9 |
| heisenberg | tensorcircuit_jax | 3 | 3 | 0.0251 | 0.000133035 [0.00012318, 0.000192061] | 571.1 |

Classifier score is held-out test accuracy after fixed-step training. Its time-to-target uses validation accuracy >= 0.8. Heisenberg score is final relative energy error; its target is <= 0.01.
The six-parameter quadratic classical control is well matched to the XOR label rule. Its zero initialization is identical across paired runs, so those runs do not establish seed-dependent model variability.

## Per-seed outcomes

| Task | Engine | Seed | Final score | Training seconds | Warmup seconds | First value/gradient seconds | Time to target seconds | Replay max error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 11 | 1 | 2.6786 | 0.0354 | 0.0138 | 1.0206 | 0 |
| classifier | flagquantum | 29 | 1 | 2.7186 | 0.0363 | 0.0098 | 1.0472 | 0 |
| classifier | flagquantum | 47 | 1 | 2.5791 | 0.0390 | 0.0130 | 0.8969 | 0 |
| classifier | flagquantum_jax_kernel | 11 | 1 | 0.1405 | 0.0023 | 0.1771 | 0.0599 | 0 |
| classifier | flagquantum_jax_kernel | 29 | 1 | 0.1442 | 0.0021 | 0.1931 | 0.0600 | 0 |
| classifier | flagquantum_jax_kernel | 47 | 1 | 0.1477 | 0.0022 | 0.1961 | 0.0528 | 0 |
| classifier | flagquantum_jax_module | 11 | 1 | 0.3151 | 0.0043 | 1.0630 | 0.1289 | 0 |
| classifier | flagquantum_jax_module | 29 | 1 | 0.3160 | 0.0042 | 0.7472 | 0.1270 | 0 |
| classifier | flagquantum_jax_module | 47 | 1 | 0.3075 | 0.0047 | 0.7988 | 0.1132 | 0 |
| classifier | flagquantum_torch_module | 11 | 1 | 2.3687 | 0.0369 | 0.4485 | 0.9310 | 0 |
| classifier | flagquantum_torch_module | 29 | 1 | 2.3412 | 0.0376 | 0.4209 | 0.9278 | 0 |
| classifier | flagquantum_torch_module | 47 | 1 | 2.3932 | 0.0342 | 0.4267 | 0.8221 | 0 |
| classifier | tensorcircuit_jax | 11 | 1 | 0.1489 | 0.0023 | 0.1307 | 0.0612 | 0 |
| classifier | tensorcircuit_jax | 29 | 1 | 0.1405 | 0.0023 | 0.1211 | 0.0532 | 0 |
| classifier | tensorcircuit_jax | 47 | 1 | 0.1506 | 0.0021 | 0.1267 | 0.0545 | 0 |
| heisenberg | flagquantum | 11 | 0.00019206107 | 0.3239 | 0.0041 | 0.0162 | 0.1146 | 0 |
| heisenberg | flagquantum | 29 | 0.00012317999 | 0.3169 | 0.0042 | 0.0117 | 0.0962 | 0 |
| heisenberg | flagquantum | 47 | 0.00013303469 | 0.3147 | 0.0039 | 0.0108 | 0.0521 | 0 |
| heisenberg | flagquantum_jax_kernel | 11 | 0.00019206107 | 0.0266 | 0.0003 | 0.8101 | 0.0093 | 0 |
| heisenberg | flagquantum_jax_kernel | 29 | 0.00012317999 | 0.0234 | 0.0004 | 0.8224 | 0.0081 | 0 |
| heisenberg | flagquantum_jax_kernel | 47 | 0.00013303469 | 0.0263 | 0.0004 | 0.7896 | 0.0055 | 0 |
| heisenberg | flagquantum_jax_module | 11 | 0.00019206107 | 0.0351 | 0.0007 | 1.3470 | 0.0131 | 0 |
| heisenberg | flagquantum_jax_module | 29 | 0.00012317999 | 0.0372 | 0.0012 | 1.1455 | 0.0115 | 0 |
| heisenberg | flagquantum_jax_module | 47 | 0.00013303469 | 0.0342 | 0.0006 | 1.1725 | 0.0063 | 0 |
| heisenberg | flagquantum_torch_module | 11 | 0.00019206107 | 0.3236 | 0.0049 | 0.4232 | 0.1145 | 0 |
| heisenberg | flagquantum_torch_module | 29 | 0.00012317999 | 0.3308 | 0.0047 | 0.4162 | 0.1027 | 0 |
| heisenberg | flagquantum_torch_module | 47 | 0.00013303469 | 0.3325 | 0.0051 | 0.5073 | 0.0575 | 0 |
| heisenberg | tensorcircuit_jax | 11 | 0.00019206107 | 0.0251 | 0.0003 | 0.8531 | 0.0092 | 0 |
| heisenberg | tensorcircuit_jax | 29 | 0.00012317999 | 0.0248 | 0.0004 | 0.8405 | 0.0083 | 0 |
| heisenberg | tensorcircuit_jax | 47 | 0.00013303469 | 0.0256 | 0.0003 | 0.8407 | 0.0050 | 0 |

## Total cost including first call

Measured total includes setup, first call, warmup, initial validation (when recorded), training and final assessment. Training already includes checkpoint writing. Oracle and recovery are excluded from measured total. Process wall includes interpreter startup, imports, oracle and recovery; it is an application diagnostic, not a pure simulator ranking.

| Task | Engine | Seed | Per-step update s, median | Total measured s | Total time to target s | Fresh-process wall s |
| --- | --- | ---: | ---: | ---: | --- | --- |
| classifier | flagquantum | 11 | 0.034660 | 2.774037 | 1.080338 | 5.039989 |
| classifier | flagquantum | 29 | 0.035825 | 2.811489 | 1.104047 | 5.057601 |
| classifier | flagquantum | 47 | 0.033497 | 2.679452 | 0.959160 | 4.983685 |
| classifier | flagquantum_jax_kernel | 11 | 0.001775 | 0.534452 | 0.452227 | 1.839321 |
| classifier | flagquantum_jax_kernel | 29 | 0.001827 | 0.577991 | 0.492018 | 1.915221 |
| classifier | flagquantum_jax_kernel | 47 | 0.001843 | 0.558224 | 0.461503 | 1.818636 |
| classifier | flagquantum_jax_module | 11 | 0.003822 | 1.447435 | 1.256131 | 2.435178 |
| classifier | flagquantum_jax_module | 29 | 0.003837 | 1.132948 | 0.938787 | 2.211877 |
| classifier | flagquantum_jax_module | 47 | 0.003738 | 1.174709 | 0.975272 | 2.168277 |
| classifier | flagquantum_torch_module | 11 | 0.032358 | 2.896417 | 1.429899 | 4.753350 |
| classifier | flagquantum_torch_module | 29 | 0.031880 | 2.839128 | 1.398199 | 4.689303 |
| classifier | flagquantum_torch_module | 47 | 0.032509 | 2.892770 | 1.294019 | 4.746186 |
| classifier | tensorcircuit_jax | 11 | 0.001708 | 1.330503 | 1.240169 | 2.703364 |
| classifier | tensorcircuit_jax | 29 | 0.001595 | 1.057701 | 0.967836 | 2.377736 |
| classifier | tensorcircuit_jax | 47 | 0.001713 | 1.078448 | 0.979733 | 2.416800 |
| heisenberg | flagquantum | 11 | 0.003593 | 0.355695 | 0.141688 | 1.698513 |
| heisenberg | flagquantum | 29 | 0.003516 | 0.340530 | 0.114958 | 1.856018 |
| heisenberg | flagquantum | 47 | 0.003454 | 0.337307 | 0.069978 | 1.776499 |
| heisenberg | flagquantum_jax_kernel | 11 | 0.000102 | 1.180767 | 1.163009 | 2.552529 |
| heisenberg | flagquantum_jax_kernel | 29 | 0.000101 | 1.185606 | 1.169823 | 2.568768 |
| heisenberg | flagquantum_jax_kernel | 47 | 0.000110 | 1.161441 | 1.140162 | 2.588228 |
| heisenberg | flagquantum_jax_module | 11 | 0.000193 | 1.578986 | 1.556217 | 2.530137 |
| heisenberg | flagquantum_jax_module | 29 | 0.000204 | 1.415511 | 1.389009 | 2.412146 |
| heisenberg | flagquantum_jax_module | 47 | 0.000195 | 1.400350 | 1.371735 | 2.338723 |
| heisenberg | flagquantum_torch_module | 11 | 0.003744 | 0.768060 | 0.554389 | 1.705710 |
| heisenberg | flagquantum_torch_module | 29 | 0.003763 | 0.762302 | 0.529683 | 1.749621 |
| heisenberg | flagquantum_torch_module | 47 | 0.003830 | 0.856354 | 0.576698 | 1.867030 |
| heisenberg | tensorcircuit_jax | 11 | 0.000112 | 1.897477 | 1.880995 | 3.281249 |
| heisenberg | tensorcircuit_jax | 29 | 0.000107 | 1.742709 | 1.725496 | 3.093170 |
| heisenberg | tensorcircuit_jax | 47 | 0.000111 | 1.663773 | 1.642496 | 3.039655 |

## Retained failures and limits

No case failed or timed out in this directory.

QNG algorithm/fusion ablation, batched/vmapped throughput, larger-task quality, repeated timing campaigns, GPU runs and AI-assisted error correction are outside this artifact. No quantum advantage or generalization advantage is established.
