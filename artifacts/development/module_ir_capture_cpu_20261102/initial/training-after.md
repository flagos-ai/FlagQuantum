# CPU training comparison

Producer: `053adcfbae359c8204c3617d9e1dbf8ac524e248`.
Protocol SHA-256: `02f15449c3f5aaf8ebb34151aaaadb196ea8cffa01602bd6ee5d73882ba190a0`.
Smoke: `False`. This is CPU comparison evidence, not release or scalability evidence.

Training time includes optimizer updates and per-update validation; imports, oracle checks and replay are outside that timer.
Peak RSS includes imports, oracle checks, warmup and replay. These small, serial-example workloads are not a ranking of the fastest available configurations.

| Task | Engine | Completed | Target reached | Training seconds, median | Final task score, median [min, max] | Peak process RSS MiB, median |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 3 | 3 | 2.6404 | 1 [1, 1] | 298.2 |
| classifier | flagquantum_jax_kernel | 3 | 3 | 0.1469 | 1 [1, 1] | 461.6 |
| classifier | flagquantum_jax_module | 3 | 3 | 0.3519 | 1 [1, 1] | 480.2 |
| classifier | flagquantum_torch_module | 3 | 3 | 2.3892 | 1 [1, 1] | 303.3 |
| classifier | tensorcircuit_jax | 3 | 3 | 0.1424 | 1 [1, 1] | 514.0 |
| heisenberg | flagquantum | 3 | 3 | 0.3252 | 0.000133035 [0.00012318, 0.000192061] | 288.7 |
| heisenberg | flagquantum_jax_kernel | 3 | 3 | 0.0239 | 0.000133035 [0.00012318, 0.000192061] | 532.0 |
| heisenberg | flagquantum_jax_module | 3 | 3 | 0.0391 | 0.000133035 [0.00012318, 0.000192061] | 556.0 |
| heisenberg | flagquantum_torch_module | 3 | 3 | 0.3415 | 0.000133035 [0.00012318, 0.000192061] | 292.1 |
| heisenberg | tensorcircuit_jax | 3 | 3 | 0.0277 | 0.000133035 [0.00012318, 0.000192061] | 547.3 |

Classifier score is held-out test accuracy after fixed-step training. Its time-to-target uses validation accuracy >= 0.8. Heisenberg score is final relative energy error; its target is <= 0.01.
The six-parameter quadratic classical control is well matched to the XOR label rule. Its zero initialization is identical across paired runs, so those runs do not establish seed-dependent model variability.

## Per-seed outcomes

| Task | Engine | Seed | Final score | Training seconds | Warmup seconds | First value/gradient seconds | Time to target seconds | Replay max error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 11 | 1 | 2.5953 | 0.0359 | 0.0142 | 1.0320 | 0 |
| classifier | flagquantum | 29 | 1 | 2.6572 | 0.0379 | 0.0097 | 1.0611 | 0 |
| classifier | flagquantum | 47 | 1 | 2.6404 | 0.0377 | 0.0190 | 0.9164 | 0 |
| classifier | flagquantum_jax_kernel | 11 | 1 | 0.1525 | 0.0023 | 0.1859 | 0.0615 | 0 |
| classifier | flagquantum_jax_kernel | 29 | 1 | 0.1435 | 0.0022 | 0.1983 | 0.0586 | 0 |
| classifier | flagquantum_jax_kernel | 47 | 1 | 0.1469 | 0.0023 | 0.2024 | 0.0534 | 0 |
| classifier | flagquantum_jax_module | 11 | 1 | 0.3515 | 0.0046 | 1.0558 | 0.1390 | 0 |
| classifier | flagquantum_jax_module | 29 | 1 | 0.3633 | 0.0052 | 0.7259 | 0.1371 | 0 |
| classifier | flagquantum_jax_module | 47 | 1 | 0.3519 | 0.0049 | 0.9925 | 0.1279 | 0 |
| classifier | flagquantum_torch_module | 11 | 1 | 2.4271 | 0.0341 | 0.4301 | 0.9368 | 0 |
| classifier | flagquantum_torch_module | 29 | 1 | 2.3476 | 0.0346 | 0.4163 | 0.9278 | 0 |
| classifier | flagquantum_torch_module | 47 | 1 | 2.3892 | 0.0347 | 0.4152 | 0.8112 | 0 |
| classifier | tensorcircuit_jax | 11 | 1 | 0.1424 | 0.0020 | 0.1317 | 0.0576 | 0 |
| classifier | tensorcircuit_jax | 29 | 1 | 0.1437 | 0.0021 | 0.1190 | 0.0570 | 0 |
| classifier | tensorcircuit_jax | 47 | 1 | 0.1371 | 0.0021 | 0.1203 | 0.0504 | 0 |
| heisenberg | flagquantum | 11 | 0.00019206107 | 0.3221 | 0.0041 | 0.0151 | 0.1162 | 0 |
| heisenberg | flagquantum | 29 | 0.00012317999 | 0.3686 | 0.0041 | 0.0111 | 0.1103 | 0 |
| heisenberg | flagquantum | 47 | 0.00013303469 | 0.3252 | 0.0039 | 0.0116 | 0.0562 | 0 |
| heisenberg | flagquantum_jax_kernel | 11 | 0.00019206107 | 0.0238 | 0.0004 | 0.8031 | 0.0085 | 0 |
| heisenberg | flagquantum_jax_kernel | 29 | 0.00012317999 | 0.0239 | 0.0003 | 0.7438 | 0.0071 | 0 |
| heisenberg | flagquantum_jax_kernel | 47 | 0.00013303469 | 0.0239 | 0.0004 | 0.8451 | 0.0044 | 0 |
| heisenberg | flagquantum_jax_module | 11 | 0.00019206107 | 0.0389 | 0.0007 | 1.1806 | 0.0152 | 0 |
| heisenberg | flagquantum_jax_module | 29 | 0.00012317999 | 0.0416 | 0.0009 | 1.1245 | 0.0148 | 0 |
| heisenberg | flagquantum_jax_module | 47 | 0.00013303469 | 0.0391 | 0.0009 | 1.1207 | 0.0079 | 0 |
| heisenberg | flagquantum_torch_module | 11 | 0.00019206107 | 0.3415 | 0.0047 | 0.4179 | 0.1190 | 0 |
| heisenberg | flagquantum_torch_module | 29 | 0.00012317999 | 0.3522 | 0.0047 | 0.5269 | 0.1060 | 0 |
| heisenberg | flagquantum_torch_module | 47 | 0.00013303469 | 0.3236 | 0.0046 | 0.4247 | 0.0568 | 0 |
| heisenberg | tensorcircuit_jax | 11 | 0.00019206107 | 0.0277 | 0.0005 | 0.8855 | 0.0103 | 0 |
| heisenberg | tensorcircuit_jax | 29 | 0.00012317999 | 0.0267 | 0.0003 | 0.8374 | 0.0087 | 0 |
| heisenberg | tensorcircuit_jax | 47 | 0.00013303469 | 0.0293 | 0.0005 | 0.8789 | 0.0059 | 0 |

## Total cost including first call

Measured total includes setup, first call, warmup, initial validation (when recorded), training and final assessment. Training already includes checkpoint writing. Oracle and recovery are excluded from measured total. Process wall includes interpreter startup, imports, oracle and recovery; it is an application diagnostic, not a pure simulator ranking.

| Task | Engine | Seed | Per-step update s, median | Total measured s | Total time to target s | Fresh-process wall s |
| --- | --- | ---: | ---: | ---: | --- | --- |
| classifier | flagquantum | 11 | 0.033809 | 2.691364 | 1.093171 | 4.930963 |
| classifier | flagquantum | 29 | 0.034840 | 2.749715 | 1.118657 | 4.968021 |
| classifier | flagquantum | 47 | 0.034534 | 2.742710 | 0.982525 | 4.985579 |
| classifier | flagquantum_jax_kernel | 11 | 0.001936 | 0.550352 | 0.457376 | 1.953419 |
| classifier | flagquantum_jax_kernel | 29 | 0.001827 | 0.564660 | 0.477918 | 1.890981 |
| classifier | flagquantum_jax_kernel | 47 | 0.001864 | 0.568685 | 0.473425 | 1.883563 |
| classifier | flagquantum_jax_module | 11 | 0.004175 | 1.480612 | 1.262170 | 2.514464 |
| classifier | flagquantum_jax_module | 29 | 0.004334 | 1.161173 | 0.928777 | 2.274902 |
| classifier | flagquantum_jax_module | 47 | 0.004117 | 1.416545 | 1.186900 | 2.437607 |
| classifier | flagquantum_torch_module | 11 | 0.032986 | 2.931108 | 1.413806 | 4.702013 |
| classifier | flagquantum_torch_module | 29 | 0.031967 | 2.836991 | 1.389752 | 4.645519 |
| classifier | flagquantum_torch_module | 47 | 0.032638 | 2.879348 | 1.272048 | 4.701363 |
| classifier | tensorcircuit_jax | 11 | 0.001631 | 1.935678 | 1.848130 | 3.293922 |
| classifier | tensorcircuit_jax | 29 | 0.001633 | 1.028794 | 0.939432 | 2.454180 |
| classifier | tensorcircuit_jax | 47 | 0.001581 | 1.091606 | 1.002249 | 2.442131 |
| heisenberg | flagquantum | 11 | 0.003538 | 0.349365 | 0.138895 | 1.696322 |
| heisenberg | flagquantum | 29 | 0.003916 | 0.391811 | 0.128502 | 1.787650 |
| heisenberg | flagquantum | 47 | 0.003561 | 0.348330 | 0.074592 | 1.720322 |
| heisenberg | flagquantum_jax_kernel | 11 | 0.000104 | 1.182909 | 1.167035 | 2.532612 |
| heisenberg | flagquantum_jax_kernel | 29 | 0.000101 | 1.108618 | 1.091361 | 2.527132 |
| heisenberg | flagquantum_jax_kernel | 47 | 0.000103 | 1.227545 | 1.207467 | 2.623137 |
| heisenberg | flagquantum_jax_module | 11 | 0.000213 | 1.424861 | 1.400455 | 2.508654 |
| heisenberg | flagquantum_jax_module | 29 | 0.000215 | 1.368119 | 1.340592 | 2.348765 |
| heisenberg | flagquantum_jax_module | 47 | 0.000219 | 1.371528 | 1.339552 | 2.404328 |
| heisenberg | flagquantum_torch_module | 11 | 0.003905 | 0.780371 | 0.552853 | 1.834511 |
| heisenberg | flagquantum_torch_module | 29 | 0.003959 | 0.895289 | 0.644393 | 1.979115 |
| heisenberg | flagquantum_torch_module | 47 | 0.003734 | 0.763504 | 0.492351 | 1.702628 |
| heisenberg | tensorcircuit_jax | 11 | 0.000116 | 1.937140 | 1.919158 | 3.369900 |
| heisenberg | tensorcircuit_jax | 29 | 0.000116 | 1.651027 | 1.632434 | 3.011479 |
| heisenberg | tensorcircuit_jax | 47 | 0.000125 | 1.688131 | 1.664102 | 3.199133 |

## Retained failures and limits

No case failed or timed out in this directory.

QNG algorithm/fusion ablation, batched/vmapped throughput, larger-task quality, repeated timing campaigns, GPU runs and AI-assisted error correction are outside this artifact. No quantum advantage or generalization advantage is established.
