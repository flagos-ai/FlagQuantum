# CPU training comparison

Producer: `abde02fd05100b3ae2a65d68128d459cfd437dbe`.
Protocol SHA-256: `02f15449c3f5aaf8ebb34151aaaadb196ea8cffa01602bd6ee5d73882ba190a0`.
Smoke: `False`. This is CPU comparison evidence, not release or scalability evidence.

Training time includes optimizer updates and per-update validation; imports, oracle checks and replay are outside that timer.
Peak RSS includes imports, oracle checks, warmup and replay. These small, serial-example workloads are not a ranking of the fastest available configurations.

| Task | Engine | Completed | Target reached | Training seconds, median | Final task score, median [min, max] | Peak process RSS MiB, median |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 3 | 3 | 2.5645 | 1 [1, 1] | 298.2 |
| classifier | flagquantum_jax_kernel | 3 | 3 | 0.1454 | 1 [1, 1] | 467.0 |
| classifier | flagquantum_jax_module | 3 | 3 | 0.3400 | 1 [1, 1] | 485.1 |
| classifier | flagquantum_torch_module | 3 | 3 | 2.3828 | 1 [1, 1] | 303.4 |
| classifier | tensorcircuit_jax | 3 | 3 | 0.1475 | 1 [1, 1] | 534.0 |
| heisenberg | flagquantum | 3 | 3 | 0.3298 | 0.000133035 [0.00012318, 0.000192061] | 288.6 |
| heisenberg | flagquantum_jax_kernel | 3 | 3 | 0.0247 | 0.000133035 [0.00012318, 0.000192061] | 546.4 |
| heisenberg | flagquantum_jax_module | 3 | 3 | 0.0354 | 0.000133035 [0.00012318, 0.000192061] | 567.7 |
| heisenberg | flagquantum_torch_module | 3 | 3 | 0.3404 | 0.000133035 [0.00012318, 0.000192061] | 292.2 |
| heisenberg | tensorcircuit_jax | 3 | 3 | 0.0264 | 0.000133035 [0.00012318, 0.000192061] | 545.6 |

Classifier score is held-out test accuracy after fixed-step training. Its time-to-target uses validation accuracy >= 0.8. Heisenberg score is final relative energy error; its target is <= 0.01.
The six-parameter quadratic classical control is well matched to the XOR label rule. Its zero initialization is identical across paired runs, so those runs do not establish seed-dependent model variability.

## Per-seed outcomes

| Task | Engine | Seed | Final score | Training seconds | Warmup seconds | First value/gradient seconds | Time to target seconds | Replay max error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 11 | 1 | 2.5907 | 0.0363 | 0.0150 | 1.0347 | 0 |
| classifier | flagquantum | 29 | 1 | 2.5421 | 0.0369 | 0.0098 | 1.0196 | 0 |
| classifier | flagquantum | 47 | 1 | 2.5645 | 0.0368 | 0.0102 | 0.8878 | 0 |
| classifier | flagquantum_jax_kernel | 11 | 1 | 0.1454 | 0.0027 | 0.2162 | 0.0576 | 0 |
| classifier | flagquantum_jax_kernel | 29 | 1 | 0.1449 | 0.0023 | 0.2072 | 0.0603 | 0 |
| classifier | flagquantum_jax_kernel | 47 | 1 | 0.1472 | 0.0024 | 0.1940 | 0.0531 | 0 |
| classifier | flagquantum_jax_module | 11 | 1 | 0.3355 | 0.0049 | 1.0028 | 0.1366 | 0 |
| classifier | flagquantum_jax_module | 29 | 1 | 0.3490 | 0.0048 | 0.9034 | 0.1433 | 0 |
| classifier | flagquantum_jax_module | 47 | 1 | 0.3400 | 0.0047 | 0.7938 | 0.1183 | 0 |
| classifier | flagquantum_torch_module | 11 | 1 | 2.3257 | 0.0321 | 0.4349 | 0.9267 | 0 |
| classifier | flagquantum_torch_module | 29 | 1 | 2.3930 | 0.0353 | 0.4473 | 0.9551 | 0 |
| classifier | flagquantum_torch_module | 47 | 1 | 2.3828 | 0.0359 | 0.4290 | 0.8260 | 0 |
| classifier | tensorcircuit_jax | 11 | 1 | 0.1508 | 0.0022 | 0.1245 | 0.0602 | 0 |
| classifier | tensorcircuit_jax | 29 | 1 | 0.1475 | 0.0020 | 0.1213 | 0.0588 | 0 |
| classifier | tensorcircuit_jax | 47 | 1 | 0.1429 | 0.0022 | 0.1257 | 0.0524 | 0 |
| heisenberg | flagquantum | 11 | 0.00019206107 | 0.3209 | 0.0040 | 0.0123 | 0.1116 | 0 |
| heisenberg | flagquantum | 29 | 0.00012317999 | 0.3298 | 0.0047 | 0.0164 | 0.1046 | 0 |
| heisenberg | flagquantum | 47 | 0.00013303469 | 0.3481 | 0.0045 | 0.0124 | 0.0571 | 0 |
| heisenberg | flagquantum_jax_kernel | 11 | 0.00019206107 | 0.0236 | 0.0005 | 0.8276 | 0.0082 | 0 |
| heisenberg | flagquantum_jax_kernel | 29 | 0.00012317999 | 0.0259 | 0.0004 | 0.8214 | 0.0095 | 0 |
| heisenberg | flagquantum_jax_kernel | 47 | 0.00013303469 | 0.0247 | 0.0004 | 0.8464 | 0.0046 | 0 |
| heisenberg | flagquantum_jax_module | 11 | 0.00019206107 | 0.0351 | 0.0005 | 1.7493 | 0.0124 | 0 |
| heisenberg | flagquantum_jax_module | 29 | 0.00012317999 | 0.0354 | 0.0007 | 1.4242 | 0.0118 | 0 |
| heisenberg | flagquantum_jax_module | 47 | 0.00013303469 | 0.0391 | 0.0006 | 1.4088 | 0.0073 | 0 |
| heisenberg | flagquantum_torch_module | 11 | 0.00019206107 | 0.3333 | 0.0045 | 0.4367 | 0.1190 | 0 |
| heisenberg | flagquantum_torch_module | 29 | 0.00012317999 | 0.3429 | 0.0046 | 0.4262 | 0.1006 | 0 |
| heisenberg | flagquantum_torch_module | 47 | 0.00013303469 | 0.3404 | 0.0053 | 0.6125 | 0.0590 | 0 |
| heisenberg | tensorcircuit_jax | 11 | 0.00019206107 | 0.0264 | 0.0004 | 0.8407 | 0.0097 | 0 |
| heisenberg | tensorcircuit_jax | 29 | 0.00012317999 | 0.0910 | 0.0009 | 1.1072 | 0.0469 | 0 |
| heisenberg | tensorcircuit_jax | 47 | 0.00013303469 | 0.0262 | 0.0005 | 0.9518 | 0.0050 | 0 |

## Total cost including first call

Measured total includes setup, first call, warmup, initial validation (when recorded), training and final assessment. Training already includes checkpoint writing. Oracle and recovery are excluded from measured total. Process wall includes interpreter startup, imports, oracle and recovery; it is an application diagnostic, not a pure simulator ranking.

| Task | Engine | Seed | Per-step update s, median | Total measured s | Total time to target s | Fresh-process wall s |
| --- | --- | ---: | ---: | ---: | --- | --- |
| classifier | flagquantum | 11 | 0.034026 | 2.687572 | 1.095977 | 5.031594 |
| classifier | flagquantum | 29 | 0.033584 | 2.633729 | 1.077286 | 4.984361 |
| classifier | flagquantum | 47 | 0.033716 | 2.655803 | 0.944220 | 4.927202 |
| classifier | flagquantum_jax_kernel | 11 | 0.001841 | 0.580204 | 0.490596 | 1.938416 |
| classifier | flagquantum_jax_kernel | 29 | 0.001845 | 0.558911 | 0.472599 | 1.918930 |
| classifier | flagquantum_jax_kernel | 47 | 0.001882 | 0.561476 | 0.465553 | 1.964889 |
| classifier | flagquantum_jax_module | 11 | 0.004045 | 1.409539 | 1.205254 | 2.431092 |
| classifier | flagquantum_jax_module | 29 | 0.004200 | 1.320940 | 1.109549 | 2.407875 |
| classifier | flagquantum_jax_module | 47 | 0.004075 | 1.202975 | 0.975624 | 2.276291 |
| classifier | flagquantum_torch_module | 11 | 0.031773 | 2.834541 | 1.408078 | 4.682768 |
| classifier | flagquantum_torch_module | 29 | 0.032621 | 2.915439 | 1.448864 | 4.801010 |
| classifier | flagquantum_torch_module | 47 | 0.032419 | 2.886554 | 1.302306 | 4.774268 |
| classifier | tensorcircuit_jax | 11 | 0.001709 | 1.304140 | 1.210908 | 2.714522 |
| classifier | tensorcircuit_jax | 29 | 0.001692 | 1.067132 | 0.975755 | 2.496817 |
| classifier | tensorcircuit_jax | 47 | 0.001647 | 1.064881 | 0.971807 | 2.367502 |
| heisenberg | flagquantum | 11 | 0.003563 | 0.344965 | 0.130769 | 1.700096 |
| heisenberg | flagquantum | 29 | 0.003579 | 0.358761 | 0.128835 | 1.854744 |
| heisenberg | flagquantum | 47 | 0.003830 | 0.373208 | 0.077048 | 1.847713 |
| heisenberg | flagquantum_jax_kernel | 11 | 0.000103 | 1.213669 | 1.197750 | 2.647396 |
| heisenberg | flagquantum_jax_kernel | 29 | 0.000102 | 1.208876 | 1.191835 | 2.705451 |
| heisenberg | flagquantum_jax_kernel | 47 | 0.000103 | 1.223582 | 1.202946 | 2.627019 |
| heisenberg | flagquantum_jax_module | 11 | 0.000198 | 1.974183 | 1.950773 | 2.935332 |
| heisenberg | flagquantum_jax_module | 29 | 0.000201 | 1.657321 | 1.632911 | 2.604392 |
| heisenberg | flagquantum_jax_module | 47 | 0.000216 | 1.634335 | 1.601677 | 2.604029 |
| heisenberg | flagquantum_torch_module | 11 | 0.003844 | 0.788905 | 0.570140 | 1.808797 |
| heisenberg | flagquantum_torch_module | 29 | 0.003954 | 0.784522 | 0.537725 | 1.808791 |
| heisenberg | flagquantum_torch_module | 47 | 0.003954 | 0.969511 | 0.683632 | 1.982683 |
| heisenberg | tensorcircuit_jax | 11 | 0.000117 | 1.912486 | 1.895159 | 3.333713 |
| heisenberg | tensorcircuit_jax | 29 | 0.000403 | 2.098131 | 2.053179 | 4.475544 |
| heisenberg | tensorcircuit_jax | 47 | 0.000114 | 1.932106 | 1.910249 | 3.363426 |

## Retained failures and limits

No case failed or timed out in this directory.

QNG algorithm/fusion ablation, batched/vmapped throughput, larger-task quality, repeated timing campaigns, GPU runs and AI-assisted error correction are outside this artifact. No quantum advantage or generalization advantage is established.
