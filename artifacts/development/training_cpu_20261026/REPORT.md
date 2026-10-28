# CPU training comparison

Producer: `a6c634e9317d7f08cfa4053c6fbd712dabfdd99f`.
Protocol SHA-256: `5750e9cc054737447b2c16ea3b9dfa79934248294f5e058686c27b5c082e8d5e`.
Smoke: `False`. This is CPU comparison evidence, not release or scalability evidence.

Training time includes optimizer updates and per-update validation; imports, oracle checks and replay are outside that timer.
Peak RSS includes imports, oracle checks, warmup and replay. These small, serial-example workloads are not a ranking of the fastest available configurations.

| Task | Engine | Completed | Target reached | Training seconds, median | Final task score, median [min, max] | Peak process RSS MiB, median |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| classifier | classical | 3 | 3 | 0.0047 | 0.90625 [0.90625, 0.90625] | 282.5 |
| classifier | flagquantum | 3 | 3 | 2.4000 | 1 [1, 1] | 297.1 |
| classifier | pennylane_adjoint | 3 | 3 | 2.6896 | 1 [1, 1] | 461.0 |
| classifier | pennylane_backprop | 3 | 3 | 2.9622 | 1 [1, 1] | 466.2 |
| classifier | tensorcircuit_jax | 3 | 3 | 0.1360 | 1 [1, 1] | 529.5 |
| classifier | tensorcircuit_torch | 3 | 3 | 3.3753 | 1 [1, 1] | 451.8 |
| heisenberg | flagquantum | 3 | 3 | 0.2927 | 0.000133035 [0.00012318, 0.000192061] | 287.5 |
| heisenberg | pennylane_adjoint | 3 | 3 | 0.1462 | 0.000133035 [0.00012318, 0.000192061] | 457.7 |
| heisenberg | pennylane_backprop | 3 | 3 | 0.4058 | 0.000133035 [0.00012318, 0.000192061] | 458.4 |
| heisenberg | tensorcircuit_jax | 3 | 3 | 0.0244 | 0.000133035 [0.00012318, 0.000192061] | 597.4 |
| heisenberg | tensorcircuit_torch | 3 | 3 | 0.5023 | 0.000133035 [0.00012318, 0.000192061] | 444.5 |

Classifier score is held-out test accuracy after fixed-step training. Its time-to-target uses validation accuracy >= 0.8. Heisenberg score is final relative energy error; its target is <= 0.01.
The six-parameter quadratic classical control is well matched to the XOR label rule. Its zero initialization is identical across paired runs, so those runs do not establish seed-dependent model variability.

## Per-seed outcomes

| Task | Engine | Seed | Final score | Training seconds | Warmup seconds | First value/gradient seconds | Time to target seconds | Replay max error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| classifier | classical | 11 | 0.90625 | 0.0049 | 0.0001 | 0.0048 | 0.0020 | 0 |
| classifier | classical | 29 | 0.90625 | 0.0047 | 0.0001 | 0.0048 | 0.0020 | 0 |
| classifier | classical | 47 | 0.90625 | 0.0046 | 0.0001 | 0.0048 | 0.0020 | 0 |
| classifier | flagquantum | 11 | 1 | 2.3923 | 0.0342 | 0.0086 | 0.9440 | 0 |
| classifier | flagquantum | 29 | 1 | 2.4000 | 0.0340 | 0.0086 | 0.9645 | 0 |
| classifier | flagquantum | 47 | 1 | 2.4000 | 0.0376 | 0.0092 | 0.8395 | 0 |
| classifier | pennylane_adjoint | 11 | 1 | 2.7246 | 0.0327 | 0.0102 | 1.1009 | 0 |
| classifier | pennylane_adjoint | 29 | 1 | 2.6875 | 0.0334 | 0.0105 | 1.0671 | 0 |
| classifier | pennylane_adjoint | 47 | 1 | 2.6896 | 0.0315 | 0.0097 | 0.9441 | 0 |
| classifier | pennylane_backprop | 11 | 1 | 2.9622 | 0.0392 | 0.0097 | 1.1910 | 0 |
| classifier | pennylane_backprop | 29 | 1 | 2.9594 | 0.0392 | 0.0096 | 1.1756 | 0 |
| classifier | pennylane_backprop | 47 | 1 | 3.1544 | 0.0387 | 0.0099 | 1.2361 | 0 |
| classifier | tensorcircuit_jax | 11 | 1 | 0.1371 | 0.0020 | 0.1119 | 0.0561 | 0 |
| classifier | tensorcircuit_jax | 29 | 1 | 0.1316 | 0.0020 | 0.1099 | 0.0527 | 0 |
| classifier | tensorcircuit_jax | 47 | 1 | 0.1360 | 0.0018 | 0.1086 | 0.0489 | 0 |
| classifier | tensorcircuit_torch | 11 | 1 | 3.3660 | 0.0443 | 0.0249 | 1.3585 | 0 |
| classifier | tensorcircuit_torch | 29 | 1 | 3.3753 | 0.0424 | 0.0099 | 1.3822 | 0 |
| classifier | tensorcircuit_torch | 47 | 1 | 3.4655 | 0.0437 | 0.0100 | 1.2409 | 0 |
| heisenberg | flagquantum | 11 | 0.00019206107 | 0.2927 | 0.0041 | 0.0146 | 0.1050 | 0 |
| heisenberg | flagquantum | 29 | 0.00012317999 | 0.2917 | 0.0039 | 0.0101 | 0.0898 | 0 |
| heisenberg | flagquantum | 47 | 0.00013303469 | 0.3001 | 0.0041 | 0.0099 | 0.0517 | 0 |
| heisenberg | pennylane_adjoint | 11 | 0.00019206107 | 0.1402 | 0.0017 | 0.0093 | 0.0497 | 0 |
| heisenberg | pennylane_adjoint | 29 | 0.00012317999 | 0.1546 | 0.0020 | 0.0098 | 0.0473 | 0 |
| heisenberg | pennylane_adjoint | 47 | 0.00013303469 | 0.1462 | 0.0021 | 0.0096 | 0.0290 | 0 |
| heisenberg | pennylane_backprop | 11 | 0.00019206107 | 0.4058 | 0.0048 | 0.0122 | 0.1426 | 0 |
| heisenberg | pennylane_backprop | 29 | 0.00012317999 | 0.4029 | 0.0047 | 0.0125 | 0.1248 | 0 |
| heisenberg | pennylane_backprop | 47 | 0.00013303469 | 0.4087 | 0.0049 | 0.0126 | 0.0706 | 0 |
| heisenberg | tensorcircuit_jax | 11 | 0.00019206107 | 0.0243 | 0.0004 | 0.7741 | 0.0088 | 0 |
| heisenberg | tensorcircuit_jax | 29 | 0.00012317999 | 0.0253 | 0.0003 | 0.7348 | 0.0082 | 0 |
| heisenberg | tensorcircuit_jax | 47 | 0.00013303469 | 0.0244 | 0.0003 | 0.7403 | 0.0047 | 0 |
| heisenberg | tensorcircuit_torch | 11 | 0.00019206107 | 0.5023 | 0.0050 | 0.0115 | 0.1738 | 0 |
| heisenberg | tensorcircuit_torch | 29 | 0.00012317999 | 0.5015 | 0.0052 | 0.0118 | 0.1520 | 0 |
| heisenberg | tensorcircuit_torch | 47 | 0.00013303469 | 0.5083 | 0.0052 | 0.0116 | 0.0829 | 0 |

## Retained failures and limits

No case failed or timed out in this directory.

QNG algorithm/fusion ablation, batched/vmapped throughput, larger-task quality, repeated timing campaigns, GPU runs and AI-assisted error correction are outside this artifact. No quantum advantage or generalization advantage is established.
