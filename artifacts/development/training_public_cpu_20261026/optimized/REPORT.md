# CPU training comparison

Producer: `c06cabd1ef3edb4453433e31f93e67de4af65945`.
Protocol SHA-256: `02f15449c3f5aaf8ebb34151aaaadb196ea8cffa01602bd6ee5d73882ba190a0`.
Smoke: `False`. This is CPU comparison evidence, not release or scalability evidence.

Training time includes optimizer updates and per-update validation; imports, oracle checks and replay are outside that timer.
Peak RSS includes imports, oracle checks, warmup and replay. These small, serial-example workloads are not a ranking of the fastest available configurations.

| Task | Engine | Completed | Target reached | Training seconds, median | Final task score, median [min, max] | Peak process RSS MiB, median |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 3 | 3 | 2.6349 | 1 [1, 1] | 297.7 |
| classifier | flagquantum_jax_kernel | 3 | 3 | 0.1528 | 1 [1, 1] | 433.8 |
| classifier | flagquantum_jax_module | 3 | 3 | 0.1989 | 1 [1, 1] | 444.8 |
| classifier | flagquantum_torch_module | 3 | 3 | 2.6675 | 1 [1, 1] | 305.0 |
| classifier | tensorcircuit_jax | 3 | 3 | 0.1524 | 1 [1, 1] | 525.6 |
| heisenberg | flagquantum | 3 | 3 | 0.3520 | 0.000133035 [0.00012318, 0.000192061] | 287.3 |
| heisenberg | flagquantum_jax_kernel | 3 | 3 | 0.0247 | 0.000133035 [0.00012318, 0.000192061] | 491.4 |
| heisenberg | flagquantum_jax_module | 3 | 3 | 0.0324 | 0.000133035 [0.00012318, 0.000192061] | 503.0 |
| heisenberg | flagquantum_torch_module | 3 | 3 | 0.3622 | 0.000133035 [0.00012318, 0.000192061] | 283.6 |
| heisenberg | tensorcircuit_jax | 3 | 3 | 0.0251 | 0.000133035 [0.00012318, 0.000192061] | 581.2 |

Classifier score is held-out test accuracy after fixed-step training. Its time-to-target uses validation accuracy >= 0.8. Heisenberg score is final relative energy error; its target is <= 0.01.
The six-parameter quadratic classical control is well matched to the XOR label rule. Its zero initialization is identical across paired runs, so those runs do not establish seed-dependent model variability.

## Per-seed outcomes

| Task | Engine | Seed | Final score | Training seconds | Warmup seconds | First value/gradient seconds | Time to target seconds | Replay max error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 11 | 1 | 2.6856 | 0.0352 | 0.0265 | 1.2159 | 0 |
| classifier | flagquantum | 29 | 1 | 2.4280 | 0.0336 | 0.0093 | 0.9723 | 0 |
| classifier | flagquantum | 47 | 1 | 2.6349 | 0.0354 | 0.0094 | 0.9352 | 0 |
| classifier | flagquantum_jax_kernel | 11 | 1 | 0.1560 | 0.0023 | 0.1147 | 0.0609 | 0 |
| classifier | flagquantum_jax_kernel | 29 | 1 | 0.1528 | 0.0022 | 0.1160 | 0.0611 | 0 |
| classifier | flagquantum_jax_kernel | 47 | 1 | 0.1505 | 0.0020 | 0.1072 | 0.0528 | 0 |
| classifier | flagquantum_jax_module | 11 | 1 | 0.1989 | 0.0033 | 0.8980 | 0.0821 | 0 |
| classifier | flagquantum_jax_module | 29 | 1 | 0.1980 | 0.0026 | 0.6154 | 0.0790 | 0 |
| classifier | flagquantum_jax_module | 47 | 1 | 0.2083 | 0.0030 | 0.6423 | 0.0769 | 0 |
| classifier | flagquantum_torch_module | 11 | 1 | 2.6675 | 0.0371 | 0.4107 | 1.0549 | 0 |
| classifier | flagquantum_torch_module | 29 | 1 | 3.1648 | 0.0403 | 0.4462 | 1.1523 | 0 |
| classifier | flagquantum_torch_module | 47 | 1 | 2.6357 | 0.0388 | 0.4119 | 0.9176 | 0 |
| classifier | tensorcircuit_jax | 11 | 1 | 0.1337 | 0.0018 | 0.1225 | 0.0539 | 0 |
| classifier | tensorcircuit_jax | 29 | 1 | 0.1524 | 0.0020 | 0.1356 | 0.0638 | 0 |
| classifier | tensorcircuit_jax | 47 | 1 | 0.1632 | 0.0019 | 0.1324 | 0.0592 | 0 |
| heisenberg | flagquantum | 11 | 0.00019206107 | 0.3520 | 0.0044 | 0.0205 | 0.1258 | 0 |
| heisenberg | flagquantum | 29 | 0.00012317999 | 0.7378 | 0.0043 | 0.0144 | 0.1296 | 0 |
| heisenberg | flagquantum | 47 | 0.00013303469 | 0.3130 | 0.0037 | 0.0111 | 0.0568 | 0 |
| heisenberg | flagquantum_jax_kernel | 11 | 0.00019206107 | 0.0254 | 0.0003 | 0.4638 | 0.0097 | 0 |
| heisenberg | flagquantum_jax_kernel | 29 | 0.00012317999 | 0.0240 | 0.0003 | 0.4618 | 0.0075 | 0 |
| heisenberg | flagquantum_jax_kernel | 47 | 0.00013303469 | 0.0247 | 0.0003 | 0.4619 | 0.0050 | 0 |
| heisenberg | flagquantum_jax_module | 11 | 0.00019206107 | 0.0383 | 0.0005 | 1.3931 | 0.0140 | 0 |
| heisenberg | flagquantum_jax_module | 29 | 0.00012317999 | 0.0324 | 0.0004 | 1.0367 | 0.0083 | 0 |
| heisenberg | flagquantum_jax_module | 47 | 0.00013303469 | 0.0246 | 0.0004 | 1.0053 | 0.0042 | 0 |
| heisenberg | flagquantum_torch_module | 11 | 0.00019206107 | 0.3622 | 0.0046 | 0.4672 | 0.1236 | 0 |
| heisenberg | flagquantum_torch_module | 29 | 0.00012317999 | 0.4465 | 0.0047 | 0.5766 | 0.1255 | 0 |
| heisenberg | flagquantum_torch_module | 47 | 0.00013303469 | 0.3466 | 0.0049 | 0.4864 | 0.0581 | 0 |
| heisenberg | tensorcircuit_jax | 11 | 0.00019206107 | 0.0249 | 0.0003 | 0.8314 | 0.0093 | 0 |
| heisenberg | tensorcircuit_jax | 29 | 0.00012317999 | 0.0276 | 0.0004 | 1.0046 | 0.0095 | 0 |
| heisenberg | tensorcircuit_jax | 47 | 0.00013303469 | 0.0251 | 0.0003 | 0.7789 | 0.0053 | 0 |

## Retained failures and limits

No case failed or timed out in this directory.

QNG algorithm/fusion ablation, batched/vmapped throughput, larger-task quality, repeated timing campaigns, GPU runs and AI-assisted error correction are outside this artifact. No quantum advantage or generalization advantage is established.
