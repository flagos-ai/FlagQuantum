# CPU training comparison

Producer: `c8a4eb038dbd2c9624a5af3a55d62f197bd07a51`.
Protocol SHA-256: `02f15449c3f5aaf8ebb34151aaaadb196ea8cffa01602bd6ee5d73882ba190a0`.
Smoke: `False`. This is CPU comparison evidence, not release or scalability evidence.

Training time includes optimizer updates and per-update validation; imports, oracle checks and replay are outside that timer.
Peak RSS includes imports, oracle checks, warmup and replay. These small, serial-example workloads are not a ranking of the fastest available configurations.

| Task | Engine | Completed | Target reached | Training seconds, median | Final task score, median [min, max] | Peak process RSS MiB, median |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 3 | 3 | 2.4391 | 1 [1, 1] | 299.7 |
| classifier | flagquantum_jax_kernel | 3 | 3 | 0.2693 | 1 [1, 1] | 451.5 |
| classifier | flagquantum_jax_module | 3 | 3 | 0.6325 | 1 [1, 1] | 459.1 |
| classifier | flagquantum_torch_module | 3 | 3 | 2.4319 | 1 [1, 1] | 306.9 |
| classifier | tensorcircuit_jax | 3 | 3 | 0.1300 | 1 [1, 1] | 533.1 |
| heisenberg | flagquantum | 3 | 3 | 0.2894 | 0.000133035 [0.00012318, 0.000192061] | 289.2 |
| heisenberg | flagquantum_jax_kernel | 3 | 3 | 0.0256 | 0.000133035 [0.00012318, 0.000192061] | 489.3 |
| heisenberg | flagquantum_jax_module | 3 | 3 | 0.0479 | 0.000133035 [0.00012318, 0.000192061] | 503.1 |
| heisenberg | flagquantum_torch_module | 3 | 3 | 0.3091 | 0.000133035 [0.00012318, 0.000192061] | 294.9 |
| heisenberg | tensorcircuit_jax | 3 | 3 | 0.0245 | 0.000133035 [0.00012318, 0.000192061] | 605.7 |

Classifier score is held-out test accuracy after fixed-step training. Its time-to-target uses validation accuracy >= 0.8. Heisenberg score is final relative energy error; its target is <= 0.01.
The six-parameter quadratic classical control is well matched to the XOR label rule. Its zero initialization is identical across paired runs, so those runs do not establish seed-dependent model variability.

## Per-seed outcomes

| Task | Engine | Seed | Final score | Training seconds | Warmup seconds | First value/gradient seconds | Time to target seconds | Replay max error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 11 | 1 | 2.5159 | 0.0328 | 0.0281 | 0.9607 | 0 |
| classifier | flagquantum | 29 | 1 | 2.4391 | 0.0331 | 0.0103 | 0.9688 | 0 |
| classifier | flagquantum | 47 | 1 | 2.4189 | 0.0337 | 0.0109 | 0.8397 | 0 |
| classifier | flagquantum_jax_kernel | 11 | 1 | 0.2693 | 0.0030 | 0.1104 | 0.1307 | 0 |
| classifier | flagquantum_jax_kernel | 29 | 1 | 0.2796 | 0.0030 | 0.1068 | 0.1377 | 0 |
| classifier | flagquantum_jax_kernel | 47 | 1 | 0.2680 | 0.0050 | 0.1196 | 0.0814 | 0 |
| classifier | flagquantum_jax_module | 11 | 1 | 0.6325 | 0.0078 | 0.7981 | 0.2521 | 0 |
| classifier | flagquantum_jax_module | 29 | 1 | 0.6309 | 0.0082 | 0.5758 | 0.2529 | 0 |
| classifier | flagquantum_jax_module | 47 | 1 | 0.6495 | 0.0079 | 0.5723 | 0.2263 | 0 |
| classifier | flagquantum_torch_module | 11 | 1 | 2.4319 | 0.0359 | 0.3694 | 0.9763 | 0 |
| classifier | flagquantum_torch_module | 29 | 1 | 2.4572 | 0.0375 | 0.3551 | 0.9941 | 0 |
| classifier | flagquantum_torch_module | 47 | 1 | 2.4164 | 0.0357 | 0.3506 | 0.8425 | 0 |
| classifier | tensorcircuit_jax | 11 | 1 | 0.1295 | 0.0020 | 0.1106 | 0.0526 | 0 |
| classifier | tensorcircuit_jax | 29 | 1 | 0.1300 | 0.0018 | 0.1083 | 0.0530 | 0 |
| classifier | tensorcircuit_jax | 47 | 1 | 0.1313 | 0.0018 | 0.1180 | 0.0465 | 0 |
| heisenberg | flagquantum | 11 | 0.00019206107 | 0.2881 | 0.0038 | 0.0162 | 0.1028 | 0 |
| heisenberg | flagquantum | 29 | 0.00012317999 | 0.2966 | 0.0037 | 0.0117 | 0.0948 | 0 |
| heisenberg | flagquantum | 47 | 0.00013303469 | 0.2894 | 0.0037 | 0.0119 | 0.0495 | 0 |
| heisenberg | flagquantum_jax_kernel | 11 | 0.00019206107 | 0.0302 | 0.0003 | 0.4411 | 0.0141 | 0 |
| heisenberg | flagquantum_jax_kernel | 29 | 0.00012317999 | 0.0255 | 0.0003 | 0.4386 | 0.0079 | 0 |
| heisenberg | flagquantum_jax_kernel | 47 | 0.00013303469 | 0.0256 | 0.0003 | 0.4654 | 0.0049 | 0 |
| heisenberg | flagquantum_jax_module | 11 | 0.00019206107 | 0.0482 | 0.0007 | 0.9601 | 0.0174 | 0 |
| heisenberg | flagquantum_jax_module | 29 | 0.00012317999 | 0.0479 | 0.0007 | 0.9317 | 0.0145 | 0 |
| heisenberg | flagquantum_jax_module | 47 | 0.00013303469 | 0.0473 | 0.0008 | 0.9367 | 0.0080 | 0 |
| heisenberg | flagquantum_torch_module | 11 | 0.00019206107 | 0.3075 | 0.0040 | 0.3667 | 0.1091 | 0 |
| heisenberg | flagquantum_torch_module | 29 | 0.00012317999 | 0.3221 | 0.0045 | 0.3625 | 0.1000 | 0 |
| heisenberg | flagquantum_torch_module | 47 | 0.00013303469 | 0.3091 | 0.0042 | 0.3674 | 0.0525 | 0 |
| heisenberg | tensorcircuit_jax | 11 | 0.00019206107 | 0.0244 | 0.0003 | 0.7506 | 0.0091 | 0 |
| heisenberg | tensorcircuit_jax | 29 | 0.00012317999 | 0.0245 | 0.0003 | 0.7466 | 0.0078 | 0 |
| heisenberg | tensorcircuit_jax | 47 | 0.00013303469 | 0.0249 | 0.0004 | 0.7230 | 0.0048 | 0 |

## Retained failures and limits

No case failed or timed out in this directory.

QNG algorithm/fusion ablation, batched/vmapped throughput, larger-task quality, repeated timing campaigns, GPU runs and AI-assisted error correction are outside this artifact. No quantum advantage or generalization advantage is established.
