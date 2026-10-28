# CPU training comparison

Producer: `3b18d75fc9e11464575249a65ee5708392b9866c`.
Protocol SHA-256: `258f5736f36988f3c3533ecad36d3498163a809d5c88eba12cfb2ea66eab01f7`.
Smoke: `False`. This is CPU comparison evidence, not release or scalability evidence.

Training time includes optimizer updates and per-update validation; imports, oracle checks and replay are outside that timer.
Peak RSS includes imports, oracle checks, warmup and replay. These small, serial-example workloads are not a ranking of the fastest available configurations.

| Task | Engine | Completed | Target reached | Training seconds, median | Final task score, median [min, max] | Peak process RSS MiB, median |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 3 | 3 | 2.5200 | 1 [1, 1] | 300.5 |
| classifier | flagquantum_jax_kernel | 3 | 3 | 0.6820 | 1 [1, 1] | 448.4 |
| classifier | flagquantum_jax_module | 3 | 3 | 1.4463 | 1 [1, 1] | 462.5 |
| classifier | tensorcircuit_jax | 3 | 3 | 0.1332 | 1 [1, 1] | 533.7 |
| heisenberg | flagquantum | 3 | 3 | 0.2969 | 0.000133035 [0.00012318, 0.000192061] | 289.6 |
| heisenberg | flagquantum_jax_kernel | 3 | 3 | 0.0339 | 0.000133035 [0.00012318, 0.000192061] | 495.2 |
| heisenberg | flagquantum_jax_module | 3 | 3 | 0.0720 | 0.000133035 [0.00012318, 0.000192061] | 501.3 |
| heisenberg | tensorcircuit_jax | 3 | 3 | 0.0261 | 0.000133035 [0.00012318, 0.000192061] | 592.5 |

Classifier score is held-out test accuracy after fixed-step training. Its time-to-target uses validation accuracy >= 0.8. Heisenberg score is final relative energy error; its target is <= 0.01.
The six-parameter quadratic classical control is well matched to the XOR label rule. Its zero initialization is identical across paired runs, so those runs do not establish seed-dependent model variability.

## Per-seed outcomes

| Task | Engine | Seed | Final score | Training seconds | Warmup seconds | First value/gradient seconds | Time to target seconds | Replay max error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 11 | 1 | 2.5607 | 0.0380 | 0.0378 | 1.0492 | 0 |
| classifier | flagquantum | 29 | 1 | 2.5200 | 0.0346 | 0.0107 | 1.0264 | 0 |
| classifier | flagquantum | 47 | 1 | 2.4670 | 0.0341 | 0.0102 | 0.8824 | 0 |
| classifier | flagquantum_jax_kernel | 11 | 1 | 0.8564 | 0.0138 | 0.1931 | 0.4792 | 0 |
| classifier | flagquantum_jax_kernel | 29 | 1 | 0.6776 | 0.0080 | 0.1201 | 0.2929 | 0 |
| classifier | flagquantum_jax_kernel | 47 | 1 | 0.6820 | 0.0078 | 0.1186 | 0.2660 | 0 |
| classifier | flagquantum_jax_module | 11 | 1 | 1.3433 | 0.0160 | 0.9452 | 0.5563 | 0 |
| classifier | flagquantum_jax_module | 29 | 1 | 1.4463 | 0.0172 | 0.6953 | 0.5679 | 0 |
| classifier | flagquantum_jax_module | 47 | 1 | 2.2838 | 0.0175 | 0.7575 | 0.6757 | 0 |
| classifier | tensorcircuit_jax | 11 | 1 | 0.1411 | 0.0019 | 0.1277 | 0.0569 | 0 |
| classifier | tensorcircuit_jax | 29 | 1 | 0.1332 | 0.0020 | 0.1170 | 0.0549 | 0 |
| classifier | tensorcircuit_jax | 47 | 1 | 0.1314 | 0.0018 | 0.1089 | 0.0473 | 0 |
| heisenberg | flagquantum | 11 | 0.00019206107 | 0.2988 | 0.0042 | 0.0134 | 0.1074 | 0 |
| heisenberg | flagquantum | 29 | 0.00012317999 | 0.2969 | 0.0038 | 0.0119 | 0.0881 | 0 |
| heisenberg | flagquantum | 47 | 0.00013303469 | 0.2945 | 0.0041 | 0.0112 | 0.0509 | 0 |
| heisenberg | flagquantum_jax_kernel | 11 | 0.00019206107 | 0.0339 | 0.0005 | 0.5311 | 0.0128 | 0 |
| heisenberg | flagquantum_jax_kernel | 29 | 0.00012317999 | 0.0342 | 0.0005 | 0.4937 | 0.0107 | 0 |
| heisenberg | flagquantum_jax_kernel | 47 | 0.00013303469 | 0.0332 | 0.0005 | 0.5027 | 0.0059 | 0 |
| heisenberg | flagquantum_jax_module | 11 | 0.00019206107 | 0.0720 | 0.0012 | 0.9890 | 0.0260 | 0 |
| heisenberg | flagquantum_jax_module | 29 | 0.00012317999 | 0.0725 | 0.0011 | 1.0095 | 0.0221 | 0 |
| heisenberg | flagquantum_jax_module | 47 | 0.00013303469 | 0.0719 | 0.0010 | 1.0163 | 0.0128 | 0 |
| heisenberg | tensorcircuit_jax | 11 | 0.00019206107 | 0.0261 | 0.0003 | 0.9023 | 0.0099 | 0 |
| heisenberg | tensorcircuit_jax | 29 | 0.00012317999 | 0.0265 | 0.0003 | 1.0135 | 0.0089 | 0 |
| heisenberg | tensorcircuit_jax | 47 | 0.00013303469 | 0.0241 | 0.0003 | 0.8316 | 0.0049 | 0 |

## Retained failures and limits

No case failed or timed out in this directory.

QNG algorithm/fusion ablation, batched/vmapped throughput, larger-task quality, repeated timing campaigns, GPU runs and AI-assisted error correction are outside this artifact. No quantum advantage or generalization advantage is established.
