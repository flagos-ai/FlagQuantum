# CPU training comparison

Producer: `cdbf4ecf513596ec8338468ed5ba1ca84c5b5cd8`.
Protocol SHA-256: `258f5736f36988f3c3533ecad36d3498163a809d5c88eba12cfb2ea66eab01f7`.
Smoke: `False`. This is CPU comparison evidence, not release or scalability evidence.

Training time includes optimizer updates and per-update validation; imports, oracle checks and replay are outside that timer.
Peak RSS includes imports, oracle checks, warmup and replay. These small, serial-example workloads are not a ranking of the fastest available configurations.

| Task | Engine | Completed | Target reached | Training seconds, median | Final task score, median [min, max] | Peak process RSS MiB, median |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 3 | 3 | 2.7593 | 1 [1, 1] | 297.0 |
| classifier | flagquantum_jax_kernel | 3 | 3 | 0.2807 | 1 [1, 1] | 451.1 |
| classifier | flagquantum_jax_module | 3 | 3 | 0.6738 | 1 [1, 1] | 458.9 |
| classifier | tensorcircuit_jax | 3 | 3 | 0.1360 | 1 [1, 1] | 523.7 |
| heisenberg | flagquantum | 3 | 3 | 0.3632 | 0.000133035 [0.00012318, 0.000192061] | 287.1 |
| heisenberg | flagquantum_jax_kernel | 3 | 3 | 0.0350 | 0.000133035 [0.00012318, 0.000192061] | 493.0 |
| heisenberg | flagquantum_jax_module | 3 | 3 | 0.0563 | 0.000133035 [0.00012318, 0.000192061] | 498.4 |
| heisenberg | tensorcircuit_jax | 3 | 3 | 0.0370 | 0.000133035 [0.00012318, 0.000192061] | 563.4 |

Classifier score is held-out test accuracy after fixed-step training. Its time-to-target uses validation accuracy >= 0.8. Heisenberg score is final relative energy error; its target is <= 0.01.
The six-parameter quadratic classical control is well matched to the XOR label rule. Its zero initialization is identical across paired runs, so those runs do not establish seed-dependent model variability.

## Per-seed outcomes

| Task | Engine | Seed | Final score | Training seconds | Warmup seconds | First value/gradient seconds | Time to target seconds | Replay max error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 11 | 1 | 4.8966 | 0.0592 | 0.0373 | 2.0969 | 0 |
| classifier | flagquantum | 29 | 1 | 2.7593 | 0.0402 | 0.0118 | 1.1067 | 0 |
| classifier | flagquantum | 47 | 1 | 2.4910 | 0.0352 | 0.0086 | 0.8904 | 0 |
| classifier | flagquantum_jax_kernel | 11 | 1 | 0.3005 | 0.0040 | 0.1141 | 0.1400 | 0 |
| classifier | flagquantum_jax_kernel | 29 | 1 | 0.2742 | 0.0035 | 0.1239 | 0.1333 | 0 |
| classifier | flagquantum_jax_kernel | 47 | 1 | 0.2807 | 0.0035 | 0.1159 | 0.1266 | 0 |
| classifier | flagquantum_jax_module | 11 | 1 | 0.6456 | 0.0083 | 0.8373 | 0.2604 | 0 |
| classifier | flagquantum_jax_module | 29 | 1 | 0.6830 | 0.0077 | 0.6221 | 0.2650 | 0 |
| classifier | flagquantum_jax_module | 47 | 1 | 0.6738 | 0.0090 | 0.6497 | 0.2452 | 0 |
| classifier | tensorcircuit_jax | 11 | 1 | 0.1357 | 0.0019 | 0.1152 | 0.0555 | 0 |
| classifier | tensorcircuit_jax | 29 | 1 | 0.3600 | 0.0025 | 0.1433 | 0.1163 | 0 |
| classifier | tensorcircuit_jax | 47 | 1 | 0.1360 | 0.0021 | 0.1169 | 0.0473 | 0 |
| heisenberg | flagquantum | 11 | 0.00019206107 | 0.3078 | 0.0081 | 0.0468 | 0.1086 | 0 |
| heisenberg | flagquantum | 29 | 0.00012317999 | 0.3632 | 0.0039 | 0.0099 | 0.1049 | 0 |
| heisenberg | flagquantum | 47 | 0.00013303469 | 0.3780 | 0.0050 | 0.0118 | 0.0641 | 0 |
| heisenberg | flagquantum_jax_kernel | 11 | 0.00019206107 | 0.0279 | 0.0004 | 0.4963 | 0.0100 | 0 |
| heisenberg | flagquantum_jax_kernel | 29 | 0.00012317999 | 0.0350 | 0.0008 | 2.5169 | 0.0114 | 0 |
| heisenberg | flagquantum_jax_kernel | 47 | 0.00013303469 | 0.0655 | 0.0007 | 0.6556 | 0.0093 | 0 |
| heisenberg | flagquantum_jax_module | 11 | 0.00019206107 | 0.0905 | 0.0013 | 1.1841 | 0.0307 | 0 |
| heisenberg | flagquantum_jax_module | 29 | 0.00012317999 | 0.0539 | 0.0007 | 1.0457 | 0.0158 | 0 |
| heisenberg | flagquantum_jax_module | 47 | 0.00013303469 | 0.0563 | 0.0008 | 1.0803 | 0.0100 | 0 |
| heisenberg | tensorcircuit_jax | 11 | 0.00019206107 | 0.0360 | 0.0008 | 1.4342 | 0.0148 | 0 |
| heisenberg | tensorcircuit_jax | 29 | 0.00012317999 | 0.0370 | 0.0004 | 0.8983 | 0.0123 | 0 |
| heisenberg | tensorcircuit_jax | 47 | 0.00013303469 | 0.0616 | 0.0004 | 1.2623 | 0.0129 | 0 |

## Retained failures and limits

No case failed or timed out in this directory.

QNG algorithm/fusion ablation, batched/vmapped throughput, larger-task quality, repeated timing campaigns, GPU runs and AI-assisted error correction are outside this artifact. No quantum advantage or generalization advantage is established.
