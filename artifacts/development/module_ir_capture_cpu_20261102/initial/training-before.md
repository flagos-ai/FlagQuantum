# CPU training comparison

Producer: `abde02fd05100b3ae2a65d68128d459cfd437dbe`.
Protocol SHA-256: `02f15449c3f5aaf8ebb34151aaaadb196ea8cffa01602bd6ee5d73882ba190a0`.
Smoke: `False`. This is CPU comparison evidence, not release or scalability evidence.

Training time includes optimizer updates and per-update validation; imports, oracle checks and replay are outside that timer.
Peak RSS includes imports, oracle checks, warmup and replay. These small, serial-example workloads are not a ranking of the fastest available configurations.

| Task | Engine | Completed | Target reached | Training seconds, median | Final task score, median [min, max] | Peak process RSS MiB, median |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 3 | 3 | 2.7531 | 1 [1, 1] | 298.0 |
| classifier | flagquantum_jax_kernel | 3 | 3 | 0.1584 | 1 [1, 1] | 453.6 |
| classifier | flagquantum_jax_module | 3 | 3 | 0.3698 | 1 [1, 1] | 462.8 |
| classifier | flagquantum_torch_module | 3 | 3 | 2.5608 | 1 [1, 1] | 303.1 |
| classifier | tensorcircuit_jax | 3 | 3 | 0.1466 | 1 [1, 1] | 529.5 |
| heisenberg | flagquantum | 3 | 3 | 0.3533 | 0.000133035 [0.00012318, 0.000192061] | 288.7 |
| heisenberg | flagquantum_jax_kernel | 3 | 3 | 0.0266 | 0.000133035 [0.00012318, 0.000192061] | 545.6 |
| heisenberg | flagquantum_jax_module | 3 | 3 | 0.0376 | 0.000133035 [0.00012318, 0.000192061] | 562.4 |
| heisenberg | flagquantum_torch_module | 3 | 3 | 0.3520 | 0.000133035 [0.00012318, 0.000192061] | 292.0 |
| heisenberg | tensorcircuit_jax | 3 | 3 | 0.0291 | 0.000133035 [0.00012318, 0.000192061] | 570.5 |

Classifier score is held-out test accuracy after fixed-step training. Its time-to-target uses validation accuracy >= 0.8. Heisenberg score is final relative energy error; its target is <= 0.01.
The six-parameter quadratic classical control is well matched to the XOR label rule. Its zero initialization is identical across paired runs, so those runs do not establish seed-dependent model variability.

## Per-seed outcomes

| Task | Engine | Seed | Final score | Training seconds | Warmup seconds | First value/gradient seconds | Time to target seconds | Replay max error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| classifier | flagquantum | 11 | 1 | 2.7724 | 0.0389 | 0.0149 | 1.0917 | 0 |
| classifier | flagquantum | 29 | 1 | 2.7516 | 0.0370 | 0.0108 | 1.0646 | 0 |
| classifier | flagquantum | 47 | 1 | 2.7531 | 0.0381 | 0.0100 | 0.9683 | 0 |
| classifier | flagquantum_jax_kernel | 11 | 1 | 0.1734 | 0.0023 | 0.2055 | 0.0672 | 0 |
| classifier | flagquantum_jax_kernel | 29 | 1 | 0.1565 | 0.0026 | 0.1995 | 0.0609 | 0 |
| classifier | flagquantum_jax_kernel | 47 | 1 | 0.1584 | 0.0025 | 0.2780 | 0.0578 | 0 |
| classifier | flagquantum_jax_module | 11 | 1 | 0.3698 | 0.0141 | 1.2506 | 0.1548 | 0 |
| classifier | flagquantum_jax_module | 29 | 1 | 0.3867 | 0.0051 | 0.9169 | 0.1595 | 0 |
| classifier | flagquantum_jax_module | 47 | 1 | 0.3504 | 0.0048 | 0.7819 | 0.1215 | 0 |
| classifier | flagquantum_torch_module | 11 | 1 | 2.7233 | 0.0366 | 0.4487 | 0.9711 | 0 |
| classifier | flagquantum_torch_module | 29 | 1 | 2.5608 | 0.0398 | 0.4339 | 1.0227 | 0 |
| classifier | flagquantum_torch_module | 47 | 1 | 2.5277 | 0.0363 | 0.4283 | 0.8586 | 0 |
| classifier | tensorcircuit_jax | 11 | 1 | 0.1487 | 0.0022 | 0.1326 | 0.0598 | 0 |
| classifier | tensorcircuit_jax | 29 | 1 | 0.1460 | 0.0022 | 0.1386 | 0.0598 | 0 |
| classifier | tensorcircuit_jax | 47 | 1 | 0.1466 | 0.0028 | 0.1255 | 0.0520 | 0 |
| heisenberg | flagquantum | 11 | 0.00019206107 | 0.3350 | 0.0044 | 0.0186 | 0.1168 | 0 |
| heisenberg | flagquantum | 29 | 0.00012317999 | 0.3628 | 0.0043 | 0.0125 | 0.1174 | 0 |
| heisenberg | flagquantum | 47 | 0.00013303469 | 0.3533 | 0.0042 | 0.0124 | 0.0540 | 0 |
| heisenberg | flagquantum_jax_kernel | 11 | 0.00019206107 | 0.0266 | 0.0004 | 0.7796 | 0.0091 | 0 |
| heisenberg | flagquantum_jax_kernel | 29 | 0.00012317999 | 0.0258 | 0.0005 | 0.8488 | 0.0083 | 0 |
| heisenberg | flagquantum_jax_kernel | 47 | 0.00013303469 | 0.0305 | 0.0004 | 0.8255 | 0.0071 | 0 |
| heisenberg | flagquantum_jax_module | 11 | 0.00019206107 | 0.0375 | 0.0006 | 1.7904 | 0.0130 | 0 |
| heisenberg | flagquantum_jax_module | 29 | 0.00012317999 | 0.0381 | 0.0005 | 1.3784 | 0.0113 | 0 |
| heisenberg | flagquantum_jax_module | 47 | 0.00013303469 | 0.0376 | 0.0008 | 1.3886 | 0.0067 | 0 |
| heisenberg | flagquantum_torch_module | 11 | 0.00019206107 | 0.3573 | 0.0048 | 0.5745 | 0.1297 | 0 |
| heisenberg | flagquantum_torch_module | 29 | 0.00012317999 | 0.3425 | 0.0047 | 0.4443 | 0.1028 | 0 |
| heisenberg | flagquantum_torch_module | 47 | 0.00013303469 | 0.3520 | 0.0046 | 0.4484 | 0.0594 | 0 |
| heisenberg | tensorcircuit_jax | 11 | 0.00019206107 | 0.0293 | 0.0004 | 0.8703 | 0.0113 | 0 |
| heisenberg | tensorcircuit_jax | 29 | 0.00012317999 | 0.0291 | 0.0003 | 0.8727 | 0.0098 | 0 |
| heisenberg | tensorcircuit_jax | 47 | 0.00013303469 | 0.0264 | 0.0003 | 0.8624 | 0.0053 | 0 |

## Total cost including first call

Measured total includes setup, first call, warmup, initial validation (when recorded), training and final assessment. Training already includes checkpoint writing. Oracle and recovery are excluded from measured total. Process wall includes interpreter startup, imports, oracle and recovery; it is an application diagnostic, not a pure simulator ranking.

| Task | Engine | Seed | Per-step update s, median | Total measured s | Total time to target s | Fresh-process wall s |
| --- | --- | ---: | ---: | ---: | --- | --- |
| classifier | flagquantum | 11 | 0.036852 | 2.872933 | 1.155806 | 5.152190 |
| classifier | flagquantum | 29 | 0.035400 | 2.845335 | 1.122028 | 5.115395 |
| classifier | flagquantum | 47 | 0.036249 | 2.847198 | 1.026475 | 5.182999 |
| classifier | flagquantum_jax_kernel | 11 | 0.002150 | 0.623919 | 0.515717 | 2.095295 |
| classifier | flagquantum_jax_kernel | 29 | 0.001995 | 0.575013 | 0.477420 | 1.930684 |
| classifier | flagquantum_jax_kernel | 47 | 0.001988 | 0.689712 | 0.587203 | 2.152491 |
| classifier | flagquantum_jax_module | 11 | 0.004393 | 1.709541 | 1.488808 | 2.820157 |
| classifier | flagquantum_jax_module | 29 | 0.004622 | 1.380226 | 1.146720 | 2.656180 |
| classifier | flagquantum_jax_module | 47 | 0.004226 | 1.203534 | 0.969000 | 2.343135 |
| classifier | flagquantum_torch_module | 11 | 0.034361 | 3.252032 | 1.471967 | 5.141697 |
| classifier | flagquantum_torch_module | 29 | 0.035079 | 3.074582 | 1.508441 | 4.909369 |
| classifier | flagquantum_torch_module | 47 | 0.033899 | 3.031639 | 1.334394 | 4.888248 |
| classifier | tensorcircuit_jax | 11 | 0.001718 | 1.299907 | 1.208350 | 2.644436 |
| classifier | tensorcircuit_jax | 29 | 0.001672 | 1.168866 | 1.079969 | 2.571751 |
| classifier | tensorcircuit_jax | 47 | 0.001667 | 1.076867 | 0.979650 | 2.498262 |
| heisenberg | flagquantum | 11 | 0.003711 | 0.367321 | 0.142930 | 1.808984 |
| heisenberg | flagquantum | 29 | 0.003920 | 0.387848 | 0.137463 | 1.897439 |
| heisenberg | flagquantum | 47 | 0.003697 | 0.378756 | 0.073557 | 1.799771 |
| heisenberg | flagquantum_jax_kernel | 11 | 0.000112 | 1.161537 | 1.143496 | 2.599509 |
| heisenberg | flagquantum_jax_kernel | 29 | 0.000111 | 1.247096 | 1.229070 | 2.755020 |
| heisenberg | flagquantum_jax_kernel | 47 | 0.000140 | 1.242311 | 1.218329 | 2.829486 |
| heisenberg | flagquantum_jax_module | 11 | 0.000218 | 2.021464 | 1.996176 | 2.971011 |
| heisenberg | flagquantum_jax_module | 29 | 0.000218 | 1.604694 | 1.577035 | 2.650311 |
| heisenberg | flagquantum_jax_module | 47 | 0.000211 | 1.621614 | 1.589888 | 2.632661 |
| heisenberg | flagquantum_torch_module | 11 | 0.004117 | 0.949725 | 0.717566 | 1.942951 |
| heisenberg | flagquantum_torch_module | 29 | 0.003925 | 0.803459 | 0.558534 | 1.791079 |
| heisenberg | flagquantum_torch_module | 47 | 0.004047 | 0.816564 | 0.519012 | 1.806220 |
| heisenberg | tensorcircuit_jax | 11 | 0.000126 | 1.940437 | 1.921887 | 3.367471 |
| heisenberg | tensorcircuit_jax | 29 | 0.000132 | 1.720704 | 1.700786 | 3.171592 |
| heisenberg | tensorcircuit_jax | 47 | 0.000117 | 1.693577 | 1.671777 | 3.033369 |

## Retained failures and limits

No case failed or timed out in this directory.

QNG algorithm/fusion ablation, batched/vmapped throughput, larger-task quality, repeated timing campaigns, GPU runs and AI-assisted error correction are outside this artifact. No quantum advantage or generalization advantage is established.
