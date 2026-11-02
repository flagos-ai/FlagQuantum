# CPU training comparison

Producer: `d76c095afba994972d4a0dfce9a924a2f7143673`.
Protocol SHA-256: `5750e9cc054737447b2c16ea3b9dfa79934248294f5e058686c27b5c082e8d5e`.
Smoke: `False`. This is CPU comparison evidence, not release or scalability evidence.

Training time includes optimizer updates and per-update validation; imports, oracle checks and replay are outside that timer.
Peak RSS includes imports, oracle checks, warmup and replay. These small, serial-example workloads are not a ranking of the fastest available configurations.

| Task | Engine | Completed | Target reached | Training seconds, median | Final task score, median [min, max] | Peak process RSS MiB, median |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| classifier | classical | 3 | 3 | 0.0048 | 0.90625 [0.90625, 0.90625] | 285.1 |
| classifier | flagquantum | 3 | 3 | 2.5635 | 1 [1, 1] | 299.5 |
| classifier | pennylane_adjoint | 3 | 3 | 2.9312 | 1 [1, 1] | 465.8 |
| classifier | pennylane_backprop | 3 | 3 | 3.3483 | 1 [1, 1] | 453.1 |
| classifier | tensorcircuit_jax | 3 | 3 | 0.1305 | 1 [1, 1] | 537.7 |
| classifier | tensorcircuit_torch | 3 | 3 | 3.3597 | 1 [1, 1] | 455.7 |
| heisenberg | flagquantum | 3 | 3 | 0.2895 | 0.000133035 [0.00012318, 0.000192061] | 290.0 |
| heisenberg | pennylane_adjoint | 3 | 3 | 0.1405 | 0.000133035 [0.00012318, 0.000192061] | 463.3 |
| heisenberg | pennylane_backprop | 3 | 3 | 0.4005 | 0.000133035 [0.00012318, 0.000192061] | 463.7 |
| heisenberg | tensorcircuit_jax | 3 | 3 | 0.0245 | 0.000133035 [0.00012318, 0.000192061] | 603.7 |
| heisenberg | tensorcircuit_torch | 3 | 3 | 0.4873 | 0.000133035 [0.00012318, 0.000192061] | 449.8 |

Classifier score is held-out test accuracy after fixed-step training. Its time-to-target uses validation accuracy >= 0.8. Heisenberg score is final relative energy error; its target is <= 0.01.
The six-parameter quadratic classical control is well matched to the XOR label rule. Its zero initialization is identical across paired runs, so those runs do not establish seed-dependent model variability.

## Per-seed outcomes

| Task | Engine | Seed | Final score | Training seconds | Warmup seconds | First value/gradient seconds | Time to target seconds | Replay max error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| classifier | classical | 11 | 0.90625 | 0.0048 | 0.0001 | 0.0047 | 0.0021 | 0 |
| classifier | classical | 29 | 0.90625 | 0.0052 | 0.0000 | 0.0048 | 0.0021 | 0 |
| classifier | classical | 47 | 0.90625 | 0.0047 | 0.0001 | 0.0047 | 0.0020 | 0 |
| classifier | flagquantum | 11 | 1 | 2.5865 | 0.0357 | 0.0222 | 1.0188 | 0 |
| classifier | flagquantum | 29 | 1 | 2.5314 | 0.0348 | 0.0086 | 0.9930 | 0 |
| classifier | flagquantum | 47 | 1 | 2.5635 | 0.0351 | 0.0093 | 0.8681 | 0 |
| classifier | pennylane_adjoint | 11 | 1 | 3.1346 | 0.0332 | 0.0137 | 1.3018 | 0 |
| classifier | pennylane_adjoint | 29 | 1 | 2.7883 | 0.0338 | 0.0113 | 1.1663 | 0 |
| classifier | pennylane_adjoint | 47 | 1 | 2.9312 | 0.0323 | 0.0100 | 0.9842 | 0 |
| classifier | pennylane_backprop | 11 | 1 | 3.2473 | 0.0384 | 0.0109 | 1.2011 | 0 |
| classifier | pennylane_backprop | 29 | 1 | 3.3730 | 0.0400 | 0.0108 | 1.2013 | 0 |
| classifier | pennylane_backprop | 47 | 1 | 3.3483 | 0.0396 | 0.0109 | 1.0833 | 0 |
| classifier | tensorcircuit_jax | 11 | 1 | 0.1324 | 0.0019 | 0.2411 | 0.0541 | 0 |
| classifier | tensorcircuit_jax | 29 | 1 | 0.1297 | 0.0018 | 0.1088 | 0.0527 | 0 |
| classifier | tensorcircuit_jax | 47 | 1 | 0.1305 | 0.0018 | 0.1096 | 0.0472 | 0 |
| classifier | tensorcircuit_torch | 11 | 1 | 3.3597 | 0.0432 | 0.0108 | 1.3799 | 0 |
| classifier | tensorcircuit_torch | 29 | 1 | 3.3934 | 0.0426 | 0.0100 | 1.4124 | 0 |
| classifier | tensorcircuit_torch | 47 | 1 | 3.3578 | 0.0427 | 0.0106 | 1.1909 | 0 |
| heisenberg | flagquantum | 11 | 0.00019206107 | 0.2886 | 0.0039 | 0.0145 | 0.1038 | 0 |
| heisenberg | flagquantum | 29 | 0.00012317999 | 0.2926 | 0.0038 | 0.0099 | 0.0923 | 0 |
| heisenberg | flagquantum | 47 | 0.00013303469 | 0.2895 | 0.0038 | 0.0101 | 0.0500 | 0 |
| heisenberg | pennylane_adjoint | 11 | 0.00019206107 | 0.1405 | 0.0017 | 0.0096 | 0.0511 | 0 |
| heisenberg | pennylane_adjoint | 29 | 0.00012317999 | 0.1377 | 0.0019 | 0.0087 | 0.0429 | 0 |
| heisenberg | pennylane_adjoint | 47 | 0.00013303469 | 0.1481 | 0.0021 | 0.0096 | 0.0253 | 0 |
| heisenberg | pennylane_backprop | 11 | 0.00019206107 | 0.4005 | 0.0045 | 0.0118 | 0.1410 | 0 |
| heisenberg | pennylane_backprop | 29 | 0.00012317999 | 0.4146 | 0.0047 | 0.0120 | 0.1281 | 0 |
| heisenberg | pennylane_backprop | 47 | 0.00013303469 | 0.3988 | 0.0048 | 0.0119 | 0.0698 | 0 |
| heisenberg | tensorcircuit_jax | 11 | 0.00019206107 | 0.0253 | 0.0003 | 0.7492 | 0.0096 | 0 |
| heisenberg | tensorcircuit_jax | 29 | 0.00012317999 | 0.0245 | 0.0003 | 0.7382 | 0.0078 | 0 |
| heisenberg | tensorcircuit_jax | 47 | 0.00013303469 | 0.0245 | 0.0003 | 0.7657 | 0.0049 | 0 |
| heisenberg | tensorcircuit_torch | 11 | 0.00019206107 | 0.4873 | 0.0049 | 0.0117 | 0.1716 | 0 |
| heisenberg | tensorcircuit_torch | 29 | 0.00012317999 | 0.4843 | 0.0049 | 0.0122 | 0.1455 | 0 |
| heisenberg | tensorcircuit_torch | 47 | 0.00013303469 | 0.4878 | 0.0049 | 0.0117 | 0.0835 | 0 |

## Total cost including first call

Measured total includes setup, first call, warmup, initial validation (when recorded), training and final assessment. Training already includes checkpoint writing. Oracle and recovery are excluded from measured total. Process wall includes interpreter startup, imports, oracle and recovery; it is an application diagnostic, not a pure simulator ranking.

| Task | Engine | Seed | Per-step update s, median | Total measured s | Total time to target s | Fresh-process wall s |
| --- | --- | ---: | ---: | ---: | --- | --- |
| classifier | classical | 11 | 0.000041 | 0.009743 | 0.006998 | 1.055205 |
| classifier | classical | 29 | 0.000042 | 0.010243 | 0.007162 | 1.159022 |
| classifier | classical | 47 | 0.000040 | 0.009625 | 0.006908 | 1.139448 |
| classifier | flagquantum | 11 | 0.033618 | 2.689961 | 1.088133 | 4.997085 |
| classifier | flagquantum | 29 | 0.031830 | 2.616874 | 1.045952 | 4.784469 |
| classifier | flagquantum | 47 | 0.032912 | 2.650926 | 0.922197 | 5.059206 |
| classifier | pennylane_adjoint | 11 | 0.034681 | 4.257668 | 2.379716 | 7.133516 |
| classifier | pennylane_adjoint | 29 | 0.031881 | 3.715730 | 2.048097 | 6.084423 |
| classifier | pennylane_adjoint | 47 | 0.032990 | 3.790421 | 1.795554 | 6.186173 |
| classifier | pennylane_backprop | 11 | 0.036769 | 4.639737 | 2.542405 | 7.595013 |
| classifier | pennylane_backprop | 29 | 0.037038 | 4.407213 | 2.167339 | 7.216170 |
| classifier | pennylane_backprop | 47 | 0.038646 | 4.253992 | 1.934441 | 7.084484 |
| classifier | tensorcircuit_jax | 11 | 0.001523 | 1.086741 | 1.006127 | 2.279493 |
| classifier | tensorcircuit_jax | 29 | 0.001493 | 0.949061 | 0.869615 | 2.186357 |
| classifier | tensorcircuit_jax | 47 | 0.001496 | 0.946414 | 0.860720 | 2.131225 |
| classifier | tensorcircuit_torch | 11 | 0.040444 | 4.343794 | 2.306987 | 6.762668 |
| classifier | tensorcircuit_torch | 29 | 0.040815 | 4.214729 | 2.177670 | 6.689310 |
| classifier | tensorcircuit_torch | 47 | 0.040456 | 4.188370 | 1.964879 | 6.628092 |
| heisenberg | flagquantum | 11 | 0.003171 | 0.313995 | 0.124832 | 1.537958 |
| heisenberg | flagquantum | 29 | 0.003197 | 0.313731 | 0.108691 | 1.544530 |
| heisenberg | flagquantum | 47 | 0.003197 | 0.310609 | 0.066599 | 1.529100 |
| heisenberg | pennylane_adjoint | 11 | 0.001271 | 0.928704 | 0.836654 | 2.241190 |
| heisenberg | pennylane_adjoint | 29 | 0.001254 | 0.896013 | 0.798559 | 2.189211 |
| heisenberg | pennylane_adjoint | 47 | 0.001351 | 0.912811 | 0.787087 | 2.345829 |
| heisenberg | pennylane_backprop | 11 | 0.003929 | 1.291619 | 1.024286 | 2.686426 |
| heisenberg | pennylane_backprop | 29 | 0.003991 | 1.181986 | 0.887838 | 2.578209 |
| heisenberg | pennylane_backprop | 47 | 0.003892 | 1.164692 | 0.828074 | 2.586888 |
| heisenberg | tensorcircuit_jax | 11 | 0.000114 | 1.488406 | 1.472166 | 2.688314 |
| heisenberg | tensorcircuit_jax | 29 | 0.000110 | 1.447657 | 1.430342 | 2.628664 |
| heisenberg | tensorcircuit_jax | 47 | 0.000109 | 1.495923 | 1.475707 | 2.689066 |
| heisenberg | tensorcircuit_torch | 11 | 0.004473 | 1.361849 | 1.035753 | 2.635885 |
| heisenberg | tensorcircuit_torch | 29 | 0.004427 | 1.262788 | 0.913613 | 2.601412 |
| heisenberg | tensorcircuit_torch | 47 | 0.004468 | 1.199400 | 0.784735 | 2.510239 |

## Retained failures and limits

No case failed or timed out in this directory.

QNG algorithm/fusion ablation, batched/vmapped throughput, larger-task quality, repeated timing campaigns, GPU runs and AI-assisted error correction are outside this artifact. No quantum advantage or generalization advantage is established.
