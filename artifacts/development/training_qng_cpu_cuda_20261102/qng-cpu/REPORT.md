# Matched QNG algorithm and fusion ablation

Producer: `d76c095afba994972d4a0dfce9a924a2f7143673`. Protocol: `qng_hva_ablation_cpu_20261102`. Smoke: `False`.

The objective gradient, damping, layer-block projection and update rule are identical. Only metric construction changes. PennyLane supplies its framework metric, with the same FlagQuantum objective gradient; this isolates geometry and is not an end-to-end PennyLane optimizer ranking.

First-call/JIT and warmup are separate and included in total cost and total time to target. Oracle validation and common imports are outside measured total; fresh-process wall includes them and interpreter startup. Peak RSS includes the entire worker. Fixed updates continue after reaching the target. No quantum advantage, state-of-the-art or scalability claim is made.

| Wires | Depth | Parameters | Method | Seed | Status | Final error | Training s | First call s | Total s | Total target s | RSS MiB | CUDA MiB |
| ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 4 | 1 | 13 | [reverse_jacobian](q4-d1-reverse_jacobian-11.json) | 11 | completed | 0.0354933 | 0.139368 | 0.114114 | 0.263321 | not reached | 246.9 | — |
| 4 | 1 | 13 | [reverse_jacobian](q4-d1-reverse_jacobian-29.json) | 29 | completed | 0.0354933 | 0.132957 | 0.110452 | 0.251627 | not reached | 246.7 | — |
| 4 | 1 | 13 | [reverse_jacobian](q4-d1-reverse_jacobian-47.json) | 47 | completed | 0.0354933 | 0.135067 | 0.111047 | 0.254279 | not reached | 246.8 | — |
| 4 | 1 | 13 | [unfused_forward](q4-d1-unfused_forward-11.json) | 11 | completed | 0.0354933 | 0.067238 | 0.006719 | 0.080226 | not reached | 213.7 | — |
| 4 | 1 | 13 | [unfused_forward](q4-d1-unfused_forward-29.json) | 29 | completed | 0.0354933 | 0.065768 | 0.006636 | 0.078394 | not reached | 213.5 | — |
| 4 | 1 | 13 | [unfused_forward](q4-d1-unfused_forward-47.json) | 47 | completed | 0.0354933 | 0.065783 | 0.006798 | 0.078853 | not reached | 213.7 | — |
| 4 | 1 | — | fused_forward | 11 | unsupported: the existing Triton HVA kernel requires CUDA complex64 | — | — | — | — | — | — | — |
| 4 | 1 | — | fused_forward | 29 | unsupported: the existing Triton HVA kernel requires CUDA complex64 | — | — | — | — | — | — | — |
| 4 | 1 | — | fused_forward | 47 | unsupported: the existing Triton HVA kernel requires CUDA complex64 | — | — | — | — | — | — | — |
| 4 | 1 | 13 | [pennylane_metric](q4-d1-pennylane_metric-11.json) | 11 | completed | 0.0354933 | 0.747671 | 0.035565 | 1.749196 | not reached | 396.7 | — |
| 4 | 1 | 13 | [pennylane_metric](q4-d1-pennylane_metric-29.json) | 29 | completed | 0.0354933 | 0.728382 | 0.033585 | 1.506215 | not reached | 396.5 | — |
| 4 | 1 | 13 | [pennylane_metric](q4-d1-pennylane_metric-47.json) | 47 | completed | 0.0354933 | 0.739870 | 0.033977 | 1.535553 | not reached | 396.5 | — |
| 4 | 2 | 26 | [reverse_jacobian](q4-d2-reverse_jacobian-11.json) | 11 | completed | 6.37891e-08 | 0.249795 | 0.196278 | 0.459115 | 0.243020 | 247.7 | — |
| 4 | 2 | 26 | [reverse_jacobian](q4-d2-reverse_jacobian-29.json) | 29 | completed | 2.01881e-06 | 0.251084 | 0.114688 | 0.378702 | 0.161838 | 247.4 | — |
| 4 | 2 | 26 | [reverse_jacobian](q4-d2-reverse_jacobian-47.json) | 47 | completed | 1.4021e-06 | 0.252266 | 0.118040 | 0.383719 | 0.165630 | 247.7 | — |
| 4 | 2 | 26 | [unfused_forward](q4-d2-unfused_forward-11.json) | 11 | completed | 6.37891e-08 | 0.140199 | 0.008853 | 0.158631 | 0.035584 | 214.5 | — |
| 4 | 2 | 26 | [unfused_forward](q4-d2-unfused_forward-29.json) | 29 | completed | 2.01881e-06 | 0.132034 | 0.008035 | 0.148497 | 0.031880 | 214.3 | — |
| 4 | 2 | 26 | [unfused_forward](q4-d2-unfused_forward-47.json) | 47 | completed | 1.4021e-06 | 0.136096 | 0.008093 | 0.152893 | 0.033324 | 214.4 | — |
| 4 | 2 | — | fused_forward | 11 | unsupported: the existing Triton HVA kernel requires CUDA complex64 | — | — | — | — | — | — | — |
| 4 | 2 | — | fused_forward | 29 | unsupported: the existing Triton HVA kernel requires CUDA complex64 | — | — | — | — | — | — | — |
| 4 | 2 | — | fused_forward | 47 | unsupported: the existing Triton HVA kernel requires CUDA complex64 | — | — | — | — | — | — | — |
| 4 | 2 | 26 | [pennylane_metric](q4-d2-pennylane_metric-11.json) | 11 | completed | 6.37891e-08 | 2.631468 | 0.097089 | 3.627047 | 1.334237 | 412.8 | — |
| 4 | 2 | 26 | [pennylane_metric](q4-d2-pennylane_metric-29.json) | 29 | completed | 2.01881e-06 | 2.631332 | 0.096986 | 3.543164 | 1.250769 | 412.8 | — |
| 4 | 2 | 26 | [pennylane_metric](q4-d2-pennylane_metric-47.json) | 47 | completed | 1.4021e-06 | 2.627136 | 0.097622 | 3.566075 | 1.279177 | 413.0 | — |
| 6 | 1 | 21 | [reverse_jacobian](q6-d1-reverse_jacobian-11.json) | 11 | completed | 0.0502094 | 0.403834 | 0.176377 | 0.601247 | not reached | 250.2 | — |
| 6 | 1 | 21 | [reverse_jacobian](q6-d1-reverse_jacobian-29.json) | 29 | completed | 0.0502094 | 0.402300 | 0.120217 | 0.541433 | not reached | 249.8 | — |
| 6 | 1 | 21 | [reverse_jacobian](q6-d1-reverse_jacobian-47.json) | 47 | completed | 0.0502253 | 0.429377 | 0.121257 | 0.569873 | not reached | 250.3 | — |
| 6 | 1 | 21 | [unfused_forward](q6-d1-unfused_forward-11.json) | 11 | completed | 0.0502094 | 0.120135 | 0.007754 | 0.137031 | not reached | 216.7 | — |
| 6 | 1 | 21 | [unfused_forward](q6-d1-unfused_forward-29.json) | 29 | completed | 0.0502094 | 0.118479 | 0.007648 | 0.135225 | not reached | 216.5 | — |
| 6 | 1 | 21 | [unfused_forward](q6-d1-unfused_forward-47.json) | 47 | completed | 0.0502253 | 0.118394 | 0.007452 | 0.135083 | not reached | 216.5 | — |
| 6 | 1 | — | fused_forward | 11 | unsupported: the existing Triton HVA kernel requires CUDA complex64 | — | — | — | — | — | — | — |
| 6 | 1 | — | fused_forward | 29 | unsupported: the existing Triton HVA kernel requires CUDA complex64 | — | — | — | — | — | — | — |
| 6 | 1 | — | fused_forward | 47 | unsupported: the existing Triton HVA kernel requires CUDA complex64 | — | — | — | — | — | — | — |
| 6 | 1 | 21 | [pennylane_metric](q6-d1-pennylane_metric-11.json) | 11 | completed | 0.0502094 | 1.882710 | 0.075324 | 3.000905 | not reached | 391.2 | — |
| 6 | 1 | 21 | [pennylane_metric](q6-d1-pennylane_metric-29.json) | 29 | completed | 0.0502094 | 1.844124 | 0.071087 | 2.713370 | not reached | 408.5 | — |
| 6 | 1 | 21 | [pennylane_metric](q6-d1-pennylane_metric-47.json) | 47 | completed | 0.0502253 | 1.844360 | 0.072424 | 2.728693 | not reached | 408.5 | — |

## Separate savings

Ratios use paired median metric construction phase samples from the complete optimizer trajectories. Reverse/unfused isolates derivative algorithm; unfused/fused isolates fusion. A ratio below one is a loss. No ratio is emitted for missing, unsupported or failed pairs.

| Wires | Depth | Seed | Reverse / unfused metric time | Unfused / fused metric time |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 11 | 11.4621 | unavailable |
| 4 | 1 | 29 | 11.3094 | unavailable |
| 4 | 1 | 47 | 11.3711 | unavailable |
| 4 | 2 | 11 | 11.3494 | unavailable |
| 4 | 2 | 29 | 11.8859 | unavailable |
| 4 | 2 | 47 | 11.4612 | unavailable |
| 6 | 1 | 11 | 26.0528 | unavailable |
| 6 | 1 | 29 | 26.3914 | unavailable |
| 6 | 1 | 47 | 27.4359 | unavailable |

## Complete optimizer trajectories

Each table retains every post-update error at the fixed budget, including non-convergence. Step zero is the initial state. Full parameter vectors, directions and timings are in the linked JSON above.

<details><summary>4 wires, depth 1, seed 11</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.07282835831 | 0.07282835831 | unavailable | 0.07282835831 |
| 1 | 0.06958027887 | 0.06958027887 | unavailable | 0.06958027887 |
| 2 | 0.05573889598 | 0.05573889598 | unavailable | 0.05573889598 |
| 3 | 0.0459947002 | 0.0459947002 | unavailable | 0.0459947002 |
| 4 | 0.03809115539 | 0.03809115539 | unavailable | 0.03809115539 |
| 5 | 0.03609484441 | 0.03609484441 | unavailable | 0.03609484441 |
| 6 | 0.03563989249 | 0.03563989249 | unavailable | 0.03563989249 |
| 7 | 0.03553285879 | 0.03553285879 | unavailable | 0.03553285879 |
| 8 | 0.03550585069 | 0.03550585069 | unavailable | 0.03550585069 |
| 9 | 0.0354981648 | 0.0354981648 | unavailable | 0.0354981648 |
| 10 | 0.03549555886 | 0.03549555886 | unavailable | 0.03549555886 |
| 11 | 0.03549448287 | 0.03549448287 | unavailable | 0.03549448287 |
| 12 | 0.03549396008 | 0.03549396008 | unavailable | 0.03549396008 |
| 13 | 0.03549367862 | 0.03549367862 | unavailable | 0.03549367862 |
| 14 | 0.03549351864 | 0.03549351864 | unavailable | 0.03549351864 |
| 15 | 0.03549342529 | 0.03549342529 | unavailable | 0.03549342529 |
| 16 | 0.03549337013 | 0.03549337013 | unavailable | 0.03549337013 |
| 17 | 0.03549333736 | 0.03549333736 | unavailable | 0.03549333736 |
| 18 | 0.03549331783 | 0.03549331783 | unavailable | 0.03549331783 |
| 19 | 0.03549330618 | 0.03549330618 | unavailable | 0.03549330618 |
| 20 | 0.03549329922 | 0.03549329922 | unavailable | 0.03549329922 |
| 21 | 0.03549329506 | 0.03549329506 | unavailable | 0.03549329506 |
| 22 | 0.03549329258 | 0.03549329258 | unavailable | 0.03549329258 |
| 23 | 0.0354932911 | 0.0354932911 | unavailable | 0.0354932911 |
| 24 | 0.03549329021 | 0.03549329021 | unavailable | 0.03549329021 |
| 25 | 0.03549328968 | 0.03549328968 | unavailable | 0.03549328968 |
| 26 | 0.03549328936 | 0.03549328936 | unavailable | 0.03549328936 |
| 27 | 0.03549328917 | 0.03549328917 | unavailable | 0.03549328917 |
| 28 | 0.03549328906 | 0.03549328906 | unavailable | 0.03549328906 |
| 29 | 0.03549328899 | 0.03549328899 | unavailable | 0.03549328899 |
| 30 | 0.03549328895 | 0.03549328895 | unavailable | 0.03549328895 |

</details>

<details><summary>4 wires, depth 1, seed 29</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.07216177404 | 0.07216177404 | unavailable | 0.07216177404 |
| 1 | 0.0710020079 | 0.0710020079 | unavailable | 0.0710020079 |
| 2 | 0.05633115996 | 0.05633115996 | unavailable | 0.05633115996 |
| 3 | 0.04209778628 | 0.04209778628 | unavailable | 0.04209778628 |
| 4 | 0.03688560465 | 0.03688560465 | unavailable | 0.03688560465 |
| 5 | 0.0357872219 | 0.0357872219 | unavailable | 0.0357872219 |
| 6 | 0.03555807373 | 0.03555807373 | unavailable | 0.03555807373 |
| 7 | 0.03550931179 | 0.03550931179 | unavailable | 0.03550931179 |
| 8 | 0.03549820463 | 0.03549820463 | unavailable | 0.03549820463 |
| 9 | 0.03549525774 | 0.03549525774 | unavailable | 0.03549525774 |
| 10 | 0.03549425875 | 0.03549425875 | unavailable | 0.03549425875 |
| 11 | 0.03549382209 | 0.03549382209 | unavailable | 0.03549382209 |
| 12 | 0.03549359609 | 0.03549359609 | unavailable | 0.03549359609 |
| 13 | 0.03549346917 | 0.03549346917 | unavailable | 0.03549346917 |
| 14 | 0.03549339545 | 0.03549339545 | unavailable | 0.03549339545 |
| 15 | 0.03549335206 | 0.03549335206 | unavailable | 0.03549335206 |
| 16 | 0.03549332639 | 0.03549332639 | unavailable | 0.03549332639 |
| 17 | 0.03549331117 | 0.03549331117 | unavailable | 0.03549331117 |
| 18 | 0.03549330213 | 0.03549330213 | unavailable | 0.03549330213 |
| 19 | 0.03549329676 | 0.03549329676 | unavailable | 0.03549329676 |
| 20 | 0.03549329357 | 0.03549329357 | unavailable | 0.03549329357 |
| 21 | 0.03549329167 | 0.03549329167 | unavailable | 0.03549329167 |
| 22 | 0.03549329054 | 0.03549329054 | unavailable | 0.03549329054 |
| 23 | 0.03549328987 | 0.03549328987 | unavailable | 0.03549328987 |
| 24 | 0.03549328947 | 0.03549328947 | unavailable | 0.03549328947 |
| 25 | 0.03549328924 | 0.03549328924 | unavailable | 0.03549328924 |
| 26 | 0.0354932891 | 0.0354932891 | unavailable | 0.0354932891 |
| 27 | 0.03549328901 | 0.03549328901 | unavailable | 0.03549328901 |
| 28 | 0.03549328896 | 0.03549328896 | unavailable | 0.03549328896 |
| 29 | 0.03549328893 | 0.03549328893 | unavailable | 0.03549328893 |
| 30 | 0.03549328891 | 0.03549328891 | unavailable | 0.03549328891 |

</details>

<details><summary>4 wires, depth 1, seed 47</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.07177364784 | 0.07177364784 | unavailable | 0.07177364784 |
| 1 | 0.06867205501 | 0.06867205501 | unavailable | 0.06867205501 |
| 2 | 0.04615845659 | 0.04615845659 | unavailable | 0.04615845659 |
| 3 | 0.03797178321 | 0.03797178321 | unavailable | 0.03797178321 |
| 4 | 0.03601823226 | 0.03601823226 | unavailable | 0.03601823226 |
| 5 | 0.035603671 | 0.035603671 | unavailable | 0.035603671 |
| 6 | 0.03551683956 | 0.03551683956 | unavailable | 0.03551683956 |
| 7 | 0.03549848762 | 0.03549848762 | unavailable | 0.03549848762 |
| 8 | 0.03549450877 | 0.03549450877 | unavailable | 0.03549450877 |
| 9 | 0.0354936055 | 0.0354936055 | unavailable | 0.0354936055 |
| 10 | 0.03549338428 | 0.03549338428 | unavailable | 0.03549338428 |
| 11 | 0.0354933234 | 0.0354933234 | unavailable | 0.0354933234 |
| 12 | 0.03549330377 | 0.03549330377 | unavailable | 0.03549330377 |
| 13 | 0.03549329621 | 0.03549329621 | unavailable | 0.03549329621 |
| 14 | 0.0354932928 | 0.0354932928 | unavailable | 0.0354932928 |
| 15 | 0.03549329108 | 0.03549329108 | unavailable | 0.03549329108 |
| 16 | 0.03549329015 | 0.03549329015 | unavailable | 0.03549329015 |
| 17 | 0.03549328962 | 0.03549328962 | unavailable | 0.03549328962 |
| 18 | 0.03549328932 | 0.03549328932 | unavailable | 0.03549328932 |
| 19 | 0.03549328914 | 0.03549328914 | unavailable | 0.03549328914 |
| 20 | 0.03549328904 | 0.03549328904 | unavailable | 0.03549328904 |
| 21 | 0.03549328898 | 0.03549328898 | unavailable | 0.03549328898 |
| 22 | 0.03549328894 | 0.03549328894 | unavailable | 0.03549328894 |
| 23 | 0.03549328892 | 0.03549328892 | unavailable | 0.03549328892 |
| 24 | 0.03549328891 | 0.03549328891 | unavailable | 0.03549328891 |
| 25 | 0.0354932889 | 0.0354932889 | unavailable | 0.0354932889 |
| 26 | 0.03549328889 | 0.03549328889 | unavailable | 0.03549328889 |
| 27 | 0.03549328889 | 0.03549328889 | unavailable | 0.03549328889 |
| 28 | 0.03549328889 | 0.03549328889 | unavailable | 0.03549328889 |
| 29 | 0.03549328889 | 0.03549328889 | unavailable | 0.03549328889 |
| 30 | 0.03549328889 | 0.03549328889 | unavailable | 0.03549328889 |

</details>

<details><summary>4 wires, depth 2, seed 11</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.07235737605 | 0.07235737605 | unavailable | 0.07235737605 |
| 1 | 0.07604555311 | 0.07604555311 | unavailable | 0.07604555311 |
| 2 | 0.04409280026 | 0.04409280026 | unavailable | 0.04409280026 |
| 3 | 0.01707544933 | 0.01707544933 | unavailable | 0.01707544933 |
| 4 | 0.005705593602 | 0.005705593602 | unavailable | 0.005705593602 |
| 5 | 0.004205311492 | 0.004205311492 | unavailable | 0.004205311492 |
| 6 | 0.0005574593886 | 0.0005574593886 | unavailable | 0.0005574593886 |
| 7 | 9.858980873e-05 | 9.858980874e-05 | unavailable | 9.858980873e-05 |
| 8 | 2.858568017e-05 | 2.858568017e-05 | unavailable | 2.858568017e-05 |
| 9 | 9.497039075e-06 | 9.497039075e-06 | unavailable | 9.497039075e-06 |
| 10 | 3.576130002e-06 | 3.576130003e-06 | unavailable | 3.576130002e-06 |
| 11 | 1.639405725e-06 | 1.639405724e-06 | unavailable | 1.639405724e-06 |
| 12 | 9.376761679e-07 | 9.376761669e-07 | unavailable | 9.376761679e-07 |
| 13 | 6.366708215e-07 | 6.366708213e-07 | unavailable | 6.366708204e-07 |
| 14 | 4.783317385e-07 | 4.783317379e-07 | unavailable | 4.783317385e-07 |
| 15 | 3.793377439e-07 | 3.793377435e-07 | unavailable | 3.793377432e-07 |
| 16 | 3.103287455e-07 | 3.103287453e-07 | unavailable | 3.103287456e-07 |
| 17 | 2.593035373e-07 | 2.593035376e-07 | unavailable | 2.593035384e-07 |
| 18 | 2.203312417e-07 | 2.203312419e-07 | unavailable | 2.203312417e-07 |
| 19 | 1.899307476e-07 | 1.899307468e-07 | unavailable | 1.899307466e-07 |
| 20 | 1.658126679e-07 | 1.65812668e-07 | unavailable | 1.65812668e-07 |
| 21 | 1.463771652e-07 | 1.46377165e-07 | unavailable | 1.463771655e-07 |
| 22 | 1.30472919e-07 | 1.304729182e-07 | unavailable | 1.304729186e-07 |
| 23 | 1.1725934e-07 | 1.172593398e-07 | unavailable | 1.172593396e-07 |
| 24 | 1.061177036e-07 | 1.061177036e-07 | unavailable | 1.061177042e-07 |
| 25 | 9.658993439e-08 | 9.658993384e-08 | unavailable | 9.658993397e-08 |
| 26 | 8.833508165e-08 | 8.833508178e-08 | unavailable | 8.833508151e-08 |
| 27 | 8.10979222e-08 | 8.109792262e-08 | unavailable | 8.109792248e-08 |
| 28 | 7.468616351e-08 | 7.468616447e-08 | unavailable | 7.468616461e-08 |
| 29 | 6.895381833e-08 | 6.895381819e-08 | unavailable | 6.895381778e-08 |
| 30 | 6.378906932e-08 | 6.378906877e-08 | unavailable | 6.378906946e-08 |

</details>

<details><summary>4 wires, depth 2, seed 29</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.07253824048 | 0.07253824048 | unavailable | 0.07253824048 |
| 1 | 0.07362960228 | 0.07362960228 | unavailable | 0.07362960228 |
| 2 | 0.0452470223 | 0.0452470223 | unavailable | 0.0452470223 |
| 3 | 0.02520010842 | 0.02520010842 | unavailable | 0.02520010842 |
| 4 | 0.009689343843 | 0.009689343843 | unavailable | 0.009689343843 |
| 5 | 0.004874955429 | 0.004874955429 | unavailable | 0.004874955429 |
| 6 | 0.0008043961517 | 0.0008043961517 | unavailable | 0.0008043961517 |
| 7 | 0.0002436016544 | 0.0002436016544 | unavailable | 0.0002436016544 |
| 8 | 5.338822657e-05 | 5.338822657e-05 | unavailable | 5.338822657e-05 |
| 9 | 2.1610093e-05 | 2.1610093e-05 | unavailable | 2.1610093e-05 |
| 10 | 1.197051783e-05 | 1.197051783e-05 | unavailable | 1.197051783e-05 |
| 11 | 8.620311898e-06 | 8.620311897e-06 | unavailable | 8.620311898e-06 |
| 12 | 7.173254623e-06 | 7.173254623e-06 | unavailable | 7.173254624e-06 |
| 13 | 6.350022347e-06 | 6.350022347e-06 | unavailable | 6.350022347e-06 |
| 14 | 5.766192405e-06 | 5.766192405e-06 | unavailable | 5.766192404e-06 |
| 15 | 5.298132059e-06 | 5.29813206e-06 | unavailable | 5.298132058e-06 |
| 16 | 4.900335662e-06 | 4.900335663e-06 | unavailable | 4.900335663e-06 |
| 17 | 4.552260808e-06 | 4.552260809e-06 | unavailable | 4.552260808e-06 |
| 18 | 4.242435798e-06 | 4.242435798e-06 | unavailable | 4.242435798e-06 |
| 19 | 3.963395642e-06 | 3.963395642e-06 | unavailable | 3.963395641e-06 |
| 20 | 3.709847022e-06 | 3.709847023e-06 | unavailable | 3.709847024e-06 |
| 21 | 3.477860276e-06 | 3.477860277e-06 | unavailable | 3.477860277e-06 |
| 22 | 3.264433974e-06 | 3.264433974e-06 | unavailable | 3.264433974e-06 |
| 23 | 3.067224365e-06 | 3.067224364e-06 | unavailable | 3.067224364e-06 |
| 24 | 2.88436405e-06 | 2.884364049e-06 | unavailable | 2.884364049e-06 |
| 25 | 2.714336096e-06 | 2.714336096e-06 | unavailable | 2.714336096e-06 |
| 26 | 2.555885166e-06 | 2.555885166e-06 | unavailable | 2.555885166e-06 |
| 27 | 2.407954123e-06 | 2.407954123e-06 | unavailable | 2.407954124e-06 |
| 28 | 2.269638431e-06 | 2.269638432e-06 | unavailable | 2.269638431e-06 |
| 29 | 2.140153056e-06 | 2.140153056e-06 | unavailable | 2.140153056e-06 |
| 30 | 2.018808211e-06 | 2.018808212e-06 | unavailable | 2.018808211e-06 |

</details>

<details><summary>4 wires, depth 2, seed 47</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.07256411781 | 0.07256411781 | unavailable | 0.07256411781 |
| 1 | 0.06812403261 | 0.06812403261 | unavailable | 0.06812403261 |
| 2 | 0.02579586244 | 0.02579586244 | unavailable | 0.02579586244 |
| 3 | 0.015560393 | 0.015560393 | unavailable | 0.015560393 |
| 4 | 0.003601693913 | 0.003601693913 | unavailable | 0.003601693913 |
| 5 | 0.0005999103799 | 0.0005999103799 | unavailable | 0.0005999103799 |
| 6 | 0.0001697957891 | 0.0001697957891 | unavailable | 0.0001697957891 |
| 7 | 5.310143661e-05 | 5.310143661e-05 | unavailable | 5.310143661e-05 |
| 8 | 1.952903249e-05 | 1.952903249e-05 | unavailable | 1.952903249e-05 |
| 9 | 9.557211553e-06 | 9.557211552e-06 | unavailable | 9.557211551e-06 |
| 10 | 6.358030993e-06 | 6.358030993e-06 | unavailable | 6.358030992e-06 |
| 11 | 5.142655982e-06 | 5.142655983e-06 | unavailable | 5.142655982e-06 |
| 12 | 4.532281095e-06 | 4.532281095e-06 | unavailable | 4.532281095e-06 |
| 13 | 4.125016276e-06 | 4.125016276e-06 | unavailable | 4.125016276e-06 |
| 14 | 3.800591972e-06 | 3.800591973e-06 | unavailable | 3.800591972e-06 |
| 15 | 3.520791092e-06 | 3.52079109e-06 | unavailable | 3.520791093e-06 |
| 16 | 3.271808903e-06 | 3.271808904e-06 | unavailable | 3.271808903e-06 |
| 17 | 3.047428699e-06 | 3.047428698e-06 | unavailable | 3.047428699e-06 |
| 18 | 2.844014669e-06 | 2.844014668e-06 | unavailable | 2.844014669e-06 |
| 19 | 2.658964784e-06 | 2.658964785e-06 | unavailable | 2.658964784e-06 |
| 20 | 2.490197085e-06 | 2.490197085e-06 | unavailable | 2.490197085e-06 |
| 21 | 2.33595609e-06 | 2.33595609e-06 | unavailable | 2.335956089e-06 |
| 22 | 2.194725022e-06 | 2.194725022e-06 | unavailable | 2.194725021e-06 |
| 23 | 2.065176995e-06 | 2.065176996e-06 | unavailable | 2.065176997e-06 |
| 24 | 1.946142922e-06 | 1.946142922e-06 | unavailable | 1.946142922e-06 |
| 25 | 1.836587932e-06 | 1.836587932e-06 | unavailable | 1.836587932e-06 |
| 26 | 1.735592915e-06 | 1.735592916e-06 | unavailable | 1.735592916e-06 |
| 27 | 1.642339475e-06 | 1.642339476e-06 | unavailable | 1.642339476e-06 |
| 28 | 1.556097369e-06 | 1.556097369e-06 | unavailable | 1.556097368e-06 |
| 29 | 1.476213876e-06 | 1.476213876e-06 | unavailable | 1.476213876e-06 |
| 30 | 1.402104675e-06 | 1.402104674e-06 | unavailable | 1.402104674e-06 |

</details>

<details><summary>6 wires, depth 1, seed 11</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.09866551971 | 0.09866551971 | unavailable | 0.09866551971 |
| 1 | 0.09470956381 | 0.09470956381 | unavailable | 0.09470956381 |
| 2 | 0.08230378569 | 0.08230378569 | unavailable | 0.08230378569 |
| 3 | 0.0673109437 | 0.0673109437 | unavailable | 0.0673109437 |
| 4 | 0.05663960463 | 0.05663960463 | unavailable | 0.05663960463 |
| 5 | 0.05334886562 | 0.05334886562 | unavailable | 0.05334886562 |
| 6 | 0.05196834604 | 0.05196834604 | unavailable | 0.05196834604 |
| 7 | 0.05107268477 | 0.05107268477 | unavailable | 0.05107268477 |
| 8 | 0.05056532588 | 0.05056532588 | unavailable | 0.05056532588 |
| 9 | 0.05034903804 | 0.05034903804 | unavailable | 0.05034903804 |
| 10 | 0.05027084837 | 0.05027084837 | unavailable | 0.05027084837 |
| 11 | 0.05024226746 | 0.05024226746 | unavailable | 0.05024226746 |
| 12 | 0.05023002884 | 0.05023002884 | unavailable | 0.05023002884 |
| 13 | 0.05022354462 | 0.05022354462 | unavailable | 0.05022354462 |
| 14 | 0.05021949252 | 0.05021949252 | unavailable | 0.05021949252 |
| 15 | 0.05021671949 | 0.05021671949 | unavailable | 0.05021671949 |
| 16 | 0.0502147405 | 0.0502147405 | unavailable | 0.0502147405 |
| 17 | 0.05021330207 | 0.05021330207 | unavailable | 0.05021330207 |
| 18 | 0.05021224796 | 0.05021224796 | unavailable | 0.05021224796 |
| 19 | 0.05021147249 | 0.05021147249 | unavailable | 0.05021147249 |
| 20 | 0.05021090085 | 0.05021090085 | unavailable | 0.05021090085 |
| 21 | 0.05021047897 | 0.05021047897 | unavailable | 0.05021047897 |
| 22 | 0.05021016736 | 0.05021016736 | unavailable | 0.05021016736 |
| 23 | 0.05020993708 | 0.05020993708 | unavailable | 0.05020993708 |
| 24 | 0.05020976683 | 0.05020976683 | unavailable | 0.05020976683 |
| 25 | 0.05020964091 | 0.05020964091 | unavailable | 0.05020964091 |
| 26 | 0.05020954775 | 0.05020954775 | unavailable | 0.05020954775 |
| 27 | 0.05020947882 | 0.05020947882 | unavailable | 0.05020947882 |
| 28 | 0.05020942781 | 0.05020942781 | unavailable | 0.05020942781 |
| 29 | 0.05020939005 | 0.05020939005 | unavailable | 0.05020939005 |
| 30 | 0.05020936209 | 0.05020936209 | unavailable | 0.05020936209 |

</details>

<details><summary>6 wires, depth 1, seed 29</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.09786070087 | 0.09786070087 | unavailable | 0.09786070087 |
| 1 | 0.09681155083 | 0.09681155083 | unavailable | 0.09681155083 |
| 2 | 0.0838267594 | 0.0838267594 | unavailable | 0.0838267594 |
| 3 | 0.07002457869 | 0.07002457869 | unavailable | 0.07002457869 |
| 4 | 0.05652163042 | 0.05652163042 | unavailable | 0.05652163042 |
| 5 | 0.0535121594 | 0.0535121594 | unavailable | 0.0535121594 |
| 6 | 0.05227966879 | 0.05227966879 | unavailable | 0.05227966879 |
| 7 | 0.0513369106 | 0.0513369106 | unavailable | 0.0513369106 |
| 8 | 0.05070031296 | 0.05070031296 | unavailable | 0.05070031296 |
| 9 | 0.0503958192 | 0.0503958192 | unavailable | 0.0503958192 |
| 10 | 0.05028311187 | 0.05028311187 | unavailable | 0.05028311187 |
| 11 | 0.05024451165 | 0.05024451165 | unavailable | 0.05024451165 |
| 12 | 0.05022991468 | 0.05022991468 | unavailable | 0.05022991468 |
| 13 | 0.05022305898 | 0.05022305898 | unavailable | 0.05022305898 |
| 14 | 0.05021906884 | 0.05021906884 | unavailable | 0.05021906884 |
| 15 | 0.05021641376 | 0.05021641376 | unavailable | 0.05021641376 |
| 16 | 0.05021453144 | 0.05021453144 | unavailable | 0.05021453144 |
| 17 | 0.05021316121 | 0.05021316121 | unavailable | 0.05021316121 |
| 18 | 0.05021215312 | 0.05021215312 | unavailable | 0.05021215312 |
| 19 | 0.05021140831 | 0.05021140831 | unavailable | 0.05021140831 |
| 20 | 0.0502108571 | 0.0502108571 | unavailable | 0.0502108571 |
| 21 | 0.05021044889 | 0.05021044889 | unavailable | 0.05021044889 |
| 22 | 0.05021014651 | 0.05021014651 | unavailable | 0.05021014651 |
| 23 | 0.05020992251 | 0.05020992251 | unavailable | 0.05020992251 |
| 24 | 0.05020975657 | 0.05020975657 | unavailable | 0.05020975657 |
| 25 | 0.05020963364 | 0.05020963364 | unavailable | 0.05020963364 |
| 26 | 0.05020954257 | 0.05020954257 | unavailable | 0.05020954257 |
| 27 | 0.05020947511 | 0.05020947511 | unavailable | 0.05020947511 |
| 28 | 0.05020942513 | 0.05020942513 | unavailable | 0.05020942513 |
| 29 | 0.05020938811 | 0.05020938811 | unavailable | 0.05020938811 |
| 30 | 0.05020936069 | 0.05020936069 | unavailable | 0.05020936069 |

</details>

<details><summary>6 wires, depth 1, seed 47</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.09801955869 | 0.09801955869 | unavailable | 0.09801955869 |
| 1 | 0.09570726491 | 0.09570726491 | unavailable | 0.09570726491 |
| 2 | 0.08412746434 | 0.08412746434 | unavailable | 0.08412746434 |
| 3 | 0.07372503769 | 0.07372503769 | unavailable | 0.07372503769 |
| 4 | 0.06615699863 | 0.06615699863 | unavailable | 0.06615699863 |
| 5 | 0.05967961239 | 0.05967961239 | unavailable | 0.05967961239 |
| 6 | 0.05547858897 | 0.05547858897 | unavailable | 0.05547858897 |
| 7 | 0.05360160811 | 0.05360160811 | unavailable | 0.05360160811 |
| 8 | 0.05295885282 | 0.05295885282 | unavailable | 0.05295885282 |
| 9 | 0.05256834828 | 0.05256834828 | unavailable | 0.05256834828 |
| 10 | 0.05221082012 | 0.05221082012 | unavailable | 0.05221082012 |
| 11 | 0.05186578032 | 0.05186578032 | unavailable | 0.05186578032 |
| 12 | 0.05155850786 | 0.05155850786 | unavailable | 0.05155850786 |
| 13 | 0.05130931359 | 0.05130931359 | unavailable | 0.05130931359 |
| 14 | 0.0511175008 | 0.0511175008 | unavailable | 0.0511175008 |
| 15 | 0.05096802813 | 0.05096802813 | unavailable | 0.05096802813 |
| 16 | 0.05084456558 | 0.05084456558 | unavailable | 0.05084456558 |
| 17 | 0.05073668328 | 0.05073668328 | unavailable | 0.05073668328 |
| 18 | 0.05064028047 | 0.05064028047 | unavailable | 0.05064028047 |
| 19 | 0.05055503154 | 0.05055503154 | unavailable | 0.05055503154 |
| 20 | 0.05048173352 | 0.05048173352 | unavailable | 0.05048173352 |
| 21 | 0.05042072161 | 0.05042072161 | unavailable | 0.05042072161 |
| 22 | 0.05037140418 | 0.05037140418 | unavailable | 0.05037140418 |
| 23 | 0.05033246436 | 0.05033246436 | unavailable | 0.05033246436 |
| 24 | 0.05030225316 | 0.05030225316 | unavailable | 0.05030225316 |
| 25 | 0.05027910931 | 0.05027910931 | unavailable | 0.05027910931 |
| 26 | 0.05026153954 | 0.05026153954 | unavailable | 0.05026153954 |
| 27 | 0.05024828785 | 0.05024828785 | unavailable | 0.05024828785 |
| 28 | 0.05023833997 | 0.05023833997 | unavailable | 0.05023833997 |
| 29 | 0.05023089789 | 0.05023089789 | unavailable | 0.05023089789 |
| 30 | 0.05022534454 | 0.05022534454 | unavailable | 0.05022534454 |

</details>


All seeds, every parameter vector and update direction, per-step timings, parity errors, process wall and memory remain in the JSON. Unsupported fused CPU cases cannot establish fusion savings; a completed CUDA campaign is required for that conclusion.
