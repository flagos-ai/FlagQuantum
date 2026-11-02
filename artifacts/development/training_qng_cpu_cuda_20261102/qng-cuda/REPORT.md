# Matched QNG algorithm and fusion ablation

Producer: `770e24b4f43290f0e1a7d9ca212a8a4e4a396da5`. Protocol: `qng_hva_ablation_cuda_cold_cache_20261102`. Smoke: `False`.

The objective gradient, damping, layer-block projection and update rule are identical. Only metric construction changes. PennyLane supplies its framework metric, with the same FlagQuantum objective gradient; this isolates geometry and is not an end-to-end PennyLane optimizer ranking.

First-call/JIT and warmup are separate and included in total cost and total time to target. Oracle validation and common imports are outside measured total; fresh-process wall includes them and interpreter startup. Peak RSS includes the entire worker. Fixed updates continue after reaching the target. No quantum advantage, state-of-the-art or scalability claim is made.

| Wires | Depth | Parameters | Method | Seed | Status | Final error | Training s | First call s | Total s | Total target s | RSS MiB | CUDA MiB |
| ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 4 | 1 | 13 | [reverse_jacobian](q4-d1-reverse_jacobian-11.json) | 11 | completed | 0.0354931 | 2.981788 | 0.573075 | 3.784879 | not reached | 1100.9 | 16.325 |
| 4 | 1 | 13 | [reverse_jacobian](q4-d1-reverse_jacobian-29.json) | 29 | completed | 0.0354932 | 2.902070 | 0.576515 | 3.692619 | not reached | 1101.4 | 16.325 |
| 4 | 1 | 13 | [reverse_jacobian](q4-d1-reverse_jacobian-47.json) | 47 | completed | 0.035493 | 2.724057 | 0.568593 | 3.517432 | not reached | 1101.2 | 16.325 |
| 4 | 1 | 13 | [unfused_forward](q4-d1-unfused_forward-11.json) | 11 | completed | 0.0354932 | 1.093061 | 0.090964 | 1.341820 | not reached | 900.8 | 16.315 |
| 4 | 1 | 13 | [unfused_forward](q4-d1-unfused_forward-29.json) | 29 | completed | 0.0354934 | 1.098772 | 0.100624 | 1.360517 | not reached | 900.4 | 16.315 |
| 4 | 1 | 13 | [unfused_forward](q4-d1-unfused_forward-47.json) | 47 | completed | 0.0354929 | 1.082664 | 0.100391 | 1.343312 | not reached | 900.6 | 16.315 |
| 4 | 1 | 13 | [fused_forward](q4-d1-fused_forward-11.json) | 11 | completed | 0.0354931 | 0.984548 | 3.407888 | 4.650837 | not reached | 1004.9 | 16.303 |
| 4 | 1 | 13 | [fused_forward](q4-d1-fused_forward-29.json) | 29 | completed | 0.0354932 | 0.982557 | 3.355528 | 4.596144 | not reached | 1004.9 | 16.303 |
| 4 | 1 | 13 | [fused_forward](q4-d1-fused_forward-47.json) | 47 | completed | 0.0354931 | 0.997065 | 3.449195 | 4.707453 | not reached | 1005.0 | 16.303 |
| 4 | 1 | — | pennylane_metric | 11 | unsupported: PennyLane adjoint_metric_tensor/default.qubit is a CPU route | — | — | — | — | — | — | — |
| 4 | 1 | — | pennylane_metric | 29 | unsupported: PennyLane adjoint_metric_tensor/default.qubit is a CPU route | — | — | — | — | — | — | — |
| 4 | 1 | — | pennylane_metric | 47 | unsupported: PennyLane adjoint_metric_tensor/default.qubit is a CPU route | — | — | — | — | — | — | — |
| 4 | 2 | 26 | [reverse_jacobian](q4-d2-reverse_jacobian-11.json) | 11 | completed | 1.94036e-07 | 5.421014 | 0.639138 | 6.369605 | 1.658085 | 1101.8 | 16.379 |
| 4 | 2 | 26 | [reverse_jacobian](q4-d2-reverse_jacobian-29.json) | 29 | completed | 2.33328e-06 | 5.551721 | 0.656138 | 6.525566 | 1.723516 | 1101.8 | 16.379 |
| 4 | 2 | 26 | [reverse_jacobian](q4-d2-reverse_jacobian-47.json) | 47 | completed | 1.59561e-06 | 5.488984 | 0.634976 | 6.429428 | 1.657650 | 1101.6 | 16.379 |
| 4 | 2 | 26 | [unfused_forward](q4-d2-unfused_forward-11.json) | 11 | completed | 1.94036e-07 | 1.632630 | 0.110783 | 1.928634 | 0.568056 | 901.3 | 16.373 |
| 4 | 2 | 26 | [unfused_forward](q4-d2-unfused_forward-29.json) | 29 | completed | 1.59561e-06 | 1.616485 | 0.107036 | 1.906387 | 0.543485 | 901.6 | 16.373 |
| 4 | 2 | 26 | [unfused_forward](q4-d2-unfused_forward-47.json) | 47 | completed | 1.66938e-06 | 1.675096 | 0.117300 | 1.979251 | 0.569290 | 901.4 | 16.373 |
| 4 | 2 | 26 | [fused_forward](q4-d2-fused_forward-11.json) | 11 | completed | 1.74799e-07 | 1.523932 | 5.827953 | 7.632382 | 6.336855 | 1015.1 | 16.348 |
| 4 | 2 | 26 | [fused_forward](q4-d2-fused_forward-29.json) | 29 | completed | 1.96444e-06 | 1.505611 | 5.702723 | 7.494415 | 6.196703 | 1015.2 | 16.348 |
| 4 | 2 | 26 | [fused_forward](q4-d2-fused_forward-47.json) | 47 | completed | 1.59561e-06 | 1.514442 | 5.798908 | 7.591704 | 6.285891 | 1014.9 | 16.348 |
| 4 | 2 | — | pennylane_metric | 11 | unsupported: PennyLane adjoint_metric_tensor/default.qubit is a CPU route | — | — | — | — | — | — | — |
| 4 | 2 | — | pennylane_metric | 29 | unsupported: PennyLane adjoint_metric_tensor/default.qubit is a CPU route | — | — | — | — | — | — | — |
| 4 | 2 | — | pennylane_metric | 47 | unsupported: PennyLane adjoint_metric_tensor/default.qubit is a CPU route | — | — | — | — | — | — | — |
| 6 | 1 | 21 | [reverse_jacobian](q6-d1-reverse_jacobian-11.json) | 11 | completed | 0.0502094 | 10.438169 | 0.810035 | 11.748109 | not reached | 1103.2 | 16.578 |
| 6 | 1 | 21 | [reverse_jacobian](q6-d1-reverse_jacobian-29.json) | 29 | completed | 0.0502099 | 10.476526 | 0.801148 | 11.760167 | not reached | 1103.1 | 16.578 |
| 6 | 1 | 21 | [reverse_jacobian](q6-d1-reverse_jacobian-47.json) | 47 | completed | 0.0502254 | 10.713925 | 0.810001 | 12.010329 | not reached | 1103.1 | 16.578 |
| 6 | 1 | 21 | [unfused_forward](q6-d1-unfused_forward-11.json) | 11 | completed | 0.0502094 | 1.436970 | 0.110913 | 1.735810 | not reached | 902.7 | 16.408 |
| 6 | 1 | 21 | [unfused_forward](q6-d1-unfused_forward-29.json) | 29 | completed | 0.0502097 | 1.413859 | 0.102909 | 1.695702 | not reached | 902.5 | 16.408 |
| 6 | 1 | 21 | [unfused_forward](q6-d1-unfused_forward-47.json) | 47 | completed | 0.0502254 | 1.439306 | 0.100434 | 1.724649 | not reached | 902.6 | 16.408 |
| 6 | 1 | 21 | [fused_forward](q6-d1-fused_forward-11.json) | 11 | completed | 0.0502094 | 1.259663 | 5.013650 | 6.550000 | not reached | 1012.2 | 16.388 |
| 6 | 1 | 21 | [fused_forward](q6-d1-fused_forward-29.json) | 29 | completed | 0.0502098 | 1.295875 | 4.908481 | 6.473620 | not reached | 1012.6 | 16.388 |
| 6 | 1 | 21 | [fused_forward](q6-d1-fused_forward-47.json) | 47 | completed | 0.0502252 | 1.315940 | 4.857646 | 6.446637 | not reached | 1012.8 | 16.388 |
| 6 | 1 | — | pennylane_metric | 11 | unsupported: PennyLane adjoint_metric_tensor/default.qubit is a CPU route | — | — | — | — | — | — | — |
| 6 | 1 | — | pennylane_metric | 29 | unsupported: PennyLane adjoint_metric_tensor/default.qubit is a CPU route | — | — | — | — | — | — | — |
| 6 | 1 | — | pennylane_metric | 47 | unsupported: PennyLane adjoint_metric_tensor/default.qubit is a CPU route | — | — | — | — | — | — | — |

## Separate savings

Ratios use paired median metric construction phase samples from the complete optimizer trajectories. Reverse/unfused isolates derivative algorithm; unfused/fused isolates fusion. A ratio below one is a loss. No ratio is emitted for missing, unsupported or failed pairs.

| Wires | Depth | Seed | Reverse / unfused metric time | Unfused / fused metric time |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 11 | 10.9088 | 3.7287 |
| 4 | 1 | 29 | 12.5210 | 3.7572 |
| 4 | 1 | 47 | 10.8530 | 3.8702 |
| 4 | 2 | 11 | 26.9175 | 4.5393 |
| 4 | 2 | 29 | 27.7340 | 4.3815 |
| 4 | 2 | 47 | 26.3566 | 4.8048 |
| 6 | 1 | 11 | 69.7639 | 4.6188 |
| 6 | 1 | 29 | 75.7537 | 4.3013 |
| 6 | 1 | 47 | 78.4939 | 4.0989 |

## Complete optimizer trajectories

Each table retains every post-update error at the fixed budget, including non-convergence. Step zero is the initial state. Full parameter vectors, directions and timings are in the linked JSON above.

<details><summary>4 wires, depth 1, seed 11</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0728284745 | 0.0728284745 | 0.0728284745 | unavailable |
| 1 | 0.06958036749 | 0.06958021996 | 0.06958007242 | unavailable |
| 2 | 0.0557393243 | 0.05573917676 | 0.0557397669 | unavailable |
| 3 | 0.04599574095 | 0.04599529834 | 0.04599603601 | unavailable |
| 4 | 0.03809131557 | 0.03809116803 | 0.03809161064 | unavailable |
| 5 | 0.03609525526 | 0.03609503396 | 0.03609518149 | unavailable |
| 6 | 0.03564003932 | 0.03563989178 | 0.03564011308 | unavailable |
| 7 | 0.03553322475 | 0.03553322475 | 0.03553322475 | unavailable |
| 8 | 0.03550593098 | 0.03550593098 | 0.03550607851 | unavailable |
| 9 | 0.03549833298 | 0.03549833298 | 0.03549825921 | unavailable |
| 10 | 0.0354958249 | 0.03549589867 | 0.0354956036 | unavailable |
| 11 | 0.03549457086 | 0.03549457086 | 0.03549457086 | unavailable |
| 12 | 0.03549398073 | 0.03549412826 | 0.03549398073 | unavailable |
| 13 | 0.03549390696 | 0.03549368566 | 0.03549368566 | unavailable |
| 14 | 0.03549390696 | 0.03549383319 | 0.03549383319 | unavailable |
| 15 | 0.03549331682 | 0.03549324306 | 0.03549331682 | unavailable |
| 16 | 0.03549324306 | 0.03549339059 | 0.03549339059 | unavailable |
| 17 | 0.03549361189 | 0.03549346436 | 0.03549353813 | unavailable |
| 18 | 0.03549339059 | 0.03549324306 | 0.03549339059 | unavailable |
| 19 | 0.03549339059 | 0.03549331682 | 0.03549339059 | unavailable |
| 20 | 0.03549339059 | 0.03549339059 | 0.03549324306 | unavailable |
| 21 | 0.03549346436 | 0.03549339059 | 0.03549331682 | unavailable |
| 22 | 0.03549339059 | 0.03549339059 | 0.03549339059 | unavailable |
| 23 | 0.03549331682 | 0.03549324306 | 0.03549324306 | unavailable |
| 24 | 0.03549339059 | 0.03549324306 | 0.03549316929 | unavailable |
| 25 | 0.03549316929 | 0.03549309552 | 0.03549331682 | unavailable |
| 26 | 0.03549331682 | 0.03549324306 | 0.03549309552 | unavailable |
| 27 | 0.03549324306 | 0.03549316929 | 0.03549324306 | unavailable |
| 28 | 0.03549309552 | 0.03549324306 | 0.03549339059 | unavailable |
| 29 | 0.03549316929 | 0.03549324306 | 0.03549309552 | unavailable |
| 30 | 0.03549309552 | 0.03549324306 | 0.03549309552 | unavailable |

</details>

<details><summary>4 wires, depth 1, seed 29</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.07216176867 | 0.07216176867 | 0.07216176867 | unavailable |
| 1 | 0.07100215198 | 0.07100185691 | 0.07100200444 | unavailable |
| 2 | 0.05633086159 | 0.05633093536 | 0.05633226316 | unavailable |
| 3 | 0.04209759944 | 0.04209686177 | 0.04209892725 | unavailable |
| 4 | 0.03688566829 | 0.03688552076 | 0.03688581582 | unavailable |
| 5 | 0.03578713065 | 0.03578713065 | 0.03578720441 | unavailable |
| 6 | 0.03555830552 | 0.03555815799 | 0.03555815799 | unavailable |
| 7 | 0.03550947179 | 0.03550961932 | 0.03550976686 | unavailable |
| 8 | 0.03549825921 | 0.03549848051 | 0.03549855428 | unavailable |
| 9 | 0.035495161 | 0.0354953823 | 0.0354956036 | unavailable |
| 10 | 0.03549442333 | 0.03549457086 | 0.03549457086 | unavailable |
| 11 | 0.03549368566 | 0.03549383319 | 0.03549412826 | unavailable |
| 12 | 0.03549339059 | 0.03549383319 | 0.03549368566 | unavailable |
| 13 | 0.03549361189 | 0.03549383319 | 0.03549368566 | unavailable |
| 14 | 0.03549361189 | 0.03549361189 | 0.03549346436 | unavailable |
| 15 | 0.03549353813 | 0.03549368566 | 0.03549346436 | unavailable |
| 16 | 0.03549383319 | 0.03549346436 | 0.03549375943 | unavailable |
| 17 | 0.03549375943 | 0.03549353813 | 0.03549361189 | unavailable |
| 18 | 0.03549353813 | 0.03549368566 | 0.03549361189 | unavailable |
| 19 | 0.03549346436 | 0.03549339059 | 0.03549353813 | unavailable |
| 20 | 0.03549353813 | 0.03549324306 | 0.03549339059 | unavailable |
| 21 | 0.03549339059 | 0.03549324306 | 0.03549353813 | unavailable |
| 22 | 0.03549316929 | 0.03549324306 | 0.03549353813 | unavailable |
| 23 | 0.03549324306 | 0.03549331682 | 0.03549331682 | unavailable |
| 24 | 0.03549309552 | 0.03549316929 | 0.03549324306 | unavailable |
| 25 | 0.03549316929 | 0.03549339059 | 0.03549324306 | unavailable |
| 26 | 0.03549309552 | 0.03549324306 | 0.03549324306 | unavailable |
| 27 | 0.03549324306 | 0.03549324306 | 0.03549339059 | unavailable |
| 28 | 0.03549324306 | 0.03549316929 | 0.03549331682 | unavailable |
| 29 | 0.03549324306 | 0.03549339059 | 0.03549324306 | unavailable |
| 30 | 0.03549316929 | 0.03549339059 | 0.03549324306 | unavailable |

</details>

<details><summary>4 wires, depth 1, seed 47</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.07177390197 | 0.07177390197 | 0.07177390197 | unavailable |
| 1 | 0.06867126341 | 0.06867148471 | 0.06867141095 | unavailable |
| 2 | 0.04615463499 | 0.04615507759 | 0.04615537266 | unavailable |
| 3 | 0.03797100165 | 0.03797092788 | 0.03797107541 | unavailable |
| 4 | 0.03601816878 | 0.03601816878 | 0.03601816878 | unavailable |
| 5 | 0.03560374597 | 0.03560389351 | 0.03560359844 | unavailable |
| 6 | 0.03551706979 | 0.03551684849 | 0.03551699602 | unavailable |
| 7 | 0.03549862805 | 0.03549840674 | 0.03549855428 | unavailable |
| 8 | 0.03549457086 | 0.0354944971 | 0.0354944971 | unavailable |
| 9 | 0.03549375943 | 0.03549375943 | 0.03549390696 | unavailable |
| 10 | 0.03549339059 | 0.03549331682 | 0.03549339059 | unavailable |
| 11 | 0.03549361189 | 0.03549339059 | 0.03549339059 | unavailable |
| 12 | 0.03549324306 | 0.03549339059 | 0.03549331682 | unavailable |
| 13 | 0.03549353813 | 0.03549331682 | 0.03549339059 | unavailable |
| 14 | 0.03549331682 | 0.03549331682 | 0.03549368566 | unavailable |
| 15 | 0.03549368566 | 0.03549339059 | 0.03549353813 | unavailable |
| 16 | 0.03549324306 | 0.03549353813 | 0.03549339059 | unavailable |
| 17 | 0.03549346436 | 0.03549339059 | 0.03549324306 | unavailable |
| 18 | 0.03549309552 | 0.03549324306 | 0.03549316929 | unavailable |
| 19 | 0.03549316929 | 0.03549331682 | 0.03549331682 | unavailable |
| 20 | 0.03549331682 | 0.03549294799 | 0.03549309552 | unavailable |
| 21 | 0.03549302176 | 0.03549294799 | 0.03549294799 | unavailable |
| 22 | 0.03549302176 | 0.03549324306 | 0.03549294799 | unavailable |
| 23 | 0.03549302176 | 0.03549287422 | 0.03549316929 | unavailable |
| 24 | 0.03549316929 | 0.03549316929 | 0.03549309552 | unavailable |
| 25 | 0.03549309552 | 0.03549294799 | 0.03549309552 | unavailable |
| 26 | 0.03549309552 | 0.03549294799 | 0.03549309552 | unavailable |
| 27 | 0.03549302176 | 0.03549302176 | 0.03549309552 | unavailable |
| 28 | 0.03549294799 | 0.03549287422 | 0.03549316929 | unavailable |
| 29 | 0.03549294799 | 0.03549294799 | 0.03549302176 | unavailable |
| 30 | 0.03549302176 | 0.03549287422 | 0.03549309552 | unavailable |

</details>

<details><summary>4 wires, depth 2, seed 11</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.07235769373 | 0.07235769373 | 0.07235769373 | unavailable |
| 1 | 0.0760435339 | 0.07604368144 | 0.07604412404 | unavailable |
| 2 | 0.04408989764 | 0.04409078284 | 0.0440886436 | unavailable |
| 3 | 0.01706492737 | 0.01706404217 | 0.01706802558 | unavailable |
| 4 | 0.005621826912 | 0.005619392602 | 0.005633408325 | unavailable |
| 5 | 0.004286939911 | 0.004288415251 | 0.004283177796 | unavailable |
| 6 | 0.000572478151 | 0.000572551918 | 0.0005712978796 | unavailable |
| 7 | 9.933683693e-05 | 9.911553603e-05 | 9.89680021e-05 | unavailable |
| 8 | 2.785664775e-05 | 2.815171561e-05 | 2.785664775e-05 | unavailable |
| 9 | 9.414906478e-06 | 9.562440408e-06 | 9.341139513e-06 | unavailable |
| 10 | 3.882384096e-06 | 3.956151061e-06 | 3.51354927e-06 | unavailable |
| 11 | 1.595608178e-06 | 1.448074248e-06 | 1.448074248e-06 | unavailable |
| 12 | 8.57938527e-07 | 1.153006387e-06 | 1.005472457e-06 | unavailable |
| 13 | 4.891037015e-07 | 5.628706666e-07 | 5.628706666e-07 | unavailable |
| 14 | 5.628706666e-07 | 5.628706666e-07 | 6.366376317e-07 | unavailable |
| 15 | 4.891037015e-07 | 6.366376317e-07 | 4.891037015e-07 | unavailable |
| 16 | 5.628706666e-07 | 7.841715619e-07 | 1.747989844e-07 | unavailable |
| 17 | 4.153367364e-07 | 1.747989844e-07 | 1.20268876e-07 | unavailable |
| 18 | 2.726505416e-08 | 4.153367364e-07 | 1.20268876e-07 | unavailable |
| 19 | 2.726505416e-08 | 2.726505416e-08 | 2.678028062e-07 | unavailable |
| 20 | 2.678028062e-07 | 4.891037015e-07 | 4.153367364e-07 | unavailable |
| 21 | 1.747989844e-07 | 4.153367364e-07 | 5.628706666e-07 | unavailable |
| 22 | 2.678028062e-07 | 3.415697713e-07 | 5.628706666e-07 | unavailable |
| 23 | 4.153367364e-07 | 1.940358411e-07 | 1.20268876e-07 | unavailable |
| 24 | 2.678028062e-07 | 4.650191093e-08 | 1.20268876e-07 | unavailable |
| 25 | 4.153367364e-07 | 4.153367364e-07 | 1.940358411e-07 | unavailable |
| 26 | 2.678028062e-07 | 1.20268876e-07 | 2.678028062e-07 | unavailable |
| 27 | 1.747989844e-07 | 1.940358411e-07 | 1.20268876e-07 | unavailable |
| 28 | 3.223329145e-07 | 2.678028062e-07 | 4.153367364e-07 | unavailable |
| 29 | 2.678028062e-07 | 2.726505416e-08 | 4.153367364e-07 | unavailable |
| 30 | 1.940358411e-07 | 1.940358411e-07 | 1.747989844e-07 | unavailable |

</details>

<details><summary>4 wires, depth 2, seed 29</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.07253871786 | 0.07253871786 | 0.07253871786 | unavailable |
| 1 | 0.07362818217 | 0.07362840347 | 0.07362825593 | unavailable |
| 2 | 0.04524021969 | 0.04523453963 | 0.04523785915 | unavailable |
| 3 | 0.02518777673 | 0.02519271912 | 0.02518755543 | unavailable |
| 4 | 0.009674510207 | 0.009677755953 | 0.00967613308 | unavailable |
| 5 | 0.004888657046 | 0.004892640462 | 0.00488637027 | unavailable |
| 6 | 0.0008063194304 | 0.0008073521679 | 0.0008059505956 | unavailable |
| 7 | 0.0002421496813 | 0.0002424447492 | 0.0002430348849 | unavailable |
| 8 | 5.367508553e-05 | 5.382261946e-05 | 5.330625071e-05 | unavailable |
| 9 | 2.188152358e-05 | 2.143892179e-05 | 2.121762089e-05 | unavailable |
| 10 | 1.214428419e-05 | 1.221805115e-05 | 1.221805115e-05 | unavailable |
| 11 | 8.824770757e-06 | 8.529702897e-06 | 8.529702897e-06 | unavailable |
| 12 | 7.12813056e-06 | 7.349431455e-06 | 6.906829665e-06 | unavailable |
| 13 | 6.316693944e-06 | 6.611761804e-06 | 6.611761804e-06 | unavailable |
| 14 | 5.652791258e-06 | 5.800325188e-06 | 6.021626083e-06 | unavailable |
| 15 | 5.505257328e-06 | 4.988888572e-06 | 5.431490363e-06 | unavailable |
| 16 | 5.210189467e-06 | 4.841354642e-06 | 5.136422502e-06 | unavailable |
| 17 | 4.693820712e-06 | 4.620053747e-06 | 4.546286782e-06 | unavailable |
| 18 | 4.546286782e-06 | 4.693820712e-06 | 3.956151061e-06 | unavailable |
| 19 | 4.029918026e-06 | 3.956151061e-06 | 4.251218921e-06 | unavailable |
| 20 | 3.587316235e-06 | 3.956151061e-06 | 3.587316235e-06 | unavailable |
| 21 | 3.587316235e-06 | 3.6610832e-06 | 3.36601534e-06 | unavailable |
| 22 | 3.51354927e-06 | 3.51354927e-06 | 3.144714445e-06 | unavailable |
| 23 | 2.849646584e-06 | 2.775879619e-06 | 3.36601534e-06 | unavailable |
| 24 | 3.36601534e-06 | 3.21848141e-06 | 2.775879619e-06 | unavailable |
| 25 | 2.702112654e-06 | 2.92341355e-06 | 2.92341355e-06 | unavailable |
| 26 | 2.92341355e-06 | 2.775879619e-06 | 2.775879619e-06 | unavailable |
| 27 | 2.480811759e-06 | 2.407044794e-06 | 2.333277829e-06 | unavailable |
| 28 | 2.775879619e-06 | 2.702112654e-06 | 2.333277829e-06 | unavailable |
| 29 | 2.259510864e-06 | 2.185743899e-06 | 2.038209968e-06 | unavailable |
| 30 | 2.333277829e-06 | 1.595608178e-06 | 1.964443003e-06 | unavailable |

</details>

<details><summary>4 wires, depth 2, seed 47</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.07256438876 | 0.07256438876 | 0.07256438876 | unavailable |
| 1 | 0.0681233224 | 0.06812369123 | 0.06812369123 | unavailable |
| 2 | 0.02579465755 | 0.02579613289 | 0.02579605913 | unavailable |
| 3 | 0.0155651712 | 0.01556199922 | 0.01555926985 | unavailable |
| 4 | 0.003603341446 | 0.003601866107 | 0.003600980903 | unavailable |
| 5 | 0.0005996981612 | 0.0006002882969 | 0.0005997719281 | unavailable |
| 6 | 0.0001697842886 | 0.0001696367547 | 0.0001702268904 | unavailable |
| 7 | 5.345378464e-05 | 5.323248374e-05 | 5.338001767e-05 | unavailable |
| 8 | 1.988981552e-05 | 1.944721373e-05 | 1.915214587e-05 | unavailable |
| 9 | 1.00050422e-05 | 9.119838617e-06 | 9.857508268e-06 | unavailable |
| 10 | 6.464227874e-06 | 6.611761804e-06 | 6.242926979e-06 | unavailable |
| 11 | 5.283956433e-06 | 5.431490363e-06 | 5.283956433e-06 | unavailable |
| 12 | 4.988888572e-06 | 4.398752851e-06 | 4.546286782e-06 | unavailable |
| 13 | 4.472519817e-06 | 4.251218921e-06 | 4.177451956e-06 | unavailable |
| 14 | 4.251218921e-06 | 4.251218921e-06 | 3.51354927e-06 | unavailable |
| 15 | 3.808617131e-06 | 3.882384096e-06 | 3.808617131e-06 | unavailable |
| 16 | 2.997180515e-06 | 3.36601534e-06 | 3.36601534e-06 | unavailable |
| 17 | 2.92341355e-06 | 3.36601534e-06 | 2.92341355e-06 | unavailable |
| 18 | 2.849646584e-06 | 2.92341355e-06 | 3.07094748e-06 | unavailable |
| 19 | 2.92341355e-06 | 2.628345689e-06 | 2.775879619e-06 | unavailable |
| 20 | 2.333277829e-06 | 2.480811759e-06 | 2.628345689e-06 | unavailable |
| 21 | 2.333277829e-06 | 2.185743899e-06 | 2.333277829e-06 | unavailable |
| 22 | 1.964443003e-06 | 2.038209968e-06 | 2.111976934e-06 | unavailable |
| 23 | 2.038209968e-06 | 2.333277829e-06 | 2.333277829e-06 | unavailable |
| 24 | 2.038209968e-06 | 1.964443003e-06 | 2.480811759e-06 | unavailable |
| 25 | 2.185743899e-06 | 1.743142108e-06 | 1.743142108e-06 | unavailable |
| 26 | 1.448074248e-06 | 1.743142108e-06 | 1.890676038e-06 | unavailable |
| 27 | 1.816909073e-06 | 1.743142108e-06 | 1.153006387e-06 | unavailable |
| 28 | 1.890676038e-06 | 1.743142108e-06 | 1.448074248e-06 | unavailable |
| 29 | 1.595608178e-06 | 1.521841213e-06 | 1.816909073e-06 | unavailable |
| 30 | 1.595608178e-06 | 1.669375143e-06 | 1.595608178e-06 | unavailable |

</details>

<details><summary>6 wires, depth 1, seed 11</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.09866586221 | 0.09866586221 | 0.09866586221 | unavailable |
| 1 | 0.09470948875 | 0.09470977559 | 0.09470948875 | unavailable |
| 2 | 0.08230053696 | 0.08229575631 | 0.08230149309 | unavailable |
| 3 | 0.06731031896 | 0.06730725934 | 0.06731012774 | unavailable |
| 4 | 0.05664028217 | 0.05663913482 | 0.05663970849 | unavailable |
| 5 | 0.05334928011 | 0.05334889766 | 0.05334880204 | unavailable |
| 6 | 0.05196843607 | 0.05196824484 | 0.05196824484 | unavailable |
| 7 | 0.05107273278 | 0.051072924 | 0.05107282839 | unavailable |
| 8 | 0.05056579225 | 0.05056483612 | 0.05056560103 | unavailable |
| 9 | 0.0503489418 | 0.05034875057 | 0.05034951547 | unavailable |
| 10 | 0.0502707303 | 0.0502707303 | 0.05027092153 | unavailable |
| 11 | 0.05024242883 | 0.05024242883 | 0.05024262006 | unavailable |
| 12 | 0.05022999913 | 0.05023028597 | 0.0502298079 | unavailable |
| 13 | 0.05022340183 | 0.05022349744 | 0.05022330621 | unavailable |
| 14 | 0.05021967292 | 0.05021967292 | 0.0502195773 | unavailable |
| 15 | 0.05021690014 | 0.05021680453 | 0.05021699575 | unavailable |
| 16 | 0.05021479665 | 0.05021450981 | 0.05021450981 | unavailable |
| 17 | 0.05021307562 | 0.05021298 | 0.05021326684 | unavailable |
| 18 | 0.05021259755 | 0.05021259755 | 0.05021259755 | unavailable |
| 19 | 0.05021164142 | 0.05021173703 | 0.05021164142 | unavailable |
| 20 | 0.05021125897 | 0.05021097213 | 0.05021125897 | unavailable |
| 21 | 0.05021049406 | 0.05021068529 | 0.05021068529 | unavailable |
| 22 | 0.05021030284 | 0.05021030284 | 0.05021011161 | unavailable |
| 23 | 0.05021011161 | 0.05021011161 | 0.05021020722 | unavailable |
| 24 | 0.05020992038 | 0.05020972916 | 0.05020982477 | unavailable |
| 25 | 0.05020982477 | 0.05020963354 | 0.05020972916 | unavailable |
| 26 | 0.05020963354 | 0.05020972916 | 0.05020963354 | unavailable |
| 27 | 0.05020953793 | 0.05020934671 | 0.05020972916 | unavailable |
| 28 | 0.05020953793 | 0.05020953793 | 0.05020972916 | unavailable |
| 29 | 0.05020953793 | 0.05020963354 | 0.05020953793 | unavailable |
| 30 | 0.05020944232 | 0.05020944232 | 0.05020944232 | unavailable |

</details>

<details><summary>6 wires, depth 1, seed 29</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.09786099134 | 0.09786099134 | 0.09786099134 | unavailable |
| 1 | 0.09681125538 | 0.09681125538 | 0.09681115977 | unavailable |
| 2 | 0.08382977249 | 0.08382709532 | 0.08382919881 | unavailable |
| 3 | 0.07003634336 | 0.07003318813 | 0.0700373951 | unavailable |
| 4 | 0.05652382545 | 0.05652334738 | 0.05652439913 | unavailable |
| 5 | 0.05351354337 | 0.05351316092 | 0.05351316092 | unavailable |
| 6 | 0.05228080399 | 0.05228042154 | 0.05228109083 | unavailable |
| 7 | 0.05133738977 | 0.05133748538 | 0.05133786784 | unavailable |
| 8 | 0.05070060669 | 0.05070051108 | 0.05070118037 | unavailable |
| 9 | 0.05039617466 | 0.05039598343 | 0.05039598343 | unavailable |
| 10 | 0.05028316 | 0.05028306439 | 0.05028316 | unavailable |
| 11 | 0.0502444367 | 0.05024481916 | 0.05024453232 | unavailable |
| 12 | 0.05023019036 | 0.05023038158 | 0.05023019036 | unavailable |
| 13 | 0.05022330621 | 0.05022349744 | 0.05022330621 | unavailable |
| 14 | 0.05021919485 | 0.05021890801 | 0.05021909924 | unavailable |
| 15 | 0.05021642207 | 0.05021709136 | 0.0502166133 | unavailable |
| 16 | 0.05021489226 | 0.05021479665 | 0.05021479665 | unavailable |
| 17 | 0.05021345807 | 0.05021355368 | 0.05021326684 | unavailable |
| 18 | 0.05021231071 | 0.05021259755 | 0.0502122151 | unavailable |
| 19 | 0.05021183265 | 0.05021145019 | 0.05021145019 | unavailable |
| 20 | 0.05021087651 | 0.05021116335 | 0.0502107809 | unavailable |
| 21 | 0.05021068529 | 0.05021068529 | 0.05021058968 | unavailable |
| 22 | 0.05021011161 | 0.05021049406 | 0.05021030284 | unavailable |
| 23 | 0.05021011161 | 0.05021039845 | 0.05021011161 | unavailable |
| 24 | 0.050210016 | 0.05020992038 | 0.05021011161 | unavailable |
| 25 | 0.05020992038 | 0.05021011161 | 0.05021020722 | unavailable |
| 26 | 0.05020953793 | 0.05021030284 | 0.05021011161 | unavailable |
| 27 | 0.050210016 | 0.05020992038 | 0.050210016 | unavailable |
| 28 | 0.05020963354 | 0.05020992038 | 0.05020972916 | unavailable |
| 29 | 0.05020953793 | 0.05020972916 | 0.05020972916 | unavailable |
| 30 | 0.05020992038 | 0.05020972916 | 0.05020982477 | unavailable |

</details>

<details><summary>6 wires, depth 1, seed 47</summary>

| Step | Reverse error | Unfused error | Fused error | Framework error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.09801951782 | 0.09801951782 | 0.09801951782 | unavailable |
| 1 | 0.09570730681 | 0.09570778488 | 0.09570778488 | unavailable |
| 2 | 0.08412579057 | 0.0841223485 | 0.08412502567 | unavailable |
| 3 | 0.07372232304 | 0.07372194059 | 0.07372404407 | unavailable |
| 4 | 0.06615387881 | 0.06615607791 | 0.0661567472 | unavailable |
| 5 | 0.05967513681 | 0.05967991746 | 0.05967905694 | unavailable |
| 6 | 0.05547581052 | 0.05547867892 | 0.05547839208 | unavailable |
| 7 | 0.05360055127 | 0.05360208108 | 0.05360141179 | unavailable |
| 8 | 0.05295860508 | 0.05295879631 | 0.05295889192 | unavailable |
| 9 | 0.05256812128 | 0.05256850373 | 0.05256831251 | unavailable |
| 10 | 0.0522107196 | 0.0522107196 | 0.05221100644 | unavailable |
| 11 | 0.0518655564 | 0.05186593885 | 0.05186574763 | unavailable |
| 12 | 0.05155854282 | 0.05155825598 | 0.05155863843 | unavailable |
| 13 | 0.05130927953 | 0.05130918391 | 0.05130947075 | unavailable |
| 14 | 0.0511174797 | 0.0511174797 | 0.05111767092 | unavailable |
| 15 | 0.05096822769 | 0.05096813207 | 0.0509683233 | unavailable |
| 16 | 0.05084459998 | 0.05084440875 | 0.05084421753 | unavailable |
| 17 | 0.05073693966 | 0.0507365572 | 0.05073684404 | unavailable |
| 18 | 0.05064056168 | 0.05064027484 | 0.0506407529 | unavailable |
| 19 | 0.05055546604 | 0.05055527481 | 0.05055508359 | unavailable |
| 20 | 0.05048184397 | 0.0504820352 | 0.05048222642 | unavailable |
| 21 | 0.05042084283 | 0.05042084283 | 0.05042122528 | unavailable |
| 22 | 0.05037169771 | 0.05037150648 | 0.05037188893 | unavailable |
| 23 | 0.05033240073 | 0.05033259196 | 0.05033268757 | unavailable |
| 24 | 0.05030247384 | 0.05030247384 | 0.05030237823 | unavailable |
| 25 | 0.05027914425 | 0.0502787618 | 0.05027933548 | unavailable |
| 26 | 0.05026136022 | 0.05026136022 | 0.05026107338 | unavailable |
| 27 | 0.05024816561 | 0.05024826123 | 0.05024816561 | unavailable |
| 28 | 0.05023831747 | 0.05023831747 | 0.05023822185 | unavailable |
| 29 | 0.0502312421 | 0.05023114649 | 0.05023114649 | unavailable |
| 30 | 0.0502254097 | 0.0502254097 | 0.05022521848 | unavailable |

</details>


All seeds, every parameter vector and update direction, per-step timings, parity errors, process wall and memory remain in the JSON. Unsupported fused CPU cases cannot establish fusion savings; a completed CUDA campaign is required for that conclusion.
