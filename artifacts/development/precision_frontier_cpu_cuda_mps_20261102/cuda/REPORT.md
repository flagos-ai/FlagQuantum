# Complete-circuit precision frontier

Source: `3ba42c605f36d86ce52c499dc216fab72f4ddcac`. Device: `cuda:0`. Smoke: `False`.

Development comparison only. No precision provider, capability maturity, vendor certification, simulation/GEMM speedup or general FP64 equivalence is established.

All routes share the same initialized FP32-representable angles and exactly represented observable coefficients. Reference: independent CPU NumPy complex128 dense matrices and analytic initial-RY tangent. Repeated Z coefficients are coalesced before reference evaluation. CPU reference and diagnostic conversion are outside candidate timing, not a fallback execution path.

Timing includes circuit construction, gate generation, state evolution, expectation and supported parameter-shift gradient. Samples are synchronized absolute seconds; the statistic is the median. First call includes preflight and is separate. Renormalization is disabled for every route. Payload memory is exact storage, not peak memory; worker RSS and CUDA allocator peak have separately stated scopes.

| Case | Mode | Status | State max error | Norm drift | Observable max error | Gradient error | Median s | First s | RSS MiB | State bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [q2-d2-noncommuting-11-fp32](q2-d2-noncommuting-11-fp32.json) | fp32 | completed | 1.4303e-07 | 4.3758e-07 | 4.0052e-07 | 6.3535e-08 | 0.037125 | 0.305024 | 1106.1 | 32 |
| [q2-d2-noncommuting-11-double_single_p4](q2-d2-noncommuting-11-double_single_p4.json) | double_single_p4 | completed | 2.5079e-15 | 4.6629e-15 | 3.1329e-15 | 3.2196e-15 | 1.088286 | 1.287623 | 1106.1 | 64 |
| [q2-d2-noncommuting-11-fp64](q2-d2-noncommuting-11-fp64.json) | fp64 | completed | 2.2591e-16 | 5.5511e-16 | 5.5511e-16 | 1.1102e-16 | 0.013704 | 0.178691 | 1106.1 | 64 |
| [q2-d2-noncommuting-11-selective_p2](q2-d2-noncommuting-11-selective_p2.json) | selective_p2 | completed | 1.4303e-07 | 4.3758e-07 | 4.1444e-07 | 2.3723e-08 | 0.058691 | 0.277543 | 1106.1 | 32 |
| [q2-d2-noncommuting-11-double_single_p3](q2-d2-noncommuting-11-double_single_p3.json) | double_single_p3 | completed | 2.5079e-15 | 4.6629e-15 | 3.1329e-15 | 1.2768e-14 | 0.807348 | 0.991807 | 1106.1 | 64 |
| [q2-d2-noncommuting-29-fp32](q2-d2-noncommuting-29-fp32.json) | fp32 | completed | 1.5272e-07 | 4.373e-07 | 4.1442e-07 | 1.1813e-07 | 0.036196 | 0.298756 | 1106.1 | 32 |
| [q2-d2-noncommuting-29-double_single_p4](q2-d2-noncommuting-29-double_single_p4.json) | double_single_p4 | completed | 3.7218e-15 | 2.8866e-15 | 5.218e-15 | 1.5543e-15 | 1.084304 | 1.340744 | 1106.1 | 64 |
| [q2-d2-noncommuting-29-fp64](q2-d2-noncommuting-29-fp64.json) | fp64 | completed | 1.1124e-16 | 8.8818e-16 | 1.8475e-16 | 2.2204e-16 | 0.013718 | 0.172026 | 1106.1 | 64 |
| [q2-d2-noncommuting-29-selective_p2](q2-d2-noncommuting-29-selective_p2.json) | selective_p2 | completed | 1.5272e-07 | 4.373e-07 | 4.3623e-07 | 1.0981e-07 | 0.058523 | 0.277154 | 1106.1 | 32 |
| [q2-d2-noncommuting-29-double_single_p3](q2-d2-noncommuting-29-double_single_p3.json) | double_single_p3 | completed | 3.7218e-15 | 2.8866e-15 | 5.218e-15 | 1.5543e-15 | 0.804870 | 1.070453 | 1106.1 | 64 |
| [q2-d2-cancellation_echo-11-fp32](q2-d2-cancellation_echo-11-fp32.json) | fp32 | completed | 1.9026e-07 | 6.1976e-07 | 6.5565e-07 | 8.2727e-08 | 0.045499 | 0.293410 | 1106.1 | 32 |
| [q2-d2-cancellation_echo-11-double_single_p4](q2-d2-cancellation_echo-11-double_single_p4.json) | double_single_p4 | completed | 2.4601e-15 | 5.9952e-15 | 9.7467e-08 | 9.6087e-08 | 1.377097 | 1.581608 | 1106.1 | 64 |
| [q2-d2-cancellation_echo-11-fp64](q2-d2-cancellation_echo-11-fp64.json) | fp64 | completed | 2.2263e-16 | 6.6613e-16 | 7.7716e-16 | 6.6804e-09 | 0.016489 | 0.159684 | 1106.1 | 64 |
| [q2-d2-cancellation_echo-11-selective_p2](q2-d2-cancellation_echo-11-selective_p2.json) | selective_p2 | completed | 1.9026e-07 | 6.1976e-07 | 6.1976e-07 | 9.2326e-08 | 0.081063 | 0.401089 | 1106.2 | 32 |
| [q2-d2-cancellation_echo-11-double_single_p3](q2-d2-cancellation_echo-11-double_single_p3.json) | double_single_p3 | completed | 2.4601e-15 | 5.9952e-15 | 9.7467e-08 | 9.6087e-08 | 0.979175 | 1.163599 | 1106.2 | 64 |
| [q2-d2-cancellation_echo-29-fp32](q2-d2-cancellation_echo-29-fp32.json) | fp32 | completed | 3.5618e-08 | 7.1282e-08 | 4.8736e-08 | 1.5858e-08 | 0.045674 | 0.309855 | 1106.2 | 32 |
| [q2-d2-cancellation_echo-29-double_single_p4](q2-d2-cancellation_echo-29-double_single_p4.json) | double_single_p4 | completed | 1.5543e-15 | 5.218e-15 | 1.0869e-08 | 1.5858e-08 | 1.387452 | 1.655294 | 1106.2 | 64 |
| [q2-d2-cancellation_echo-29-fp64](q2-d2-cancellation_echo-29-fp64.json) | fp64 | completed | 1.1249e-16 | 1.4433e-15 | 2.7756e-16 | 9.5724e-10 | 0.016718 | 0.164574 | 1106.2 | 64 |
| [q2-d2-cancellation_echo-29-selective_p2](q2-d2-cancellation_echo-29-selective_p2.json) | selective_p2 | completed | 3.5618e-08 | 7.1282e-08 | 8.4377e-08 | 1.2327e-07 | 0.084308 | 0.302234 | 1106.2 | 32 |
| [q2-d2-cancellation_echo-29-double_single_p3](q2-d2-cancellation_echo-29-double_single_p3.json) | double_single_p3 | completed | 1.6277e-15 | 3.7748e-15 | 1.0869e-08 | 1.5858e-08 | 0.984813 | 1.164464 | 1106.2 | 64 |
| [q2-d8-noncommuting-11-fp32](q2-d8-noncommuting-11-fp32.json) | fp32 | completed | 2.1509e-07 | 6.6546e-07 | 6.5888e-07 | 1.2532e-07 | 0.109016 | 0.354763 | 1106.2 | 32 |
| [q2-d8-noncommuting-11-double_single_p4](q2-d8-noncommuting-11-double_single_p4.json) | double_single_p4 | completed | 5.1548e-15 | 9.1038e-15 | 6.0507e-15 | 6.6613e-16 | 3.659698 | 3.871331 | 1106.2 | 64 |
| [q2-d8-noncommuting-11-fp64](q2-d8-noncommuting-11-fp64.json) | fp64 | completed | 3.3307e-16 | 2.7756e-15 | 6.6613e-16 | 5.5511e-16 | 0.042594 | 0.198915 | 1106.2 | 64 |
| [q2-d8-noncommuting-11-selective_p2](q2-d8-noncommuting-11-selective_p2.json) | selective_p2 | completed | 2.1509e-07 | 6.6546e-07 | 6.6559e-07 | 8.7958e-08 | 0.132482 | 0.353133 | 1106.2 | 32 |
| [q2-d8-noncommuting-11-double_single_p3](q2-d8-noncommuting-11-double_single_p3.json) | double_single_p3 | completed | 5.1548e-15 | 9.1038e-15 | 6.0507e-15 | 1.4877e-14 | 2.602834 | 2.786124 | 1106.2 | 64 |
| [q2-d8-noncommuting-29-fp32](q2-d8-noncommuting-29-fp32.json) | fp32 | completed | 1.9752e-07 | 5.4025e-07 | 5.088e-07 | 3.2703e-07 | 0.108043 | 0.378249 | 1106.2 | 32 |
| [q2-d8-noncommuting-29-double_single_p4](q2-d8-noncommuting-29-double_single_p4.json) | double_single_p4 | completed | 7.5736e-15 | 1.3323e-14 | 1.1102e-14 | 1.6653e-15 | 3.695362 | 3.953447 | 1106.2 | 64 |
| [q2-d8-noncommuting-29-fp64](q2-d8-noncommuting-29-fp64.json) | fp64 | completed | 1.1444e-16 | 1.5543e-15 | 2.2204e-16 | 6.6613e-16 | 0.043134 | 0.201579 | 1106.2 | 64 |
| [q2-d8-noncommuting-29-selective_p2](q2-d8-noncommuting-29-selective_p2.json) | selective_p2 | completed | 1.9752e-07 | 5.4025e-07 | 5.276e-07 | 3.0398e-07 | 0.129864 | 0.346986 | 1106.2 | 32 |
| [q2-d8-noncommuting-29-double_single_p3](q2-d8-noncommuting-29-double_single_p3.json) | double_single_p3 | completed | 7.5736e-15 | 6.7724e-15 | 9.4369e-15 | 1.8874e-15 | 2.597336 | 2.861184 | 1106.2 | 64 |
| [q2-d8-cancellation_echo-11-fp32](q2-d8-cancellation_echo-11-fp32.json) | fp32 | completed | 2.016e-07 | 7.1852e-07 | 7.1526e-07 | 2.153e-07 | 0.135236 | 0.373550 | 1106.2 | 32 |
| [q2-d8-cancellation_echo-11-double_single_p4](q2-d8-cancellation_echo-11-double_single_p4.json) | double_single_p4 | completed | 3.4879e-15 | 7.7716e-15 | 9.7467e-08 | 9.6087e-08 | 4.648579 | 4.885557 | 1106.2 | 64 |
| [q2-d8-cancellation_echo-11-fp64](q2-d8-cancellation_echo-11-fp64.json) | fp64 | completed | 5.5515e-16 | 3.1086e-15 | 2.1094e-15 | 6.6804e-09 | 0.049105 | 0.193985 | 1106.2 | 64 |
| [q2-d8-cancellation_echo-11-selective_p2](q2-d8-cancellation_echo-11-selective_p2.json) | selective_p2 | completed | 2.016e-07 | 7.1852e-07 | 7.1852e-07 | 2.8701e-07 | 0.173160 | 0.393148 | 1106.2 | 32 |
| [q2-d8-cancellation_echo-11-double_single_p3](q2-d8-cancellation_echo-11-double_single_p3.json) | double_single_p3 | completed | 3.4879e-15 | 7.7716e-15 | 9.7467e-08 | 3.3451e-07 | 3.089040 | 3.276530 | 1106.2 | 64 |
| [q2-d8-cancellation_echo-29-fp32](q2-d8-cancellation_echo-29-fp32.json) | fp32 | completed | 6.5814e-08 | 1.1511e-07 | 1.3814e-07 | 6.3979e-07 | 0.135469 | 0.383841 | 1106.2 | 32 |
| [q2-d8-cancellation_echo-29-double_single_p4](q2-d8-cancellation_echo-29-double_single_p4.json) | double_single_p4 | completed | 3.4436e-15 | 1.4877e-14 | 1.0869e-08 | 1.5858e-08 | 4.629541 | 4.840168 | 1106.2 | 64 |
| [q2-d8-cancellation_echo-29-fp64](q2-d8-cancellation_echo-29-fp64.json) | fp64 | completed | 8.7997e-17 | 3.5527e-15 | 6.9389e-17 | 9.5724e-10 | 0.050428 | 0.197491 | 1106.2 | 64 |
| [q2-d8-cancellation_echo-29-selective_p2](q2-d8-cancellation_echo-29-selective_p2.json) | selective_p2 | completed | 6.5814e-08 | 1.1511e-07 | 1.1511e-07 | 6.7089e-07 | 0.172543 | 0.391872 | 1106.2 | 32 |
| [q2-d8-cancellation_echo-29-double_single_p3](q2-d8-cancellation_echo-29-double_single_p3.json) | double_single_p3 | completed | 3.4797e-15 | 1.4322e-14 | 1.0869e-08 | 1.5858e-08 | 3.062750 | 3.289236 | 1106.2 | 64 |
| [q4-d8-noncommuting-11-fp32](q4-d8-noncommuting-11-fp32.json) | fp32 | completed | 1.8138e-07 | 7.4232e-07 | 5.6061e-07 | 1.6012e-07 | 0.189067 | 0.445959 | 1106.2 | 128 |
| [q4-d8-noncommuting-11-double_single_p4](q4-d8-noncommuting-11-double_single_p4.json) | double_single_p4 | completed | 8.0331e-15 | 1.3656e-14 | 1.3767e-14 | 0 | 6.277160 | 6.500438 | 1106.2 | 256 |
| [q4-d8-noncommuting-11-fp64](q4-d8-noncommuting-11-fp64.json) | fp64 | completed | 1.9478e-16 | 3.6637e-15 | 2.2204e-16 | 0 | 0.065930 | 0.256319 | 1106.2 | 256 |
| [q4-d8-noncommuting-11-selective_p2](q4-d8-noncommuting-11-selective_p2.json) | selective_p2 | completed | 1.8138e-07 | 7.4232e-07 | 5.2851e-07 | 2.3052e-07 | 0.235014 | 0.449533 | 1106.2 | 128 |
| [q4-d8-noncommuting-11-double_single_p3](q4-d8-noncommuting-11-double_single_p3.json) | double_single_p3 | completed | 8.0331e-15 | 1.2434e-14 | 1.0214e-14 | 8.8818e-15 | 4.600210 | 4.884736 | 1106.2 | 256 |
| [q4-d8-noncommuting-29-fp32](q4-d8-noncommuting-29-fp32.json) | fp32 | completed | 1.4126e-07 | 1.2325e-07 | 2.1699e-07 | 1.9278e-07 | 0.187078 | 0.449367 | 1106.2 | 128 |
| [q4-d8-noncommuting-29-double_single_p4](q4-d8-noncommuting-29-double_single_p4.json) | double_single_p4 | completed | 8.4333e-15 | 3.1086e-15 | 4.4409e-15 | 2.9976e-15 | 6.302369 | 6.514901 | 1106.2 | 256 |
| [q4-d8-noncommuting-29-fp64](q4-d8-noncommuting-29-fp64.json) | fp64 | completed | 2.2867e-16 | 2.4425e-15 | 2.498e-16 | 0 | 0.066820 | 0.258579 | 1106.2 | 256 |
| [q4-d8-noncommuting-29-selective_p2](q4-d8-noncommuting-29-selective_p2.json) | selective_p2 | completed | 1.4126e-07 | 1.2325e-07 | 2.1726e-07 | 1.9886e-07 | 0.237530 | 0.456572 | 1106.2 | 128 |
| [q4-d8-noncommuting-29-double_single_p3](q4-d8-noncommuting-29-double_single_p3.json) | double_single_p3 | completed | 8.0232e-15 | 2.6645e-15 | 1.0214e-14 | 1.4433e-15 | 4.765567 | 4.946814 | 1106.2 | 256 |
| [q4-d8-cancellation_echo-11-fp32](q4-d8-cancellation_echo-11-fp32.json) | fp32 | completed | 1.0074e-07 | 4.5174e-07 | 5.2838e-07 | 2.6154e-07 | 0.140688 | 0.401581 | 1106.2 | 128 |
| [q4-d8-cancellation_echo-11-double_single_p4](q4-d8-cancellation_echo-11-double_single_p4.json) | double_single_p4 | completed | 1.7574e-15 | 7.7716e-15 | 9.7467e-08 | 9.6087e-08 | 4.702615 | 4.939159 | 1106.2 | 256 |
| [q4-d8-cancellation_echo-11-fp64](q4-d8-cancellation_echo-11-fp64.json) | fp64 | completed | 2.2325e-16 | 3.6637e-15 | 1.4433e-15 | 6.6804e-09 | 0.050387 | 0.192713 | 1106.2 | 256 |
| [q4-d8-cancellation_echo-11-selective_p2](q4-d8-cancellation_echo-11-selective_p2.json) | selective_p2 | completed | 1.0074e-07 | 4.5174e-07 | 5.1578e-07 | 2.1306e-07 | 0.213586 | 0.424043 | 1106.2 | 128 |
| [q4-d8-cancellation_echo-11-double_single_p3](q4-d8-cancellation_echo-11-double_single_p3.json) | double_single_p3 | completed | 1.7574e-15 | 7.7716e-15 | 9.7467e-08 | 9.6087e-08 | 3.157059 | 3.464222 | 1106.2 | 256 |
| [q4-d8-cancellation_echo-29-fp32](q4-d8-cancellation_echo-29-fp32.json) | fp32 | completed | 3.303e-08 | 1.1511e-07 | 1.3814e-07 | 5.2058e-07 | 0.140968 | 0.409772 | 1106.2 | 128 |
| [q4-d8-cancellation_echo-29-double_single_p4](q4-d8-cancellation_echo-29-double_single_p4.json) | double_single_p4 | completed | 1.749e-15 | 1.4766e-14 | 1.0869e-08 | 1.5858e-08 | 4.813450 | 5.018695 | 1106.2 | 256 |
| [q4-d8-cancellation_echo-29-fp64](q4-d8-cancellation_echo-29-fp64.json) | fp64 | completed | 8.8056e-17 | 3.9968e-15 | 3.6082e-16 | 9.5724e-10 | 0.050693 | 0.194185 | 1106.2 | 256 |
| [q4-d8-cancellation_echo-29-selective_p2](q4-d8-cancellation_echo-29-selective_p2.json) | selective_p2 | completed | 3.303e-08 | 1.1511e-07 | 1.1511e-07 | 3.8405e-07 | 0.214913 | 0.428973 | 1106.2 | 128 |
| [q4-d8-cancellation_echo-29-double_single_p3](q4-d8-cancellation_echo-29-double_single_p3.json) | double_single_p3 | completed | 1.6829e-15 | 1.4322e-14 | 1.0869e-08 | 1.5858e-08 | 3.180117 | 3.364600 | 1106.2 | 256 |

## Accuracy/time/peak-memory frontier

The frontier applies to the frozen matched FP32-representable inputs, not the counterfactual intended FP64 angle. Error against that intended angle is separately retained in `quality_vs_intended_inputs`; it can exceed a chosen tolerance even when the matched-input frontier is feasible. A route is feasible only when amplitude error, norm drift, every observable error and gradient error are all within the declared dimensionless tolerance. Among feasible routes, Pareto dominance uses median complete-circuit gradient time and observed peak memory (CUDA allocator peak or CPU worker RSS). Unobservable accelerator peaks are excluded from Pareto scoring. State storage is reported separately and never substituted for a peak. These are not exclusive-system-memory claims.

| Workload | Tolerance | Feasible routes | Pareto routes |
| --- | ---: | --- | --- |
| q2-d2-cancellation_echo-11 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, double_single_p3 |
| q2-d2-cancellation_echo-11 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, double_single_p3 |
| q2-d2-cancellation_echo-11 | 1e-09 | none | none |
| q2-d2-cancellation_echo-29 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, double_single_p3 |
| q2-d2-cancellation_echo-29 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, double_single_p3 |
| q2-d2-cancellation_echo-29 | 1e-09 | fp64 | fp64 |
| q2-d2-noncommuting-11 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, double_single_p3 |
| q2-d2-noncommuting-11 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, double_single_p3 |
| q2-d2-noncommuting-11 | 1e-09 | double_single_p4, fp64, double_single_p3 | fp64, double_single_p3 |
| q2-d2-noncommuting-29 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, double_single_p3 |
| q2-d2-noncommuting-29 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, double_single_p3 |
| q2-d2-noncommuting-29 | 1e-09 | double_single_p4, fp64, double_single_p3 | fp64, double_single_p3 |
| q2-d8-cancellation_echo-11 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q2-d8-cancellation_echo-11 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q2-d8-cancellation_echo-11 | 1e-09 | none | none |
| q2-d8-cancellation_echo-29 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q2-d8-cancellation_echo-29 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q2-d8-cancellation_echo-29 | 1e-09 | fp64 | fp64 |
| q4-d8-cancellation_echo-11 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q4-d8-cancellation_echo-11 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q4-d8-cancellation_echo-11 | 1e-09 | none | none |
| q4-d8-cancellation_echo-29 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q4-d8-cancellation_echo-29 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q4-d8-cancellation_echo-29 | 1e-09 | fp64 | fp64 |
| q2-d8-noncommuting-11 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q2-d8-noncommuting-11 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q2-d8-noncommuting-11 | 1e-09 | double_single_p4, fp64, double_single_p3 | fp64, double_single_p3 |
| q2-d8-noncommuting-29 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q2-d8-noncommuting-29 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q2-d8-noncommuting-29 | 1e-09 | double_single_p4, fp64, double_single_p3 | fp64, double_single_p3 |
| q4-d8-noncommuting-11 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q4-d8-noncommuting-11 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q4-d8-noncommuting-11 | 1e-09 | double_single_p4, fp64, double_single_p3 | fp64, double_single_p3 |
| q4-d8-noncommuting-29 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q4-d8-noncommuting-29 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2, double_single_p3 |
| q4-d8-noncommuting-29 | 1e-09 | double_single_p4, fp64, double_single_p3 | fp64, double_single_p3 |

Slower and inaccurate completed cases remain in the table. Unsupported and failed routes receive no speedup score. Repeated bit identity is based on actual output dtype/shape/byte hashes; equal scalar errors alone never imply identical outputs.

Double-Single retains binary32 exponent range, is not IEEE binary64, and Dekker splitting requires finite operands small enough that multiplication by 4097 does not overflow. P4 direct angles are bounded by |theta| <= 1024; only the existing fixed and RX/RY/RZ/RXX/RYY/RZZ gates are supported. P3 uses explicit CPU complex128 gate encoding. P5 autograd/optimizer boundaries are separately probed and are not promoted by this parameter-shift benchmark.
