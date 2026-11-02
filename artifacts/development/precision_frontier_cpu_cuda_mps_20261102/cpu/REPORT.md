# Complete-circuit precision frontier

Source: `3ba42c605f36d86ce52c499dc216fab72f4ddcac`. Device: `cpu`. Smoke: `False`.

Development comparison only. No precision provider, capability maturity, vendor certification, simulation/GEMM speedup or general FP64 equivalence is established.

All routes share the same initialized FP32-representable angles and exactly represented observable coefficients. Reference: independent CPU NumPy complex128 dense matrices and analytic initial-RY tangent. Repeated Z coefficients are coalesced before reference evaluation. CPU reference and diagnostic conversion are outside candidate timing, not a fallback execution path.

Timing includes circuit construction, gate generation, state evolution, expectation and supported parameter-shift gradient. Samples are synchronized absolute seconds; the statistic is the median. First call includes preflight and is separate. Renormalization is disabled for every route. Payload memory is exact storage, not peak memory; worker RSS and CUDA allocator peak have separately stated scopes.

| Case | Mode | Status | State max error | Norm drift | Observable max error | Gradient error | Median s | First s | RSS MiB | State bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [q2-d2-noncommuting-11-fp32](q2-d2-noncommuting-11-fp32.json) | fp32 | completed | 9.467e-08 | 3.1625e-07 | 2.8131e-07 | 1.1528e-07 | 0.003889 | 0.018418 | 212.2 | 32 |
| [q2-d2-noncommuting-11-double_single_p4](q2-d2-noncommuting-11-double_single_p4.json) | double_single_p4 | completed | 2.5014e-15 | 4.6629e-15 | 3.178e-15 | 2.8866e-15 | 0.076287 | 0.089690 | 208.9 | 64 |
| [q2-d2-noncommuting-11-fp64](q2-d2-noncommuting-11-fp64.json) | fp64 | completed | 1.2118e-16 | 8.8818e-16 | 2.2204e-16 | 2.2204e-16 | 0.001635 | 0.006019 | 208.6 | 64 |
| [q2-d2-noncommuting-11-selective_p2](q2-d2-noncommuting-11-selective_p2.json) | selective_p2 | completed | 9.467e-08 | 3.1625e-07 | 2.9688e-07 | 8.0334e-08 | 0.005295 | 0.014611 | 208.5 | 32 |
| [q2-d2-noncommuting-11-double_single_p3](q2-d2-noncommuting-11-double_single_p3.json) | double_single_p3 | completed | 2.5014e-15 | 4.6629e-15 | 3.178e-15 | 1.3101e-14 | 0.057721 | 0.066100 | 209.1 | 64 |
| [q2-d2-noncommuting-29-fp32](q2-d2-noncommuting-29-fp32.json) | fp32 | completed | 1.5305e-07 | 4.378e-07 | 4.1442e-07 | 2.3734e-07 | 0.003766 | 0.018641 | 212.2 | 32 |
| [q2-d2-noncommuting-29-double_single_p4](q2-d2-noncommuting-29-double_single_p4.json) | double_single_p4 | completed | 3.723e-15 | 2.8866e-15 | 4.996e-15 | 1.4433e-15 | 0.076309 | 0.083572 | 209.0 | 64 |
| [q2-d2-noncommuting-29-fp64](q2-d2-noncommuting-29-fp64.json) | fp64 | completed | 2.2215e-16 | 6.6613e-16 | 4.4235e-16 | 3.3307e-16 | 0.001598 | 0.004404 | 208.3 | 64 |
| [q2-d2-noncommuting-29-selective_p2](q2-d2-noncommuting-29-selective_p2.json) | selective_p2 | completed | 1.5305e-07 | 4.378e-07 | 4.357e-07 | 3.1263e-07 | 0.005376 | 0.014016 | 208.8 | 32 |
| [q2-d2-noncommuting-29-double_single_p3](q2-d2-noncommuting-29-double_single_p3.json) | double_single_p3 | completed | 3.723e-15 | 2.8866e-15 | 4.996e-15 | 1.4433e-15 | 0.056388 | 0.065249 | 209.1 | 64 |
| [q2-d2-cancellation_echo-11-fp32](q2-d2-cancellation_echo-11-fp32.json) | fp32 | completed | 8.2784e-08 | 2.9868e-07 | 0.44354 | 0.89626 | 0.004559 | 0.020494 | 212.2 | 32 |
| [q2-d2-cancellation_echo-11-double_single_p4](q2-d2-cancellation_echo-11-double_single_p4.json) | double_single_p4 | completed | 2.4664e-15 | 5.9952e-15 | 9.7467e-08 | 9.6087e-08 | 0.100940 | 0.111369 | 209.2 | 64 |
| [q2-d2-cancellation_echo-11-fp64](q2-d2-cancellation_echo-11-fp64.json) | fp64 | completed | 1.1597e-16 | 1.1102e-15 | 3.3307e-16 | 6.6804e-09 | 0.002080 | 0.005199 | 208.4 | 64 |
| [q2-d2-cancellation_echo-11-selective_p2](q2-d2-cancellation_echo-11-selective_p2.json) | selective_p2 | completed | 8.2784e-08 | 2.9868e-07 | 2.9868e-07 | 2.3122e-08 | 0.007795 | 0.018914 | 208.8 | 32 |
| [q2-d2-cancellation_echo-11-double_single_p3](q2-d2-cancellation_echo-11-double_single_p3.json) | double_single_p3 | completed | 2.4664e-15 | 5.9952e-15 | 9.7467e-08 | 9.6087e-08 | 0.075609 | 0.085463 | 209.4 | 64 |
| [q2-d2-cancellation_echo-29-fp32](q2-d2-cancellation_echo-29-fp32.json) | fp32 | completed | 3.5618e-08 | 7.1282e-08 | 0.45922 | 0.88832 | 0.004708 | 0.021276 | 212.2 | 32 |
| [q2-d2-cancellation_echo-29-double_single_p4](q2-d2-cancellation_echo-29-double_single_p4.json) | double_single_p4 | completed | 1.5543e-15 | 5.3291e-15 | 1.0869e-08 | 1.5858e-08 | 0.105382 | 0.115458 | 209.5 | 64 |
| [q2-d2-cancellation_echo-29-fp64](q2-d2-cancellation_echo-29-fp64.json) | fp64 | completed | 1.6678e-16 | 1.1102e-15 | 2.7756e-16 | 9.5724e-10 | 0.002042 | 0.004531 | 208.5 | 64 |
| [q2-d2-cancellation_echo-29-selective_p2](q2-d2-cancellation_echo-29-selective_p2.json) | selective_p2 | completed | 3.5618e-08 | 7.1282e-08 | 8.4377e-08 | 1.3507e-07 | 0.007841 | 0.017038 | 208.7 | 32 |
| [q2-d2-cancellation_echo-29-double_single_p3](q2-d2-cancellation_echo-29-double_single_p3.json) | double_single_p3 | completed | 1.7351e-15 | 3.7748e-15 | 1.0869e-08 | 1.5858e-08 | 0.075242 | 0.085675 | 209.2 | 64 |
| [q2-d8-noncommuting-11-fp32](q2-d8-noncommuting-11-fp32.json) | fp32 | completed | 2.0036e-07 | 4.2036e-07 | 3.6086e-07 | 1.2532e-07 | 0.010223 | 0.026775 | 212.3 | 32 |
| [q2-d8-noncommuting-11-double_single_p4](q2-d8-noncommuting-11-double_single_p4.json) | double_single_p4 | completed | 5.0843e-15 | 9.1038e-15 | 6.4948e-15 | 6.6613e-16 | 0.275246 | 0.263593 | 209.2 | 64 |
| [q2-d8-noncommuting-11-fp64](q2-d8-noncommuting-11-fp64.json) | fp64 | completed | 2.8305e-16 | 3.3307e-15 | 1.1102e-16 | 0 | 0.004877 | 0.011003 | 196.1 | 64 |
| [q2-d8-noncommuting-11-selective_p2](q2-d8-noncommuting-11-selective_p2.json) | selective_p2 | completed | 2.0036e-07 | 4.2036e-07 | 4.5011e-07 | 1.7347e-07 | 0.012673 | 0.022235 | 209.0 | 32 |
| [q2-d8-noncommuting-11-double_single_p3](q2-d8-noncommuting-11-double_single_p3.json) | double_single_p3 | completed | 5.0843e-15 | 9.1038e-15 | 6.4948e-15 | 1.4877e-14 | 0.196713 | 0.201877 | 209.1 | 64 |
| [q2-d8-noncommuting-29-fp32](q2-d8-noncommuting-29-fp32.json) | fp32 | completed | 2.5674e-07 | 5.7513e-07 | 5.088e-07 | 6.8466e-07 | 0.010608 | 0.025597 | 212.2 | 32 |
| [q2-d8-noncommuting-29-double_single_p4](q2-d8-noncommuting-29-double_single_p4.json) | double_single_p4 | completed | 7.4488e-15 | 1.3323e-14 | 1.1213e-14 | 1.7764e-15 | 0.265266 | 0.275940 | 209.7 | 64 |
| [q2-d8-noncommuting-29-fp64](q2-d8-noncommuting-29-fp64.json) | fp64 | completed | 3.3766e-16 | 1.3323e-15 | 6.3838e-16 | 4.4409e-16 | 0.004675 | 0.007609 | 208.5 | 64 |
| [q2-d8-noncommuting-29-selective_p2](q2-d8-noncommuting-29-selective_p2.json) | selective_p2 | completed | 2.5674e-07 | 5.7513e-07 | 5.6608e-07 | 7.5625e-07 | 0.013091 | 0.023190 | 209.1 | 32 |
| [q2-d8-noncommuting-29-double_single_p3](q2-d8-noncommuting-29-double_single_p3.json) | double_single_p3 | completed | 7.4488e-15 | 6.6613e-15 | 9.2981e-15 | 1.7764e-15 | 0.194006 | 0.202281 | 209.3 | 64 |
| [q2-d8-cancellation_echo-11-fp32](q2-d8-cancellation_echo-11-fp32.json) | fp32 | completed | 6.6365e-08 | 2.24e-08 | 0.44354 | 0.89626 | 0.013118 | 0.028890 | 212.4 | 32 |
| [q2-d8-cancellation_echo-11-double_single_p4](q2-d8-cancellation_echo-11-double_single_p4.json) | double_single_p4 | completed | 3.5376e-15 | 7.7716e-15 | 9.7467e-08 | 9.6087e-08 | 0.340003 | 0.353789 | 209.8 | 64 |
| [q2-d8-cancellation_echo-11-fp64](q2-d8-cancellation_echo-11-fp64.json) | fp64 | completed | 6.7249e-16 | 3.3307e-15 | 2.2204e-15 | 6.6804e-09 | 0.005544 | 0.008194 | 208.5 | 64 |
| [q2-d8-cancellation_echo-11-selective_p2](q2-d8-cancellation_echo-11-selective_p2.json) | selective_p2 | completed | 6.6365e-08 | 2.24e-08 | 1.5215e-07 | 3.8075e-07 | 0.016313 | 0.026472 | 209.1 | 32 |
| [q2-d8-cancellation_echo-11-double_single_p3](q2-d8-cancellation_echo-11-double_single_p3.json) | double_single_p3 | completed | 3.5376e-15 | 7.7716e-15 | 9.7467e-08 | 3.3451e-07 | 0.233529 | 0.237765 | 209.2 | 64 |
| [q2-d8-cancellation_echo-29-fp32](q2-d8-cancellation_echo-29-fp32.json) | fp32 | completed | 1.9025e-07 | 5.9095e-07 | 0.45922 | 0.88832 | 0.013310 | 0.028439 | 212.4 | 32 |
| [q2-d8-cancellation_echo-29-double_single_p4](q2-d8-cancellation_echo-29-double_single_p4.json) | double_single_p4 | completed | 3.4436e-15 | 1.4766e-14 | 1.0869e-08 | 1.5858e-08 | 0.337512 | 0.345178 | 209.5 | 64 |
| [q2-d8-cancellation_echo-29-fp64](q2-d8-cancellation_echo-29-fp64.json) | fp64 | completed | 3.3353e-16 | 3.3307e-15 | 6.6613e-16 | 9.5724e-10 | 0.005341 | 0.008100 | 208.4 | 64 |
| [q2-d8-cancellation_echo-29-selective_p2](q2-d8-cancellation_echo-29-selective_p2.json) | selective_p2 | completed | 1.9025e-07 | 5.9095e-07 | 5.9095e-07 | 5.8019e-07 | 0.016591 | 0.026545 | 208.8 | 32 |
| [q2-d8-cancellation_echo-29-double_single_p3](q2-d8-cancellation_echo-29-double_single_p3.json) | double_single_p3 | completed | 3.4797e-15 | 1.4433e-14 | 1.0869e-08 | 1.5858e-08 | 0.234103 | 0.239336 | 209.5 | 64 |
| [q4-d8-noncommuting-11-fp32](q4-d8-noncommuting-11-fp32.json) | fp32 | completed | 1.8931e-07 | 7.2813e-07 | 4.7795e-07 | 1.0051e-07 | 0.018152 | 0.053302 | 212.5 | 128 |
| [q4-d8-noncommuting-11-double_single_p4](q4-d8-noncommuting-11-double_single_p4.json) | double_single_p4 | completed | 8.0331e-15 | 1.3656e-14 | 1.3878e-14 | 0 | 0.460295 | 0.504941 | 209.8 | 256 |
| [q4-d8-noncommuting-11-fp64](q4-d8-noncommuting-11-fp64.json) | fp64 | completed | 2.6658e-16 | 3.9968e-15 | 3.3307e-16 | 0 | 0.007548 | 0.012421 | 209.0 | 256 |
| [q4-d8-noncommuting-11-selective_p2](q4-d8-noncommuting-11-selective_p2.json) | selective_p2 | completed | 1.8931e-07 | 7.2813e-07 | 5.5835e-07 | 1.4315e-07 | 0.022269 | 0.030826 | 209.1 | 128 |
| [q4-d8-noncommuting-11-double_single_p3](q4-d8-noncommuting-11-double_single_p3.json) | double_single_p3 | completed | 8.0331e-15 | 1.2434e-14 | 1.0325e-14 | 8.8818e-15 | 0.352251 | 0.358469 | 209.4 | 256 |
| [q4-d8-noncommuting-29-fp32](q4-d8-noncommuting-29-fp32.json) | fp32 | completed | 1.4126e-07 | 1.2325e-07 | 2.2444e-07 | 2.5238e-07 | 0.017770 | 0.050249 | 212.4 | 128 |
| [q4-d8-noncommuting-29-double_single_p4](q4-d8-noncommuting-29-double_single_p4.json) | double_single_p4 | completed | 8.4333e-15 | 3.1086e-15 | 4.4825e-15 | 3.2196e-15 | 0.468081 | 0.527342 | 209.8 | 256 |
| [q4-d8-noncommuting-29-fp64](q4-d8-noncommuting-29-fp64.json) | fp64 | completed | 2.2308e-16 | 1.9984e-15 | 2.2204e-16 | 3.3307e-16 | 0.007462 | 0.012803 | 208.6 | 256 |
| [q4-d8-noncommuting-29-selective_p2](q4-d8-noncommuting-29-selective_p2.json) | selective_p2 | completed | 1.4126e-07 | 1.2325e-07 | 2.1726e-07 | 2.8901e-07 | 0.021863 | 0.030870 | 209.0 | 128 |
| [q4-d8-noncommuting-29-double_single_p3](q4-d8-noncommuting-29-double_single_p3.json) | double_single_p3 | completed | 8.0232e-15 | 2.6645e-15 | 1.0256e-14 | 1.2212e-15 | 0.362824 | 0.371889 | 209.7 | 256 |
| [q4-d8-cancellation_echo-11-fp32](q4-d8-cancellation_echo-11-fp32.json) | fp32 | completed | 7.0643e-08 | 1.5579e-07 | 0.44354 | 0.89626 | 0.013945 | 0.030375 | 212.5 | 128 |
| [q4-d8-cancellation_echo-11-double_single_p4](q4-d8-cancellation_echo-11-double_single_p4.json) | double_single_p4 | completed | 1.7795e-15 | 7.7716e-15 | 9.7467e-08 | 9.6087e-08 | 0.351793 | 0.390760 | 209.4 | 256 |
| [q4-d8-cancellation_echo-11-fp64](q4-d8-cancellation_echo-11-fp64.json) | fp64 | completed | 2.8406e-16 | 3.7748e-15 | 1.5543e-15 | 6.6804e-09 | 0.005480 | 0.008053 | 208.5 | 256 |
| [q4-d8-cancellation_echo-11-selective_p2](q4-d8-cancellation_echo-11-selective_p2.json) | selective_p2 | completed | 7.0643e-08 | 1.5579e-07 | 3.3806e-07 | 2.743e-07 | 0.019742 | 0.029616 | 209.0 | 128 |
| [q4-d8-cancellation_echo-11-double_single_p3](q4-d8-cancellation_echo-11-double_single_p3.json) | double_single_p3 | completed | 1.7795e-15 | 7.7716e-15 | 9.7467e-08 | 9.6087e-08 | 0.250773 | 0.243769 | 209.2 | 256 |
| [q4-d8-cancellation_echo-29-fp32](q4-d8-cancellation_echo-29-fp32.json) | fp32 | completed | 3.303e-08 | 1.1511e-07 | 0.45922 | 0.88832 | 0.012889 | 0.032612 | 212.3 | 128 |
| [q4-d8-cancellation_echo-29-double_single_p4](q4-d8-cancellation_echo-29-double_single_p4.json) | double_single_p4 | completed | 1.7492e-15 | 1.4877e-14 | 1.0869e-08 | 1.5858e-08 | 0.341190 | 0.378629 | 209.7 | 256 |
| [q4-d8-cancellation_echo-29-fp64](q4-d8-cancellation_echo-29-fp64.json) | fp64 | completed | 2.2242e-16 | 3.4417e-15 | 7.7716e-16 | 9.5724e-10 | 0.005523 | 0.008285 | 208.5 | 256 |
| [q4-d8-cancellation_echo-29-selective_p2](q4-d8-cancellation_echo-29-selective_p2.json) | selective_p2 | completed | 3.303e-08 | 1.1511e-07 | 1.1511e-07 | 5.1056e-07 | 0.019728 | 0.030374 | 208.9 | 128 |
| [q4-d8-cancellation_echo-29-double_single_p3](q4-d8-cancellation_echo-29-double_single_p3.json) | double_single_p3 | completed | 1.6829e-15 | 1.4322e-14 | 1.0869e-08 | 1.5858e-08 | 0.238298 | 0.244493 | 209.4 | 256 |

## Accuracy/time/peak-memory frontier

The frontier applies to the frozen matched FP32-representable inputs, not the counterfactual intended FP64 angle. Error against that intended angle is separately retained in `quality_vs_intended_inputs`; it can exceed a chosen tolerance even when the matched-input frontier is feasible. A route is feasible only when amplitude error, norm drift, every observable error and gradient error are all within the declared dimensionless tolerance. Among feasible routes, Pareto dominance uses median complete-circuit gradient time and observed peak memory (CUDA allocator peak or CPU worker RSS). Unobservable accelerator peaks are excluded from Pareto scoring. State storage is reported separately and never substituted for a peak. These are not exclusive-system-memory claims.

| Workload | Tolerance | Feasible routes | Pareto routes |
| --- | ---: | --- | --- |
| q2-d2-cancellation_echo-11 | 0.001 | double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q2-d2-cancellation_echo-11 | 1e-06 | double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q2-d2-cancellation_echo-11 | 1e-09 | none | none |
| q2-d2-cancellation_echo-29 | 0.001 | double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q2-d2-cancellation_echo-29 | 1e-06 | double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q2-d2-cancellation_echo-29 | 1e-09 | fp64 | fp64 |
| q2-d2-noncommuting-11 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2 |
| q2-d2-noncommuting-11 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64, selective_p2 |
| q2-d2-noncommuting-11 | 1e-09 | double_single_p4, fp64, double_single_p3 | fp64 |
| q2-d2-noncommuting-29 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q2-d2-noncommuting-29 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q2-d2-noncommuting-29 | 1e-09 | double_single_p4, fp64, double_single_p3 | fp64 |
| q2-d8-cancellation_echo-11 | 0.001 | double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q2-d8-cancellation_echo-11 | 1e-06 | double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q2-d8-cancellation_echo-11 | 1e-09 | none | none |
| q2-d8-cancellation_echo-29 | 0.001 | double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q2-d8-cancellation_echo-29 | 1e-06 | double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q2-d8-cancellation_echo-29 | 1e-09 | fp64 | fp64 |
| q4-d8-cancellation_echo-11 | 0.001 | double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q4-d8-cancellation_echo-11 | 1e-06 | double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q4-d8-cancellation_echo-11 | 1e-09 | none | none |
| q4-d8-cancellation_echo-29 | 0.001 | double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q4-d8-cancellation_echo-29 | 1e-06 | double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q4-d8-cancellation_echo-29 | 1e-09 | fp64 | fp64 |
| q2-d8-noncommuting-11 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q2-d8-noncommuting-11 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q2-d8-noncommuting-11 | 1e-09 | double_single_p4, fp64, double_single_p3 | fp64 |
| q2-d8-noncommuting-29 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q2-d8-noncommuting-29 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q2-d8-noncommuting-29 | 1e-09 | double_single_p4, fp64, double_single_p3 | fp64 |
| q4-d8-noncommuting-11 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q4-d8-noncommuting-11 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q4-d8-noncommuting-11 | 1e-09 | double_single_p4, fp64, double_single_p3 | fp64 |
| q4-d8-noncommuting-29 | 0.001 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q4-d8-noncommuting-29 | 1e-06 | fp32, double_single_p4, fp64, selective_p2, double_single_p3 | fp64 |
| q4-d8-noncommuting-29 | 1e-09 | double_single_p4, fp64, double_single_p3 | fp64 |

Slower and inaccurate completed cases remain in the table. Unsupported and failed routes receive no speedup score. Repeated bit identity is based on actual output dtype/shape/byte hashes; equal scalar errors alone never imply identical outputs.

Double-Single retains binary32 exponent range, is not IEEE binary64, and Dekker splitting requires finite operands small enough that multiplication by 4097 does not overflow. P4 direct angles are bounded by |theta| <= 1024; only the existing fixed and RX/RY/RZ/RXX/RYY/RZZ gates are supported. P3 uses explicit CPU complex128 gate encoding. P5 autograd/optimizer boundaries are separately probed and are not promoted by this parameter-shift benchmark.
