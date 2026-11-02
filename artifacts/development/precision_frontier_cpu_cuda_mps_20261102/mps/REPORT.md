# Complete-circuit precision frontier

Source: `3ba42c605f36d86ce52c499dc216fab72f4ddcac`. Device: `mps`. Smoke: `False`.

Development comparison only. No precision provider, capability maturity, vendor certification, simulation/GEMM speedup or general FP64 equivalence is established.

All routes share the same initialized FP32-representable angles and exactly represented observable coefficients. Reference: independent CPU NumPy complex128 dense matrices and analytic initial-RY tangent. Repeated Z coefficients are coalesced before reference evaluation. CPU reference and diagnostic conversion are outside candidate timing, not a fallback execution path.

Timing includes circuit construction, gate generation, state evolution, expectation and supported parameter-shift gradient. Samples are synchronized absolute seconds; the statistic is the median. First call includes preflight and is separate. Renormalization is disabled for every route. Payload memory is exact storage, not peak memory; worker RSS and CUDA allocator peak have separately stated scopes.

| Case | Mode | Status | State max error | Norm drift | Observable max error | Gradient error | Median s | First s | RSS MiB | State bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q2-d2-noncommuting-11-fp32 | fp32 | unsupported: P1 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d2-noncommuting-11-double_single_p4 | double_single_p4 | unsupported: P4 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d2-noncommuting-11-fp64 | fp64 | unsupported: native_fp64 preflight: Apple MPS has no FP64/complex128 kernels; CPU substitution is prohibited | — | — | — | — | — | — | — | — |
| q2-d2-noncommuting-29-fp32 | fp32 | unsupported: P1 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d2-noncommuting-29-double_single_p4 | double_single_p4 | unsupported: P4 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d2-noncommuting-29-fp64 | fp64 | unsupported: native_fp64 preflight: Apple MPS has no FP64/complex128 kernels; CPU substitution is prohibited | — | — | — | — | — | — | — | — |
| q2-d2-cancellation_echo-11-fp32 | fp32 | unsupported: P1 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d2-cancellation_echo-11-double_single_p4 | double_single_p4 | unsupported: P4 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d2-cancellation_echo-11-fp64 | fp64 | unsupported: native_fp64 preflight: Apple MPS has no FP64/complex128 kernels; CPU substitution is prohibited | — | — | — | — | — | — | — | — |
| q2-d2-cancellation_echo-29-fp32 | fp32 | unsupported: P1 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d2-cancellation_echo-29-double_single_p4 | double_single_p4 | unsupported: P4 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d2-cancellation_echo-29-fp64 | fp64 | unsupported: native_fp64 preflight: Apple MPS has no FP64/complex128 kernels; CPU substitution is prohibited | — | — | — | — | — | — | — | — |
| q2-d8-noncommuting-11-fp32 | fp32 | unsupported: P1 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d8-noncommuting-11-double_single_p4 | double_single_p4 | unsupported: P4 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d8-noncommuting-11-fp64 | fp64 | unsupported: native_fp64 preflight: Apple MPS has no FP64/complex128 kernels; CPU substitution is prohibited | — | — | — | — | — | — | — | — |
| q2-d8-noncommuting-29-fp32 | fp32 | unsupported: P1 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d8-noncommuting-29-double_single_p4 | double_single_p4 | unsupported: P4 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d8-noncommuting-29-fp64 | fp64 | unsupported: native_fp64 preflight: Apple MPS has no FP64/complex128 kernels; CPU substitution is prohibited | — | — | — | — | — | — | — | — |
| q2-d8-cancellation_echo-11-fp32 | fp32 | unsupported: P1 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d8-cancellation_echo-11-double_single_p4 | double_single_p4 | unsupported: P4 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d8-cancellation_echo-11-fp64 | fp64 | unsupported: native_fp64 preflight: Apple MPS has no FP64/complex128 kernels; CPU substitution is prohibited | — | — | — | — | — | — | — | — |
| q2-d8-cancellation_echo-29-fp32 | fp32 | unsupported: P1 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d8-cancellation_echo-29-double_single_p4 | double_single_p4 | unsupported: P4 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q2-d8-cancellation_echo-29-fp64 | fp64 | unsupported: native_fp64 preflight: Apple MPS has no FP64/complex128 kernels; CPU substitution is prohibited | — | — | — | — | — | — | — | — |
| q4-d8-noncommuting-11-fp32 | fp32 | unsupported: P1 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q4-d8-noncommuting-11-double_single_p4 | double_single_p4 | unsupported: P4 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q4-d8-noncommuting-11-fp64 | fp64 | unsupported: native_fp64 preflight: Apple MPS has no FP64/complex128 kernels; CPU substitution is prohibited | — | — | — | — | — | — | — | — |
| q4-d8-noncommuting-29-fp32 | fp32 | unsupported: P1 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q4-d8-noncommuting-29-double_single_p4 | double_single_p4 | unsupported: P4 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q4-d8-noncommuting-29-fp64 | fp64 | unsupported: native_fp64 preflight: Apple MPS has no FP64/complex128 kernels; CPU substitution is prohibited | — | — | — | — | — | — | — | — |
| q4-d8-cancellation_echo-11-fp32 | fp32 | unsupported: P1 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q4-d8-cancellation_echo-11-double_single_p4 | double_single_p4 | unsupported: P4 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q4-d8-cancellation_echo-11-fp64 | fp64 | unsupported: native_fp64 preflight: Apple MPS has no FP64/complex128 kernels; CPU substitution is prohibited | — | — | — | — | — | — | — | — |
| q4-d8-cancellation_echo-29-fp32 | fp32 | unsupported: P1 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q4-d8-cancellation_echo-29-double_single_p4 | double_single_p4 | unsupported: P4 preflight: KeyError: "Unknown FlagQuantum platform 'mps'." | — | — | — | — | — | — | — | — |
| q4-d8-cancellation_echo-29-fp64 | fp64 | unsupported: native_fp64 preflight: Apple MPS has no FP64/complex128 kernels; CPU substitution is prohibited | — | — | — | — | — | — | — | — |

## Accuracy/time/peak-memory frontier

The frontier applies to the frozen matched FP32-representable inputs, not the counterfactual intended FP64 angle. Error against that intended angle is separately retained in `quality_vs_intended_inputs`; it can exceed a chosen tolerance even when the matched-input frontier is feasible. A route is feasible only when amplitude error, norm drift, every observable error and gradient error are all within the declared dimensionless tolerance. Among feasible routes, Pareto dominance uses median complete-circuit gradient time and observed peak memory (CUDA allocator peak or CPU worker RSS). Unobservable accelerator peaks are excluded from Pareto scoring. State storage is reported separately and never substituted for a peak. These are not exclusive-system-memory claims.

| Workload | Tolerance | Feasible routes | Pareto routes |
| --- | ---: | --- | --- |

Slower and inaccurate completed cases remain in the table. Unsupported and failed routes receive no speedup score. Repeated bit identity is based on actual output dtype/shape/byte hashes; equal scalar errors alone never imply identical outputs.

Double-Single retains binary32 exponent range, is not IEEE binary64, and Dekker splitting requires finite operands small enough that multiplication by 4097 does not overflow. P4 direct angles are bounded by |theta| <= 1024; only the existing fixed and RX/RY/RZ/RXX/RYY/RZZ gates are supported. P3 uses explicit CPU complex128 gate encoding. P5 autograd/optimizer boundaries are separately probed and are not promoted by this parameter-shift benchmark.
