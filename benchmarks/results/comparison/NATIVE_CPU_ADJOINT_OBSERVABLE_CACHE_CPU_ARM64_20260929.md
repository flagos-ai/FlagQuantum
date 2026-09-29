# Native CPU adjoint observable-weight cache comparison

This report is generated from [`native_cpu_adjoint_observable_cache_cpu_arm64_20260929.json`](native_cpu_adjoint_observable_cache_cpu_arm64_20260929.json). It measures
steady-state repeated optimization steps with FlagQuantum's bounded observable
diagonal cache enabled and disabled. Both paths use the same native adjoint
implementation and must produce identical values and full parameter gradients.

| Workload | Qubits | Observable terms | Cached forward (ms) | Rollback forward (ms) | Forward speedup | Cached total (ms) | Rollback total (ms) | Total speedup | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 22 | 43 | 35.555 | 290.316 | 8.17x | 188.426 | 448.590 | 2.38x | 0.000e+00 |
| QAOA path MaxCut | 22 | 22 | 59.131 | 193.335 | 3.27x | 177.901 | 312.320 | 1.76x | 0.000e+00 |

## Interpretation

The first execution builds the real Z/ZZ observable diagonal. Later optimizer
steps reuse it because the Hamiltonian structure is independent of trainable
gate parameters. The cache is CPU-only, keyed by terms, width, dtype, device,
and local amplitude count, and bounded to eight entries and 256 MiB. The
warmups populate it before retained measurements. This is non-release local
evidence and does not establish distributed or accelerator scaling claims.

## Reproduce

```bash
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 22 \
  --engines flagquantum_adjoint flagquantum_adjoint_observable_cache_rollback \
  --layers 1 --threads 8 \
  --warmup 3 --iterations 11 \
  --calls-per-sample 1 --skip-memory-probe \
  --json-output native_cpu_adjoint_observable_cache_cpu_arm64_20260929.json --markdown-output REPORT.md
```

Set `FQ_STATEVECTOR_ADJOINT_OBSERVABLE_CACHE=0` for direct rollback.
