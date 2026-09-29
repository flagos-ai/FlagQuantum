# Native CPU forward CX gather comparison

This report is generated from [`native_cpu_forward_cx_gather_cpu_arm64_20260929.json`](native_cpu_forward_cx_gather_cpu_arm64_20260929.json). It compares
FlagQuantum's reusable-output native CPU CX gather with the same adjoint path
using allocating PyTorch `index_select`. The value and full parameter gradient
must agree exactly; ratios above 1 mean the native path is faster.

| Workload | Qubits | CX gates | Native forward (ms) | Rollback forward (ms) | Forward speedup | Native total (ms) | Rollback total (ms) | Total speedup | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 22 | 21 | 33.317 | 42.157 | 1.27x | 183.165 | 196.192 | 1.07x | 0.000e+00 |
| QAOA path MaxCut | 22 | 0 | 62.479 | 61.886 | 0.99x | 188.034 | 185.451 | 0.99x | 0.000e+00 |

## Interpretation

The hardware-efficient VQE case contains one 21-gate CX chain, so it
exercises the optimized gather once per forward sweep. QAOA contains no
CX gates and is a negative control: its small timing difference is ordinary
run-to-run noise, not an optimization claim. This is non-release,
single-device comparison evidence; it does not establish scaling claims.

## Reproduce

```bash
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 22 \
  --engines flagquantum_adjoint flagquantum_adjoint_forward_cx_rollback \
  --layers 1 --threads 8 \
  --warmup 3 --iterations 11 \
  --calls-per-sample 1 --skip-memory-probe \
  --json-output native_cpu_forward_cx_gather_cpu_arm64_20260929.json --markdown-output REPORT.md
```
