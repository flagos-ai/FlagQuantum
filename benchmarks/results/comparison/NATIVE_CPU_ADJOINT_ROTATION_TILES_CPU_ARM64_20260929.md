# Native CPU adjoint rotation-tile comparison

This report is generated from [`native_cpu_adjoint_rotation_tiles_cpu_arm64_20260929.json`](native_cpu_adjoint_rotation_tiles_cpu_arm64_20260929.json). It compares
FlagQuantum's full-layer, structure-specialized native adjoint kernels with the
legacy 48-gate, two-wire rotation path. The optimized path also accumulates one
gradient subtotal per tile and applies adjacent fixed Hadamards to ket and
adjoint together in 11-wire tiles. Both paths execute the same exact
statevector-adjoint method and must return matching values and gradients.

| Workload | Qubits | Rotations | Native backward (ms) | Rollback backward (ms) | Backward speedup | Native total (ms) | Rollback total (ms) | Total speedup | Max gradient error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient VQE | 22 | 66 | 73.798 | 114.272 | 1.55x | 108.920 | 149.617 | 1.37x | 1.311e-13 |
| QAOA path MaxCut | 22 | 22 | 89.157 | 117.909 | 1.32x | 151.929 | 182.442 | 1.20x | 5.507e-14 |

## Interpretation

Hardware-efficient VQE has 66 adjacent RX/RY/RZ gates. The optimized
kernel processes the complete layer in bounded statevector tiles and uses
the known sparse rotation structure instead of general complex matrix
multiplication. Its tile-local gradient subtotal avoids a hot memory update for
every amplitude pair. QAOA has only 22 RX gates, but its 22 fixed Hadamards now
use a paired ket/adjoint real-arithmetic kernel, reducing eight full-state
kernel calls to two bounded tile calls. This is non-release single-device
evidence and is not a scaling claim.

## Reproduce

```bash
flagquantum-benchmark run differentiable_simulator_corpus \
  --workloads hardware_efficient_vqe qaoa_path_maxcut \
  --n-wires 22 \
  --engines flagquantum_adjoint flagquantum_adjoint_rotation_tile_rollback \
  --layers 1 --threads 8 \
  --warmup 3 --iterations 11 \
  --calls-per-sample 1 --skip-memory-probe \
  --json-output native_cpu_adjoint_rotation_tiles_cpu_arm64_20260929.json --markdown-output REPORT.md
```

Set `FQ_NATIVE_CPU_ADJOINT_WIDE_TILES=0` for direct rollback.
