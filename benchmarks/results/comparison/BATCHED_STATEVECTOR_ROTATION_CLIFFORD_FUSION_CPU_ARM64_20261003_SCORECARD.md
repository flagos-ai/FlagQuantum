# Batched rotation/Clifford fusion CPU scorecard

## Conclusion

Pass for the measured 18-qubit, batch-32, complex128 CPU inference scope.
FlagQuantum now executes each eligible disjoint `RY`/`RZ` tile and its following
disjoint CX matching inside one native CPU traversal. The exact rollback takes
**1700.698 ms** and the selected path takes **1673.173 ms**, a conservative
**1.016x speedup**. More importantly, median execution RSS growth falls from
**399.1 MiB to 308.7 MiB**, a **22.7% reduction**, because the CX matching no
longer allocates a full batched-state output.

In the same run, PennyLane Lightning native batch takes **2093.909 ms**, so
FlagQuantum is **1.251x faster** for this complete task. Lightning uses less
execution RSS growth (275.8 MiB), so the memory comparison is not presented as
a universal win.

## What is measured and why it matters

The local-brickwork workload contains four alternating layers. Every layer has
`RY` and `RZ` on all 18 qubits followed by a nearest-neighbour CX matching; a
final `RY` layer carries 576 independent parameter values across 32 circuit
items. Each item has 196 gates and returns an exact complex128 statevector.
The logical output alone is 128 MiB.

This pattern represents local variational and tensor-network-inspired ansatzes.
Before this change the rotations and CX matching traversed the full state in
separate kernels, while the matching also materialized a second full batched
state. Fusion makes the permutation part of the rotation tile's final write.

Warm time is the median of 11 complete calls after three warmups. Peak RSS is
the median of three fresh-process cold probes. Circuit construction is outside
warm timing and inside RSS measurement. Each engine uses one CPU thread.

## Formal results

| Engine | Median task time | Relative result | Peak RSS | Execution RSS growth | Maximum state error |
| --- | ---: | ---: | ---: | ---: | ---: |
| **FlagQuantum fused rotation + CX matching** | **1673.173 ms** | **1.016x faster than rollback** | **483.8 MiB** | **308.7 MiB** | 0 |
| FlagQuantum separate rotation/Clifford rollback | 1700.698 ms | baseline | 568.7 MiB | 399.1 MiB | 5.68e-17 |
| **PennyLane Lightning native batch** | **2093.909 ms** | **FlagQuantum is 1.251x faster** | 530.8 MiB | **275.8 MiB** | 5.55e-17 |

The selected, rollback, and Lightning relative median absolute deviations are
3.42%, 2.14%, and 3.13%; all pass the 20% stability rule. Raw samples, versions,
IR hash, correctness values, and every memory probe are retained in the
[JSON artifact](batched_statevector_rotation_clifford_fusion_cpu_arm64_20261003.json).

## User code

No new public API is required. Eligible inference circuits select the native
path automatically:

```python
import torch
import flagquantum as fq

n_wires = 18
batch_size = 32
circuit = fq.Circuit(n_wires, bsz=batch_size, dtype=torch.complex128)

for layer in range(4):
    for wire in range(n_wires):
        angle = 0.07 * (layer + 1) * (wire + 1)
        circuit.ry(wire, angle)
        circuit.rz(wire, -0.6 * angle)
    for left in range(layer % 2, n_wires - 1, 2):
        circuit.cx(left, left + 1)

angles = torch.linspace(-0.4, 0.4, batch_size, dtype=torch.float64)
for wire in range(n_wires):
    circuit.ry(wire, angles + 0.01 * wire)

states = circuit.state()  # shape: (32, 2**18)
```

## Implementation and safety

The compiler only fuses consecutive native one-qubit rotation regions followed
by a pure, wire-disjoint CX matching. It packs complete CX pairs into tiles of
at most six wires, so no edge crosses a tile. The native kernel precomputes the
tile-local CX source map once and uses it during the final write; it does not
allocate a permutation output.

The path is CPU-inference-only. Autograd inputs keep the existing differentiable
program, non-CPU execution is unchanged, caller-owned input tensors are not
mutated, and complex64/complex128 both have focused numerical coverage. Set
`FQ_CPU_NATIVE_ROTATION_CLIFFORD_FUSION=0` for the exact rollback.

## Reproduce

```bash
pip install -e '.[pennylane]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m flagquantum.benchmarking.batched_statevector_memory \
  --workloads local_brickwork_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch \
            flagquantum_native_rotation_clifford_rollback \
            pennylane_lightning_native_batch \
  --threads 1 --warmup 3 --iterations 11 --memory-probes 3 \
  --json-output rotation-clifford-fusion.json \
  --markdown-output ROTATION_CLIFFORD_FUSION.md
```

## Boundaries and next target

- The 1.016x rollback speedup is small; the accepted gain is primarily the
  22.7% execution-RSS reduction without a timing regression.
- The result covers pure disjoint CX matchings after disjoint named rotations,
  not CZ edges, arbitrary two-qubit gates, gradients, or GPU execution.
- Peak RSS includes interpreter, framework, circuit construction, and allocator
  behavior. This is Apple-arm64 local comparison evidence, not a release-grade
  scalability result or universal framework ranking.
- Lightning is slower here but retains a 10.6% execution-RSS-growth advantage.
  Future work should reduce batch assembly lifetime rather than widen this
  fusion to unsupported circuit shapes.
