# Memory-forced high-width TN scaling

## Workload and backend boundary

The frozen workload is a 36-qubit 4x9 grid circuit with four full horizontal
and vertical entangling cycles, requesting one amplitude in complex64.

- Statevector requires 512 GiB and cannot fit one A800.
- The selector estimates an MPS bond of 262,144 and about 36 TiB of MPS storage.
- The unsliced native TN path has a planned 64 GiB largest intermediate, which
  leaves insufficient allocator and workspace headroom on an 80 GB device.
- A 16 GiB intermediate limit produces eight slices with recomputation factor
  2.25.  The same eight slices are used for every world size.

This is therefore a genuine TN workload rather than a 30-qubit problem that SV
or MPS should solve.

## Measured A800 strong scaling

| GPUs | Tasks per rank | Execution (s) | Speedup | Efficiency | Peak/rank |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 12.018 | 1.00x | 100% | 64.0 GiB |
| 2 | 4 | 8.113 | 1.48x | 74% | 64.0 GiB |
| 4 | 2 | 5.946 | 2.02x | 51% | 64.0 GiB |
| 8 | 1 | 4.767 | 2.52x | 32% | 64.0 GiB |

All ranks return the same complex64 amplitude (approximately 0.9848883) and
the results agree across world sizes.

The executor now shows positive scaling on a capacity-driven workload, but the
8-GPU production target of 5x is not met.  The result is evidence of useful
distributed TN capacity, not yet evidence of production-grade efficiency.

## Memory finding

The planned largest intermediate is 16 GiB, while measured allocator peak is
64 GiB.  PyTorch also reports a failed 16 GiB temporary allocation before the
contraction succeeds via a fallback route.  Consequently, intermediate-size
planning alone is not a sufficient memory contract.  Production planning must
model the pair of operands, output, einsum workspace, allocator fragmentation,
and retained tensors.

## Cotengra comparison

The exact projected network is frozen at
`benchmarks/workloads/tn_cotengra_gap/flagquantum_36q_4x9_c4_amplitude.json`.
Under the same nominal 16 GiB largest-intermediate target and an approximately
10-second search:

- FlagQuantum selects eight slices, total estimated cost 1.276e12.
- cotengra 0.8.2 selects two slices, estimated cost 1.068e14.

For this workload and budget, the native path has substantially lower estimated
work, while cotengra uses fewer slices.  This is planning evidence only; the
paths use different executors and cotengra exceeded the nominal search budget.
It does not justify a universal path-quality claim.

## Next production work

1. Replace the largest-intermediate limit with an end-to-end working-set model.
2. Avoid the known failed-allocation/fallback path by selecting a workspace-safe
   contraction kernel before execution.
3. Plan once on rank zero and broadcast the frozen sliced DAG.
4. Batch or pipeline multiple slices per rank to overlap host launch work.
5. Re-run this frozen workload; require at least 5x speedup on eight A800s
   without allocator OOM warnings before marking distributed TN production-ready.
