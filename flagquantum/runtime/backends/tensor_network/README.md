# Distributed Tensor-Network Runtime

This directory owns contraction DAGs, slice and shard ownership, checkpoint and
rematerialization plans, distributed communication, optimizer lifecycle, and
execution evidence for tensor-network backends.

It does not own pair-contraction, pair-pullback, high-rank fallback, or
compensated-accumulation mathematics. Those numerical primitives live in
`flagquantum/simulation/tensor_network/stages.py`; complex real/imaginary kernels live
in `flagquantum/simulation/real_imag_kernels.py`.

For a reverse-mode change, start with `reverse_dag.py` for DAG/tape execution or
`sliced_reverse.py` for slice orchestration. `distributed_sliced_reverse.py`
adds process-group reduction and measured communication. Keep task ownership,
checkpoint selection, cotangent lifecycle, batching schedules, collectives,
and evidence here. Do not move them into Simulation merely because they
manipulate tensors.

For end-to-end distributed tensor-network execution, start with `execution.py`.
It owns slice planning, rank-local execution, and reduction; `state.py` owns the
result objects and their evidence summaries. The numerical contractions invoked
by execution remain in Simulation. Persistent plan serialization and locking
are isolated in `plan_cache.py`.

There are two intentionally separate distributed execution paths:

- `execution.py` assigns complete, independent slice contractions to ranks and
  sums their outputs. It supports the local development mirror and a
  differentiable all-reduce.
- `distributed_execution.py` executes a planned contraction DAG whose
  intermediate tensors have explicit owners or shards. It performs point-to-
  point transfers, redistribution, and small-result replication.

They share numerical contraction primitives, but not scheduling or reduction
semantics. Do not route one through the other merely because both use
`torch.distributed`.

This boundary has reached its current stopping point. Forward contractions and
reverse pullbacks already call Simulation-owned primitives. Tensor stacking
for schedule batches, shard slicing/combining, cotangent-map accumulation, and
non-finite/evidence summaries express Runtime plans or result lifecycle; they
are not independent numerical algorithms. Extract more code only when the
operation can be reused without Runtime DAG, task, checkpoint, ownership,
process-group, or evidence types.

Run the focused checks with:

```bash
pytest tests/unit/test_tensor_stages.py tests/integration/test_cpu_vertical_slice.py
```
