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
