# Rank-Owned MPS Forward

`flagquantum.runtime.executors.mps.execute_torch_distributed_mps_forward` is the Phase-3 PyTorch-native MPS
forward executor. Every rank allocates only its contiguous owned sites; no full
`MPSState` is constructed before or during production execution. The older
`run_distributed_mps` remains a development path and keeps
its existing hybrid/replicated blockers.

One-site and adjacent same-shard two-site gates execute without communication.
An adjacent cross-shard gate sends only the right boundary tensor to the left
owner, applies the two-site update/SVD there, and returns only the updated right
tensor. Non-adjacent gates fail with a routing diagnostic before execution.

Partitions support 2 through N ranks and variable bond dimensions. After
two-site updates, ranks exchange tensor-byte metadata; materially skewed
partitions are recomputed as contiguous weighted ranges and only sites whose
owner changes migrate through point-to-point tensor transfer.

Production results expose rank placement, per-rank tensor bytes, bond
dimensions, boundary messages/bytes, partition history, and rebalance count.
Calling `full_state()` fails. `gather_mps_for_validation` is an explicitly named
test-only helper used for small dense/local-MPS differentials and is never
invoked by the executor.

This issue completes forward ownership only. Sharded backward, optimizer
ownership, accelerator capacity, canonicalization, and truncation policy remain
separate Phase-3 gates, so the result cannot yet make a scalability claim.
