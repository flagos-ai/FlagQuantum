# Hybrid Parallel Semantics

FlagQuantum reports independent input-batch, parameter-batch,
observable-batch, data-parallel, model-parallel, and quantum-state-parallel
dimensions. Only quantum-state sharding has `sharded_across_ranks` capacity
semantics. DDP remains `data_parallel_replicated` and cannot by itself support a
single-problem capacity claim.

For composed execution, process groups are orthogonal. With two data replicas
and two state shards, ranks `(0, 1)` and `(2, 3)` are state groups, while
`(0, 2)` and `(1, 3)` are DDP groups. Statevector forward/backward reductions
remain inside each state group; DDP averages the completed parameter gradients
between replicas exactly once.

`flagquantum.runtime.parallel.plan_hybrid_parallel(...)` validates the factorization and reports group
membership, ownership, per-rank state/input memory, parameter and observable
memory, state-exchange bytes, DDP gradient communication, and observable
reductions. Model-parallel execution currently fails closed. FSDP and DTensor
may own or shard model parameters, but they must not claim ownership of the
runtime-managed quantum state or reuse its process group implicitly.

`fq.Module` supports vectorized input and parameter batches. Hamiltonian terms
are greedily grouped when they are qubit-wise commuting; result metrics expose
both term count and group count. Optional JAX kernels may execute rank-local
math, but PyTorch process groups and DDP retain orchestration ownership.
