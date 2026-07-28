# FlagQuantum Distributed Quantum AI Principles

FlagQuantum's distributed quantum AI goal is not to launch many identical
quantum jobs. The goal is to make one logical quantum AI workload larger,
trainable, measurable, and deployable when one device is not enough.

## Core Definition

A distributed quantum AI feature is only a scalability feature when it partitions
one logical workload across ranks:

- statevector amplitudes are sharded across ranks
- MPS tensors, sites, or bonds are sharded across ranks
- tensor-network contraction graphs, slices, or intermediates are partitioned
  across ranks
- observables and Hamiltonian terms are explicitly scheduled as task-parallel
  work, not confused with state-capacity scaling
- gradients and optimizer updates preserve the same distribution semantics

Replicating the same complete circuit on every GPU is useful for data parallel
training or smoke tests, but it is not capacity expansion.

## Single-Device Performance Rule

Distributed design must not degrade top-tier single-GPU or single-CPU
performance. Single-device execution is a first-class fast path, not a degraded
distributed configuration with `world_size=1`.

Single-device statevector, MPS, TN, and JAX-kernel paths should:

- avoid initializing `torch.distributed`
- avoid unnecessary rank-placement, communication, or collective setup
- avoid sharded planner overhead unless explicitly requested
- keep optimized local kernels, JIT caches, tensor layouts, and autograd paths
  independent from distributed orchestration
- report `single_device_fast_path` or another non-scalability semantic when a
  benchmark needs to distinguish local peak performance from distributed
  capacity scaling

Distributed scalability metadata is required for distributed claims, but it
must not become mandatory overhead for the local peak-performance path.

For local development and regression checks, run:

```python
import flagquantum as fq

report = fq.local_fast_path_preflight()
assert report.passed
```

This verifies that `statevector`, `mps`, and `tensor_network` remain
single-device fast paths and do not depend on distributed backend policy,
torchrun, or process-group setup.

## User Experience Tiers

FlagQuantum must serve two product tiers without splitting the user-facing API:

- `single_device`: non-production, education, research, laptop, CPU-only, or
  single-GPU users. These users should be able to write `fq.Circuit`, call
  `run`, `state`, `expectation`, or `plan`, and get a fast useful result without
  learning torchrun, rank placement, NCCL, or cluster topology.
- `production_distributed`: enterprise, industrial, and large-scale users. They
  need real multi-GPU/multi-node sharding, capacity expansion, diagnostics,
  checkpointing, and gradient scalability, but should still enter through the
  same FlagQuantum IR and high-level APIs.

The API contract is one product, two execution tiers:

- local tier: `single_api_fast_path`
- production tier: `single_api_distributed_scale_out`

Complex distributed controls may exist, but the planner should make reasonable
defaults, explain decisions, and expose diagnostics without forcing every user
to become a distributed-systems expert.

## Development And Production Backend Switching

FlagQuantum must support the same user code in local development and production
deployment. Backend switching is controlled by environment variables, not by
rewriting circuits or training loops.

Primary switch:

```bash
FQ_DISTRIBUTED_PROFILE=development  # single CPU process, simulated ranks
FQ_DISTRIBUTED_PROFILE=production   # real torchrun/multi-GPU or multi-node
FQ_DISTRIBUTED_PROFILE=auto         # development unless WORLD_SIZE > 1
```

Backend defaults:

```bash
# development
FQ_JAX_DISTRIBUTED_BACKEND=pmap_local_cpu
FQ_TORCH_DISTRIBUTED_BACKEND=local_tensor
FQ_LOCAL_WORLD_SIZE=2

# production
FQ_JAX_DISTRIBUTED_BACKEND=pmap
FQ_TORCH_DISTRIBUTED_BACKEND=torch_distributed
```

Development mode must simulate rank ownership, sharding, and communication
semantics on CPU. Production mode must use real distributed backends. The
business code should remain:

```python
import flagquantum as fq
circuit = fq.Circuit(...)
```

and only the environment changes.

Development mode is meaningful only when it is a faithful mirror of production
distributed intent. The same Circuit/IR must produce the same shard ownership,
rank/task layout, and communication signature in development and production
profiles. Use `fq.validate_development_production_parity(...)` in tests and
preflight checks when changing distributed statevector, MPS, or tensor-network
runtime code.

For everyday local development, run:

```python
import flagquantum as fq

report = fq.local_distributed_development_preflight(world_size=2)
assert report.passed
```

This check needs no GPU and no torchrun. It verifies that local development
execution and production distributed intent agree for statevector, MPS, and
tensor-network modes before code is moved to a cluster.

## Execution Semantics

Every distributed runtime and benchmark must declare one of these semantics:

- `single_device_fast_path`: one CPU/GPU executes the workload using the fastest
  local path; this is peak local performance, not distributed scalability.
- `sharded_across_ranks`: one workload is partitioned across ranks; this can
  support capacity-scaling claims.
- `data_parallel_replicated`: each rank owns different data/batch samples but a
  full quantum model replica.
- `observable_term_parallel`: ranks split Hamiltonian/measurement terms while
  the circuit/state is replicated.
- `manual_sliced_tensor_contraction`: ranks/devices split contraction slices for
  a specific tensor-network contraction.
- `rank_local_replicated_kernel`: each rank runs a complete independent quantum
  kernel.
- `replicated_per_rank`: each rank runs the same complete workload for smoke or
  health checks.

Only `sharded_across_ranks` can be considered for single-problem capacity
expansion. Plan and preflight summaries should report
`sharding_plan_available: true` rather than acting as release evidence.
Release-grade claims must additionally report an approved `claim_evidence_type`,
currently `production_training_benchmark` or an explicit `release_payload`,
single-device capacity failure evidence, and optimizer updates that preserve the
same sharded semantics as forward and backward execution.

## Sharded-First Runtime

Large-scale modes must be designed sharded-first:

- distributed statevector: amplitude or qubit-address sharding with gate-aware
  communication
- distributed MPS: site/bond sharding without a required full local MPS copy
- distributed TN: graph partitioning, slicing, and intermediate tensor sharding
- noisy simulation: density-matrix or trajectory-level sharding with explicit
  memory accounting
- cloud deployment: trained parameters remain attached to the same circuit IR

The runtime should fail closed when a path silently falls back to replication.

## Unified Planning

The FlagQuantum IR should feed a planner that produces:

- execution plan
- memory plan
- communication plan
- observable plan
- gradient plan
- compiler/deployment plan

The planner must decide whether a workload should use statevector, MPS, TN,
observable parallelism, data parallelism, or hybrid strategies. It must also
report why.

## Multi-Node Communication

Distributed design must assume multi-node execution from the beginning. A
runtime that only works well on one NVLink/NVSwitch node is incomplete.

The planner must model:

- rank placement: node id, local rank, device id, and network locality
- topology tiers: intra-GPU, intra-node, inter-node, and storage/checkpoint paths
- communication type: P2P, all-gather, reduce-scatter, all-reduce, all-to-all,
  broadcast, and host-mediated fallback
- communication frequency: per gate, per layer, per observable, per backward
  step, or per optimizer step
- payload size and dtype
- overlap potential between communication and compute
- heterogeneous clusters, for example mixed A100/A800 nodes

Execution planners should prefer layouts that keep high-frequency boundary
communication inside a node and push cross-node communication into lower
frequency reductions, sliced contractions, or checkpoint boundaries whenever the
algorithm allows it.

Multi-node benchmark JSON must separate:

```json
{
  "node_count": 2,
  "ranks_per_node": {},
  "rank_placement": [],
  "intra_node_communication_bytes": 0,
  "inter_node_communication_bytes": 0,
  "collective_counts": {},
  "p2p_counts": {},
  "communication_compute_overlap": false,
  "network_backend": "nccl"
}
```

No distributed runtime should be considered complete until it has a preflight
diagnostic that validates rank placement, NCCL/Gloo reachability, bandwidth
sanity, timeout behavior, and collective correctness across nodes.

## Distributed Gradients

Forward-only sharding is not enough for quantum AI. Training requires distributed
gradients.

Required long-term targets:

- sharded statevector adjoint/backprop
- sharded MPS backprop without full-MPS replay
- sliced/partitioned TN reverse-mode contraction
- checkpointing and rematerialization
- reduce-scatter/all-reduce gradient aggregation
- PyTorch DDP/FSDP integration without hiding quantum replication

A path where forward is sharded but backward reconstructs a full state must be
reported as hybrid or incomplete.

## PyTorch + JAX Hybrid Role

FlagQuantum's hybrid design target:

- PyTorch is the main training interface.
- JAX is a quantum kernel accelerator.
- DLPack bridges tensors with minimal copies.
- `torch.autograd.Function` owns cross-framework gradient boundaries.
- `torchrun`, DDP, FSDP, and DTensor manage distributed training.
- FlagQuantum owns IR, sharding, communication, memory, and deployment semantics.

JAX rank-local kernels are valuable for single-device acceleration and data
parallel training, but they are not distributed statevector/MPS/TN capacity
scaling unless the quantum state or contraction itself is sharded across ranks.

## Memory And Communication Evidence

Every scalability benchmark must report:

- distribution semantics
- per-rank owned state/tensors/slices/tasks
- per-rank peak memory
- peak intermediate tensor size
- communication bytes and collective/P2P counts
- single-GPU expected OOM or memory-budget failure when claiming capacity
  expansion
- correctness against a smaller local reference when feasible
- weak-scaling and strong-scaling configuration

Benchmark JSON must include:

```json
{
  "distribution_semantics": "sharded_across_ranks",
  "claim_evidence_type": "production_training_benchmark",
  "scalability_claim_allowed": true,
  "local_memory_bytes_by_rank": [],
  "communication_bytes": 0,
  "peak_intermediate_bytes": 0,
  "single_gpu_expected_oom": true,
  "capacity_baseline_device": "single_gpu_24gb",
  "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
  "optimizer_update_semantics": "sharded_across_ranks",
  "training_step_count": 1,
  "gradient_distribution_semantics": "sharded_across_ranks"
}
```

Replicated benchmarks must set:

```json
{
  "scalability_claim_allowed": false
}
```

## Deployment Loop

FlagQuantum should connect training to quantum hardware:

- train a parameterized circuit on AI chips/GPU clusters
- freeze optimized parameters into the IR
- lower the IR to a target gate set
- perform layout, routing, calibration-aware compilation, and noise-aware
  optimization
- submit to quantum cloud or real hardware
- parse measurement counts
- support inference, validation, and calibration feedback

This closes the loop from quantum AI training to quantum deployment.

## Product Standard

FlagQuantum should be judged by whether it can honestly answer:

- Can this single quantum AI workload fit on one device?
- If not, how is it sharded?
- What does each rank own?
- What communication is required?
- Is backward also sharded?
- What precision is lost through truncation or slicing?
- Can trained parameters deploy to real quantum hardware?

International-level distributed quantum AI means one large quantum AI task can
be automatically partitioned, trained, optimized, benchmarked, and deployed with
clear memory, communication, precision, and gradient semantics.
