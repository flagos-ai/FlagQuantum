# FlagQuantum Distributed Scalability Principles

FlagQuantum uses the word "distributed" only when a single quantum workload is
partitioned across ranks so that multi-GPU or multi-node execution expands the
problem size, memory capacity, or time-to-solution beyond one device.

## Non-Negotiable Semantics

- `sharded_across_ranks` means one logical quantum workload is split across
  ranks. Each rank owns a distinct shard, slice, tensor block, contraction
  subgraph, or state segment.
- `replicated_per_rank` means every rank runs the same full workload
  independently. This is useful for smoke tests, environment checks, and
  throughput replication, but it is not scalability evidence.
- A benchmark may not claim multi-GPU capacity expansion unless its JSON output
  explicitly reports `distribution_semantics: "sharded_across_ranks"` and a
  release-gate `claim_evidence_type`.
- Replicated benchmarks must report `scalability_claim_allowed: false`.
- Plan and preflight summaries should report `sharding_plan_available: true`
  instead of treating a plan object as release evidence.
- Single-device peak-performance benchmarks are first-class results, but they
  should be reported as `single_device_fast_path` or an equivalent local
  semantic and must not be forced through distributed initialization.

## Why This Matters

The motivation for distributed MPS/TN/statevector simulation is that one GPU may
not fit the target workload. Copying the same circuit onto multiple GPUs does
not solve that problem. It only shows that the cluster can run multiple
independent copies.

Distributed scalability work must not tax the single-device fast path. Local
CPU/GPU execution should keep its fastest kernel, autograd, JIT, and memory
layout choices, with no communication setup unless the user explicitly selects a
distributed mode.

## Ease-Of-Use Contract

FlagQuantum must stay easy to use at both ends of the product:

- non-production users get a simple local fast path with no distributed setup;
- production, enterprise, and industrial users get real sharded scale-out;
- both use the same `import flagquantum as fq`, `fq.Circuit`, `run`, and `plan`
  concepts;
- advanced distributed controls are opt-in, discoverable, and diagnosable.

An implementation that is powerful but hard to use is not product-ready. A
benchmark that is impressive but requires hidden manual steps should document
those steps and should eventually be wrapped by a higher-level FlagQuantum API.

Local development distributed execution is valid only when it preserves
production semantics. A CPU LocalTensor or local pmap simulator must use the
same Circuit/IR, rank ownership, shard/task layout, and communication signature
as the production distributed backend. Treat
`flagquantum.runtime.parity.require_development_production_parity(...)` as the guardrail before relying
on a local distributed test to predict production behavior.

## Required Evidence For Scalability Claims

A valid distributed MPS/TN benchmark must report:

- distribution semantics: `sharded_across_ranks`
- per-rank owned tensors, wires, slices, or contraction tasks
- per-rank peak memory
- communication volume and collective/P2P counts
- node count, rank placement, and whether communication is intra-node or
  inter-node
- correctness versus a smaller local reference when feasible
- weak-scaling and strong-scaling settings
- at least one capacity case where the single-GPU baseline is expected to fail
  or exceed a stated memory budget

For multi-node runs, communication must be planned and reported separately for
intra-node and inter-node traffic. High-frequency boundary updates should be
kept within a node whenever possible; cross-node traffic should be made explicit
and minimized through layout, slicing, batching, checkpointing, or collective
aggregation.

## Runtime Metadata Requirements

Every distributed plan or result object must expose enough metadata to audit
whether a single workload is genuinely sharded:

- `distribution_semantics`
- `claim_evidence_type`
- `scalability_claim_allowed`
- `world_size`, `local_world_size`, and `node_count`
- `rank_placement` when attached to a concrete torch.distributed context
- `communication_tiers` or explicit `intra_node_communication_bytes` and
  `inter_node_communication_bytes`
- any `scalability_blockers` that prevent a scalability claim

When a communication pattern is collective and the exact physical route depends
on NCCL/Gloo, the runtime must report that as topology-dependent instead of
inventing exact inter-node bytes.

## Current Benchmark Classification

- `benchmarks/capacity_case.py` and `benchmarks/capacity_sweep.py` are
  `replicated_per_rank` smoke/capacity checks. They must not be used as
  evidence that one large MPS/TN problem is sharded over multiple GPUs.
- True distributed scalability work must use dedicated sharded MPS/TN
  benchmarks and must fail closed if the execution path silently becomes
  replicated.

## Fail-Closed Audit

Use `flagquantum.runtime.audit.release_policy.require_distributed_scalability(payload)` when a CI job, release note,
or benchmark promotion requires release-grade sharded scalability evidence. It
raises if the payload is replicated, incomplete, missing communication evidence,
not allowed to make a scalability claim, or only plan/preflight evidence.

Use the benchmark scanner before promoting JSON files:

```bash
python benchmarks/audit_results.py --input benchmarks/results
python benchmarks/audit_results.py --input benchmarks/results/scalability --require-scalability
```

The second command should pass only for `benchmarks/results/scalability/`
payloads with `claim_evidence_type="production_training_benchmark"` or an
explicit `release_payload`, `single_gpu_expected_oom=True`, capacity baseline
failure details, and sharded optimizer-update evidence. Generated audit summary
JSON files are excluded from release-gate scans.
