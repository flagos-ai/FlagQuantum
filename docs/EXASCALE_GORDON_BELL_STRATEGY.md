# FlagQuantum Exascale And Gordon Bell Strategy

> **Future intent only.** Aspirations here are not current capability or evidence.

## Strategic Objective

FlagQuantum is intended to become a flagship, production-grade, distributed
quantum AI framework capable of running scientifically meaningful workloads on
clusters ranging from one accelerator to thousands, tens of thousands, and
eventually one hundred thousand accelerators.

The long-term research objective is to produce work of Gordon Bell Prize
caliber. This is a direction and engineering constraint, not a current
capability claim. Accelerator count alone is not sufficient: a qualifying
result must combine an important scientific problem, a new algorithmic or
representational contribution, a new large-scale systems method, high sustained
performance, strong evidence, and a reproducible scientific outcome.

## Product And HPC North Star

FlagQuantum must satisfy three scales without maintaining three unrelated
frameworks:

1. **Developer scale:** correct, simple CPU and single-accelerator workflows.
2. **Production scale:** reliable multi-GPU and multi-node quantum AI training.
3. **Leadership scale:** topology-aware, failure-tolerant execution on
   10,000-100,000 accelerators.

The same versioned IR, `fq.Module`, planner, result model, checkpoint format,
and provenance system should span these scales. Backend policy may change;
user model code should not be rewritten for each scale.

## Mandatory Runtime Position

PyTorch-native execution is the production control plane:

- PyTorch tensors and autograd define the required training semantics.
- `torchrun` and `torch.distributed` provide the mandatory distributed path.
- DDP, FSDP, DTensor, `torch.optim`, `state_dict`, and checkpoint conventions
  are integrated where their ownership semantics fit.
- JAX, Triton, cuQuantum, vendor libraries, and custom kernels are optional
  acceleration providers. They must not be prerequisites for core correctness,
  orchestration, recovery, or deployment.

At leadership scale, FlagQuantum may require specialized process-group,
collective, scheduler, and kernel extensions. These extensions must preserve
the stable PyTorch-facing product contract.

## Why Small-Scale Designs Do Not Automatically Scale

A two-rank implementation cannot be extrapolated directly to 100,000 ranks.
At leadership scale:

- global barriers amplify stragglers and failures;
- flat all-to-all communication can dominate runtime;
- centralized planners and metadata services become bottlenecks;
- a tiny per-rank memory leak becomes cluster-scale waste;
- frequent global checkpointing becomes infeasible;
- one failed worker is normal, not exceptional;
- rank numbering is less useful than topology and failure-domain placement;
- reproducibility requires automated capture, not hand-authored summaries.

Therefore no production design may assume a single node, uniform links, a
reliable global barrier, one coordinator, or full-state visibility.

## Scale Ladder

| Level | Target | Required proof |
| --- | ---: | --- |
| L0 | 1 device | numerical correctness, local performance, clean lifecycle |
| L1 | 2-8 devices | true sharding, forward/backward/optimizer, measured traffic |
| L2 | 16-256 devices | topology-aware placement, strong/weak scaling, recovery |
| L3 | 256-4,096 devices | hierarchical collectives, distributed planning, checkpoint tiers |
| L4 | 4,096-16,384 devices | sustained production run, bounded metadata, failure injection |
| L5 | 16,384-100,000 devices | decentralized control, elastic recovery, leadership-scale science |

Passing one level does not imply the next. Every level requires measured
hardware execution, raw artifacts, numerical validation, and a documented
supported workload envelope.

## Multi-Dimensional Parallelism

Leadership-scale quantum AI cannot rely on one parallel dimension. The planner
must compose and report:

- data/sample parallelism;
- quantum-state amplitude sharding;
- MPS site and bond sharding;
- TN slice, subgraph, and intermediate-tensor sharding;
- observable/Hamiltonian-term parallelism;
- parameter and optimizer-state sharding;
- pipeline parallelism across circuit or contraction regions;
- trajectory/shot parallelism;
- ensemble and hyperparameter parallelism.

The planner must prevent double gradient reduction, hidden replication, and
parallel layouts whose communication cost exceeds their useful computation.

## Hierarchical Topology Model

The runtime must model at least:

```text
accelerator
  -> host/socket
  -> node
  -> high-bandwidth island
  -> rack/failure domain
  -> cluster partition
  -> site
```

Plans must distinguish intra-device, intra-node, intra-island, inter-rack, and
inter-site traffic. Rank placement, collective construction, checkpoint
replication, and recovery groups must respect bandwidth, latency, and failure
domains.

Flat global collectives are allowed only when measurements show they are
appropriate. Preferred techniques include hierarchical reduction, locality
preserving exchange, communication-avoiding gate schedules, asynchronous
progress, batching, overlap, and bounded neighborhood protocols.

## Backend-Specific Exascale Direction

### Distributed Statevector

Statevector remains the exact, general reference path. Its capacity grows only
logarithmically with accelerator count: doubling total memory typically adds
one qubit. Its leadership-scale value is therefore exact simulation of larger
high-entanglement circuits and a rigorous reference for compressed methods.

Required research directions:

- hierarchical amplitude and qubit-address sharding;
- communication-avoiding gate fusion and logical-qubit remapping;
- topology-aware pair exchange and all-to-all decomposition;
- sharded adjoint differentiation and checkpoint scheduling;
- overlap of gate execution, transfers, and checkpoint I/O;
- failure-domain-aware shard recovery without global state replication.

### Distributed MPS

MPS is the primary path toward structured thousand-, million-, and potentially
larger-qubit quantum AI workloads when entanglement remains compressible. At
fixed bond dimension, capacity can grow approximately linearly with aggregate
memory, but compute, communication, and load imbalance grow with dynamic bond
dimensions.

Required research directions:

- dynamic site/bond ownership over many topology levels;
- distributed canonicalization, truncation, and reverse mode;
- adaptive repartitioning based on measured bond cost;
- local-neighborhood communication rather than global collectives;
- error-budget-aware training and scientific convergence;
- multilevel checkpointing for variable-shape tensors;
- algorithms that control entanglement growth rather than only storing it.

Claims such as "1,000 qubits" must state topology, depth, maximum/observed bond
dimension, truncation policy, discarded weight, observables, gradient semantics,
and hardware. They never imply arbitrary 1,000-qubit circuits.

### Distributed Tensor Networks

TN is the primary path for observable-oriented workloads whose contraction
structure is favorable. Qubit count alone is not a capacity measure; effective
treewidth, depth, path, slicing, output objective, and intermediate memory are
required.

Required research directions:

- distributed contraction-path search and cost-model refinement;
- decentralized slice/task DAG scheduling and work stealing;
- subgraph and intermediate-tensor partitioning;
- reverse contraction with environment reuse;
- idempotent task retry and partial-result recovery;
- heterogeneous contraction providers;
- locality-aware reductions and persistent path/slice manifests.

## Thousand-Qubit And Beyond Capability Classes

FlagQuantum must publish separate classes rather than one ambiguous maximum:

| Class | Representation | Intended scale | Meaning |
| --- | --- | --- | --- |
| Exact general | SV | tens of qubits | complete amplitude state, hardware limited |
| Structured low-entanglement | MPS | 1,000+ qubits | bond/error-bounded state and training |
| Observable-oriented | TN | 1,000+ qubits | target contraction under path/treewidth limits |
| Hardware deployment | QPU/cloud | provider limited | shots/results, not classical full-state simulation |

Future milestones should include a maintained 1,000-qubit MPS training suite
and a 1,000-qubit TN observable suite before expanding the public scale claim.
Million-qubit goals require evidence that computation and communication—not
only tensor storage—complete within a scientifically useful time.

## Reliability At 10,000-100,000 Accelerators

The system must treat component failure as normal. Required capabilities:

- failure-domain-aware placement;
- bounded, decentralized metadata;
- heartbeat and failure detection without a single global coordinator;
- local recovery groups and elastic rank replacement where mathematically safe;
- multilevel checkpointing to device, node-local, burst buffer, and durable
  storage;
- incremental and asynchronous checkpoints;
- idempotent TN task retry;
- deterministic replay metadata for SV/MPS segments;
- cancellation and cleanup that do not require every failed rank to respond;
- preservation of scientific validity after recovery.

Recovery semantics must be backend specific. A TN slice can often be retried;
an SV collective segment may require coordinated rollback; an MPS boundary
failure may require recovery of neighboring ownership and canonical state.

## Performance And Efficiency Metrics

Leadership-scale reports must include more than wall time:

- time to solution and time per training step;
- sustained useful FLOP/s and kernel efficiency where meaningful;
- strong-scaling and weak-scaling efficiency;
- capacity expansion relative to a measured single-device baseline;
- communication bytes, time, and overlap by topology tier;
- peak and persistent memory per rank plus imbalance;
- scheduler and synchronization overhead;
- checkpoint/recovery overhead and injected-failure results;
- energy to solution when platform telemetry is available;
- numerical error, scientific convergence, and output quality;
- total allocated versus effectively contributing accelerators.

Peak theoretical FLOP/s multiplied by accelerator count is never an accepted
performance result.

## Gordon Bell Research Program

A credible program requires five coupled workstreams.

### 1. Scientific Grand Challenge

Select a quantum AI or quantum many-body problem with a defensible scientific
output that cannot be solved at the target fidelity or scale by existing
methods. The problem must motivate the chosen SV, MPS, TN, or hybrid method.

### 2. Algorithmic Innovation

Develop a contribution such as communication-avoiding differentiation,
entanglement-adaptive training, hybrid MPS/TN decomposition, distributed
environment reuse, or a new error-controlled quantum AI formulation.

### 3. Systems Innovation

Demonstrate topology-aware decomposition, scalable scheduling, communication
overlap, bounded metadata, multilevel recovery, and high hardware utilization.

### 4. Scientific Validation

Validate against exact references at smaller scales, convergence studies,
independent formulations, and domain-specific observables. A fast result without
scientific credibility is insufficient.

### 5. Reproducible Performance Evidence

Retain source commit, workload manifests, environment, compiler/runtime
configuration, hardware topology, raw logs, profiler traces, power data where
available, numerical outputs, and artifact checksums.

## Decision Gates

Before committing a leadership-scale allocation, FlagQuantum must pass:

1. **Science gate:** the question and success metric are externally defensible.
2. **Algorithm gate:** complexity and error behavior justify scale-out.
3. **Single-node gate:** kernels are efficient and numerically certified.
4. **Hundred-device gate:** no fundamental communication or imbalance wall.
5. **Thousand-device gate:** scheduler, checkpoint, and recovery remain bounded.
6. **Leadership gate:** projected time/energy to solution and scientific value
   justify the allocation.

## Claim Boundary

The following statements are prohibited without matching evidence:

- "supports 100,000 GPUs" based only on a planner or launcher;
- "exascale" based on peak hardware capability rather than measured execution;
- "1,000-qubit simulation" without representation and structure limits;
- "distributed scalability" for replicated data-parallel workloads;
- "Gordon Bell ready" without a selected scientific problem and measured
  systems result;
- extrapolated scaling curves presented as executed measurements.

Plans and projections are valuable, but must be labeled as such. Public claims
must identify the exact commit, workload, representation, precision, topology,
accelerator count, execution duration, numerical result, and failure behavior.

## Relationship To Product Maturity

The capability maturity policy establishes the correctness and product
foundation. Leadership-scale work extends that foundation through the scale
ladder, scientific co-design, resilient runtime, and submission-quality
evidence. It does not bypass earlier evidence gates.
