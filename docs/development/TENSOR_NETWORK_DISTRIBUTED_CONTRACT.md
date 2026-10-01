# Tensor-network distributed surface contract

`flagquantum.experimental.distributed.train_distributed_tensor_network` is the
entry point for training one logical tensor-network workload across ranks. It is
a discoverable experimental export, added under an explicit product-owner
directive to bring distributed tensor-network execution to the same maturity as
the statevector and MPS distributed workflows.

The addition is confined to the `distributed` second-level namespace. The stable
root surface, the four low-level
`distributed_tensor_network_{amplitude,amplitudes,expectation,expectations}`
executors, and the slice, shard and evidence records remain non-discoverable
implementation detail.

## Why this is a user workflow and not an implementation step

The excluded categories of the experimental surface reject backend-specific
executors and workflow implementation steps such as export, routing and
deployment. This name is neither. It is the task-level operation a user invokes
to train a tensor-network circuit, and it is the direct analogue of the
`train_distributed_mps` and `train_distributed_statevector` names that the
namespace already publishes. The four amplitude and expectation executors stay
hidden precisely because they are the steps of that workflow rather than
workflows themselves.

## Contract

The function accepts a circuit or FlagQuantum IR, a parameter tensor, a step
count, an observable, and an explicit slice plan, and returns a
`ShardedTensorNetworkTrainingResult`. Its declared behaviour is:

1. partition the contraction over a declared slice plan whose task ownership is
   reported per rank;
2. reduce each rank's partial observable contribution to one expectation that
   every rank reports identically, refusing the reduction when the plan places
   every task on one rank;
3. produce gradients that agree with a single-device adjoint to the declared
   tolerance, with each rank's own contribution reported separately so a dropped
   rank is detectable;
4. distribute the optimizer step, and write one integrity-checked checkpoint per
   rank that a restarted run resumes from;
5. fail closed on an unsupported runtime, on a plan that cannot satisfy the
   declared memory budget, and on a checkpoint generation whose payload
   disagrees with its recorded digest.

`steps` is the absolute total target step count for the trajectory, including
steps already taken by an interrupted leg. A resumed leg reports
`completed_steps == steps` while its loss record holds only the steps that leg
ran.

## Claims this export does not make

Distributed summaries carry `scalability_claim_allowed = false` and
`release_gate_allowed = false`, and attach their blockers, because a two-node
pair with one device per host is not a scaling result. Slicing is a declared cut,
not an automatic one: the automatic slicer chooses a cut by peak memory, which on
the recorded circuit is a label carried only by state-copy nodes and therefore
yields ranks whose partial is exactly zero. Slicing such a label would let the
run report `sharded_across_ranks` while one rank performed the arithmetic, so the
slicer excludes those labels and the workload declares its cut instead.

RDMA was not used or tested, cut width was not swept, no production performance
was measured, and the recorded evidence covers one circuit with toy parameters.
Cross-host behavior as the cut widens or as nodes are added is outside this
contract.

## Verified implementation

The implementation provides:

1. the distributed contraction, gradient and optimizer legs described above;
2. a two-node probe that checks every leg against the exact complex128
   statevector and records its own scope and blockers;
3. a checked-in pair-level artifact whose run reports accelerator runtime
   evidence rather than a CPU collective;
4. team contract tests pinning the artifact's numbers, placement integrity and
   surviving blockers; and
5. a multinode verification tier that runs the probe alongside the statevector
   and MPS workloads.
