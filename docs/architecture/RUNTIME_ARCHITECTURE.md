# Distributed Runtime Architecture

## PyTorch-Native Sharded Statevector Forward

The experimental distributed statevector executor consumes validated FlagQuantum IR under
`torchrun` and retains only rank-owned amplitudes. Local-qubit gates execute
without transport; gates touching one or several sharded qubits exchange bounded
gate-basis groups through `torch.distributed`. Scratch memory depends on batch
and gate width, not global state size, and requesting a reconstructed full state
fails explicitly.

Gloo execution is development semantic evidence. NCCL execution uses the same
numerical contract but is accelerator semantic evidence. Forward, reverse mode,
and owner-sharded optimizer lifecycles are implemented. Their presence does not
by itself establish production scalability: each claim still requires the
applicable workload, transport, memory, and release-gate evidence.

## PyTorch-Native Sharded Statevector Reverse Mode

The experimental distributed-observable boundary returns a global observable through a
custom `torch.autograd.Function`. Backward rematerializes rank-local forward
checkpoints, calculates parameter VJPs at each executed gate, reduces shared
parameter contributions across participating ranks, and propagates state
adjoints explicitly with the same bounded communication kernel using `U†`.
Neither checkpoints nor adjoints materialize the global state.

## Sharded Statevector Training Service

`flagquantum.experimental.distributed.train_distributed_statevector(...)` is the PyTorch-facing multi-step entry
point. It calls ordinary `loss.backward()` and native `torch.optim.SGD` or
`torch.optim.Adam`. Each parameter has one deterministic owner; only that rank
constructs its optimizer state, then broadcasts the updated parameter. Gradient
VJPs are reduced across participating ranks before the owner step.

Rank-local checkpoints store only owned parameters and optimizer state. Resume
validates world size, optimizer and ownership before broadcasting restored
values. OOM preflight, cancellation, injected lifecycle faults, bounded
timeouts, phase progress, device memory and deterministic process-group cleanup
are separate from release/scalability classification.

DDP normally replicates optimizer state and is therefore not used to claim this
owner-sharded lifecycle. FSDP or DTensor integration may replace ownership and
transport internals later, but must preserve this entrypoint, parameter identity,
checkpoint contract, and evidence semantics rather than stacking a second
optimizer lifecycle around it.

## Sharded MPS Training

`flagquantum.experimental.distributed.train_distributed_mps(...)` composes
rank-owned forward and reverse execution with optimizer and checkpoint ownership.
The implementation lives in `runtime/executors/mps`; numerical kernels live in
`simulation/mps`. Training and resume scenarios are covered in
`tests/distributed/test_mps_training_runtime.py`.

The distributed MPS reverse policy additionally supports an opt-in
`save_two_site_factorizations=True` mode. It retains each owner-local QR/SVD
split graph and pair matrix until that bond's reverse visit. Backward then
pulls the output adjoints through the saved factorization and rematerializes
only the cheaper two-site gate contraction. These tensors count against
`max_saved_bytes`; insufficient budget fails before execution rather than
silently reverting to a different checkpoint policy. Saved factorization
graphs are released immediately after use, so multi-step training does not
accumulate graph generations.

For truncated two-site splits, the singular-value boundary gap used by the
fail-closed degeneracy check is taken directly from the singular values of the
primary QR/SVD split. The runtime does not reconstruct the pair and launch a
second `svdvals` operation solely for diagnostics.

## Sharded Tensor-Network Training

`flagquantum.experimental.distributed.train_distributed_tensor_network(...)`
composes the sliced contraction, the slice-gradient sum, and optimizer and
checkpoint ownership. The implementation lives in
`runtime/executors/tensor_network`; the contraction and slicing kernels live in
`simulation/tensor_network`.

A slice plan partitions the contraction's slice tasks across ranks and the plan
is reported per rank, so an execution that placed every task on one rank is
refused rather than reported as sharded. The observable reduction is a real
collective, and every rank reports the same reduced expectation. Gradients are
different: the collective covers the expectation, not the parameter gradient, so
each rank holds a partial forward contribution and the total is formed by the
documented reduction. A caller reading a per-rank gradient must read it as a
partial.

Tensor-network training checkpoints are rank-local files holding the replicated
parameter vector: the owner all-gathers the updated parameters before writing, so
each rank's file is the whole resumed state and the two files carry the same
parameter digest. A resumed leg reports the absolute target step count while its
loss record holds only the steps that leg ran.

Slicing is a correctness property before it is a tuning knob. A label carried
only by a "state-copy" node -- a tensor with fewer than two dimensions above one
-- is already zero on every branch but one, so slicing it produces ranks whose
partial is exactly zero and can make one rank's arithmetic look like a
partition. The automatic slicer excludes those labels and selects by peak
memory; a workload that needs a specific cut declares it. Every summary reports
the labels it sliced and the task count per rank, so a reader can check that the
cut used is the cut the workload asked for, but the summary does not distinguish
a declared cut from a chosen one.

## Executor Boundaries and Evidence

Backend-neutral execution requests and records are defined in
`flagquantum.runtime.distributed.protocols`. PyTorch is the mandatory baseline executor; JAX is
an optional adapter split into independently importable statevector, MPS,
tensor-network, transport, common, and release-policy surfaces.

Execution and classification are separate:

1. an executor consumes FlagQuantum IR and returns a
   `DistributedExecutionRecord`;
2. backend-specific collectors attach raw execution measurements;
3. audit/release policy classifies the record after execution.

JAX orchestration lives in `flagquantum.runtime.executors.jax`; numerical
statevector, MPS, and tensor-network operations live in
`flagquantum.simulation.jax`. Shared distributed protocols remain independent
of those optional numerical implementations. Module budgets and dependency
restrictions are enforced by `architecture.toml` and
`tools/check_architecture.py`.

Importing the PyTorch-native executor and backend-neutral protocols must not
import or initialize JAX/XLA. JAX facade modules resolve extracted definitions
without importing JAX/XLA until an executing function explicitly requests it.
