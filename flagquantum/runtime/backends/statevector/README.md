# Statevector Runtime

This package owns statevector execution planning and orchestration. It supports
the local CPU reference path and the distributed execution path while keeping
the public user entry points at `fq.Circuit`, `fq.plan`, and `fq.run`.

The package is internal. Do not expose its planning, sharding, communication,
checkpoint, or evidence objects as public API.

## Responsibilities

- `models.py`: immutable execution records and evidence.
- `planning.py`: topology, sharding, memory, communication, and fusion plans.
- `local_execution.py`: local CPU execution used to validate distributed
  semantics, including rank-local state and index handling.
- `forward.py`: communication-aware forward primitives, workspace handling,
  and gate-dispatch orchestration.
- `forward_executor.py`: the authoritative distributed forward execution loop.
- `layout.py`: logical-to-physical wire layout and swap scheduling.
- `checkpointing.py`: checkpoint policy and checkpoint selection.
- `reverse.py`: the public reverse-mode boundary, validation, and evidence.
- `reverse_adjoint.py`: adjoint replay, communication, and backward
  orchestration.
- `gradient_reduction.py`: distributed gradient-reduction lifecycle.
- `training.py`: distributed optimizer and training lifecycle.
- `kernel_dispatch.py`: kernel selection decisions and their evidence.
- `environment.py` and `errors.py`: runtime configuration parsing and
  statevector-specific failures.

`noisy.py` and the `split_real_imag*` modules are specialized execution paths.
They are not the default local CPU vertical slice.

`noisy.py` is the Runtime boundary for statevector trajectories. It owns input
validation, global trajectory ownership and random streams, batching,
checkpoint/restart, retry, online and collective statistics, readout-result
processing, and result/evidence assembly. It invokes one already-lowered batch
through `simulation.noisy_statevector`; all gate, channel, normalization,
observable, and Pauli fast-path numerics are implemented there. Do not move the
remaining lifecycle code into Simulation or split it into pass-through helper
objects. The direct raw-program entry retains Compiler lowering only for its
existing compatibility behavior. Planned execution calls the private lowered-IR
entry directly, so it executes the channels sealed into the plan and never
invokes Compiler lowering a second time. Keep the public wrapper signature
unchanged until an approved compatibility migration can relocate it.

`split_real_imag.py` is the P0/P1 Runtime adapter. It owns accepted-scope
validation, parameter binding and shift scheduling, observable parsing,
platform preflight, precision/result records, and conformance reporting. The
FP32 real/imag gate matrices, state evolution, and Pauli-term expectation live
in `simulation.split_real_imag_statevector`. This is the P1 stopping point: do
not move Runtime result types into Simulation or create a parallel observable
contract merely to shorten the adapter.

`split_real_imag_precision.py` is the P2 Runtime adapter. It owns authorization
of the fixed precision plan and certified accuracy envelope, parameter-shift
scheduling, result records, host-side report reconstruction, and conformance
comparison. It composes the established Double-Single value type and reduction
primitives; Pauli-term tensor mathematics remains in
`simulation.split_real_imag_statevector`. This is the P2 stopping point: do not
move precision contracts, result schemas, reference comparisons, or
conformance reports into Simulation, and do not duplicate the Double-Single
term kernel in Runtime.

`split_real_imag_double_single.py` and
`split_real_imag_device_double_single.py` are the P3 and P4 Runtime adapters.
They own parameter binding and shifts, precision authorization, platform
preflight, the deliberate choice between host and device gate encoding, route
auditing, and result/conformance evidence. Both stream encoded gates into the
same `simulation.double_single_statevector` execution and Pauli-reduction
primitives. This is the P3/P4 stopping point: keep the adapters separate because
their host-ingestion claims differ, and do not move their contracts, provider
checks, or evidence records into Simulation merely to remove Runtime code.

`split_real_imag_autograd.py` and
`split_real_imag_autograd_optimizer.py` form the P5 Runtime boundary. They own
the PyTorch autograd bridge, parameter ownership and ordering, delivered
gradient precision, learning-rate validation, optimizer-step state, and
training/conformance evidence. P4 supplies the explicit numerical gradient and
the optimizer composes established Double-Single arithmetic. The single SGD
update expression is not a separate Simulation kernel: do not add a helper or
optimizer abstraction until a second concrete consumer requires one.

## Local execution stopping point

`local_execution.py` is not a second numerical-kernel authority. Its shard
initialization and global-index construction interpret Runtime-owned plan and
ownership records; its gate helpers adapt those records to the tensor kernels
in `simulation.statevector.operations`; its local simulator, dry run, launch spec, and
transport probe assemble Runtime-owned results and evidence. The shared wire
mask, gate-basis offset, diagonal application, and basis-vector update math live
only in Simulation.

Do not move this file wholesale into Simulation: that would force Simulation to
import Runtime plans, policies, and result models or require duplicate mirror
contracts. A later physical split is justified only when the protected root
compatibility exports can keep their behavior while orchestration remains in
Runtime and a concrete second consumer needs a Runtime-neutral numerical entry.
Distributed forward dispatch also consumes Simulation's diagonal-gate
classification directly; Runtime must not maintain a second execution copy.
The planner's narrower `communication-local` set is a conservative scheduling
policy, not a copy of the numerical classification. Likewise, JAX conversion
classifies only parameterized matrices after construction. Keep these names and
scopes explicit; widening the planner set changes plan and communication
evidence and therefore requires dedicated behavioral tests rather than a
mechanical deduplication.

## Non-responsibilities

This package does not own:

- Core circuit or IR semantics and stable public APIs;
- compiler lowering, optimization, or scheduling semantics;
- numerical gate, expectation, or adjoint kernels, which belong in
  `flagquantum/simulation/`;
- provider and platform identity, capability discovery, or fallback policy;
- benchmark baselines or release-performance claims.

Runtime decides when and where work runs, how ranks communicate, and what
execution evidence is returned. Simulation computes the numerical result.
Simulation must not import Runtime. Unsupported execution and CPU fallback
must be explicit; silent fallback is forbidden.

## Ten-minute change path

Start with the smallest authoritative file:

| Change | Start here |
| --- | --- |
| Topology, sharding, memory, or communication plan | `planning.py` |
| Rank-local indexing or local CPU reference execution | `local_execution.py` |
| Distributed forward loop | `forward_executor.py` |
| Forward communication primitive or workspace | `forward.py` |
| Wire placement or swap scheduling | `layout.py` |
| Checkpoint or reverse-mode policy and result | `checkpointing.py`, `reverse.py` |
| Adjoint replay or backward communication | `reverse_adjoint.py` |
| Gradient collectives | `gradient_reduction.py` |
| Distributed training lifecycle | `training.py` |
| Numerical kernel or precision implementation | `flagquantum/simulation/` |

A normal feature should principally change one domain. If a change repeatedly
requires edits across Core, Compiler, Runtime, Simulation, and Providers,
recheck the boundary before adding another cross-layer object.

## Verification

For every change, keep the local CPU vertical slice green:

```bash
python -m pytest tests/integration/test_cpu_vertical_slice.py -q
```

For forward or reverse execution changes, also run:

```bash
python -m pytest \
  tests/unit/test_issue041_statevector_forward.py \
  tests/unit/test_issue042_statevector_reverse.py -q
```

Use `tests/test_distributed_statevector.py` for planning and rank-index changes,
and `tests/unit/test_issue043_statevector_training.py` for training changes.
CUDA and `torchrun` tests are required when the affected path and test
environment support them; they do not replace the local CPU checks.
