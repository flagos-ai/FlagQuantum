# API change proposal: Ascend execution for native dynamics

## Status

Approved by the repository owner on 2026-10-30 through explicit approval of this proposal in the task conversation. Implementation and hardware acceptance are in progress.

## Problem and affected interfaces

The Ascend acceptance run covers static statevector, MPS, tensor networks,
density matrices, gradients, training, TEBD and two-host sharded training.
That evidence does not establish support for the entire product. Native Ising
braid execution and continuous-time Lindblad/Rydberg execution explicitly reject
accelerator devices before numerical execution.

The affected existing interfaces are `fq.plan`/`fq.run` with
`IsingBraidProgram`, and `flagquantum.lindblad.plan`/`run` with numerical models
or `RydbergProgram`. Existing native presence and sampled instrument execution
are included where their public entry points consume these plans.

## Proposed additive behavior

Accept `ExecutionOptions(device="flagos:0")` and other available FlagOS device
indices for these existing workflows through Torch-FL on Ascend 910C. CPU remains
the default. Preserve all function signatures, exports, dtype defaults,
physical model semantics, approximation rules and gradient requirements.

Device admission must follow provider activation and actual operator preflight.
An unsupported operator or unavailable device must produce an actionable error;
it must not silently execute numerical evolution on CPU. Explicit checkpoint,
serialization and bounded diagnostic copies remain observable host operations.

Ising exchange/dephasing, Lindblad evolution, Rydberg controls and sampled
presence must preserve their respective representation and physical semantics.
Differentiable workflows must retain parameter gradients on the requested
device. Sampling must retain deterministic seeds without consuming unrelated
caller RNG state. A device change must appear in execution metadata and plan
identity; device-local buffers must not be mislabeled as CPU results.

## Compatibility and serialization

Existing CPU plans and fixtures remain readable, with unchanged numerical
meaning and existing identity inputs. Existing schema versions, public fields
and protected snapshots must not be regenerated as a repair. If the current
schema cannot encode accelerator placement without ambiguity, stop that part
and propose a separate versioned serialization change before implementation.

New accelerator plans are admitted only after their existing device field and
identity can represent the request accurately. Older readers may reject an
unsupported device explicitly; silently rewriting it to CPU is prohibited.

## Acceptance

1. Keep the current CPU contract suite passing without weakening tolerances.
2. Compare device results with independent CPU references in complex128 and
   every additional precision the existing domain accepts.
3. Compare control gradients and multi-step optimizer trajectories for
   differentiable dynamics.
4. Test plan/replay equivalence, declared result devices, seeded instruments,
   presence records and RNG restoration where applicable.
5. Audit runtime fallback logs and record exact provider/dependency artifacts.
6. Run public API snapshot and candidate contract checks unchanged.

This proposal authorizes an additive device capability, not a release claim or
automatic support promotion. The README must distinguish tested functions from
pending acceptance until the complete matrix passes.

## Recorded authorization

The owner authorized implementation and testing of this additional FlagOS
device capability. Approval does not authorize removal or weakening of existing CPU,
physical-model, serialization or public API contracts.
