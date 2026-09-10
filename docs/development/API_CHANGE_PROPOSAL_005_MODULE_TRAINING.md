# API Change Proposal 005: Module and Training Contract Consolidation

## Status

**Contract frozen — core semantics and the machine contract received separate freeze approval.**

- Target: first public alpha.
- Affected interfaces: `fq.Module`, `fq.train`, `fq.TrainingResult`,
  `fq.ExecutionResult`, and `flagquantum.training`.
- Root name changes: none.
- Authorization: the API owner explicitly authorized Proposal 005 implementation
  on 2026-09-01. Implementation authorization alone did not freeze the entire API.
- Freeze record: on 2026-09-01, the API owner explicitly approved the exact
  Proposal 005 contract bound by the review packet. This is not the overall first
  public alpha freeze.

## Core Position

FlagQuantum remains PyTorch-first instead of reinventing a training framework:

```text
module(inputs)          -> autograd-compatible Tensor
module.execute(inputs)  -> ExecutionResult
fq.train(module, ...)   -> TrainingResult
```

`fq.run` accepts Circuit, CircuitIR, or ExecutionPlan, not Module. Module does not
add a synonymous `run()` method that would let forward/execute/run diverge.

## Module Contract

### Forward and Execute

- `forward()` always returns `torch.Tensor`, directly usable in PyTorch models and autograd.
- `execute()` always returns `ExecutionResult`.
- Equal inputs, parameters, policy, and module state produce equal numerical values.
- `ExecutionResult.require_value()` is the fail-closed accessor for Module values.
- Local PyTorch fast paths may avoid constructing diagnostics, but preserve
  observable, batch, dtype, gradient, and backend-selection semantics.

### Inputs, Parameters, and Batches

- `parameters=None` uses Module-owned parameters.
- Explicit parameters apply only to this call and do not replace Module parameter ownership.
- Two-argument builders receive `(parameters, inputs)`; one-argument builders
  receive only parameters.
- FlagQuantum does not implicitly construct an input-batch × parameter-batch Cartesian product.
- Builders and selected backends must produce compatible, broadcastable tensor
  shapes or fail explicitly.
- `RuntimePolicy` alone determines observables; forward and execute do not use
  different defaults.

## `fq.train` Boundary

Provide a small, stable loop without replicating Lightning/Trainer in the root API:

- The caller creates and owns the optimizer.
- `fq.train` performs `zero_grad → execute → objective → backward → step`.
- The objective receives `execution.require_value()`.
- Each callback receives a one-based step, float loss, and detached ExecutionResult.
- `TrainingResult.last_execution` must be detached.
- `fq.train` does not implement checkpointing, resume, early stopping, or validation loops.
- Callers, callbacks, or future independent Trainer extensions compose those lifecycles.

Distributed training remains in `flagquantum.experimental.distributed` until it
meets the same semantics; it does not enter the stable root.

## TrainingResult and Diagnostics

`TrainingResult` has exactly four fields:

```text
losses, completed_steps, last_execution, optimizer
```

It guarantees `len(losses) == completed_steps`, provides `final_loss`, and uses:

```text
schema  = flagquantum.training_result.summary
version = 1.0
```

`ExecutionResult.diagnostics()` returns a versioned envelope:

```text
schema, version, metrics, provenance, runtime, compatibility
```

Keys within the four sections may grow compatibly. User logic should rely on
stable accessors, not incidental backend diagnostic keys.

## Checkpoint and Resume Boundary

`Module.save_checkpoint/load_checkpoint`, rather than `fq.train`, owns checkpoints:

- Save accepts a nonnegative integer seed, avoiding hidden SeedContract construction
  in public usage.
- Files are versioned PyTorch training state written through atomic replacement,
  not a promised cross-language exchange format.
- Save module, optimizer, RuntimePolicy, IR/workload identity, RNG, precision, and topology.
- Workload, optimizer, precision, or topology mismatch fails closed.
- Restore returns `TrainingCheckpointRestore` with schema/version.
- Normal user types come from `flagquantum.training`; low-level save/load functions
  are no longer exposed at the root.

## PyTorch and JAX

- PyTorch supplies the stable autograd and optimizer interface.
- JAX is an optional compiled backend selected by `RuntimePolicy`, not a second Module API.
- Unauthorized fallback fails.
- Authorized fallback records requested backend, selected backend, and reason.
- Callback, TrainingResult, and checkpoint ownership do not change with backend.

Proposal 006 continues consolidating Module constructor deployment/provider state.
It removes `deployment_binding` and moves ownership to application models or deployment.

## Executable Acceptance Criteria

- [x] Forward returns Tensor; execute returns ExecutionResult.
- [x] Forward/execute values agree on statevector, MPS, and TN paths.
- [x] Forward values retain autograd.
- [x] `fq.run(module)` fails explicitly; Module has no `run` method.
- [x] `require_value()` fails closed when values are absent.
- [x] TrainingResult fields, invariant, final_loss, and summary version are protected.
- [x] Diagnostics envelopes include schema/version.
- [x] Checkpoint restore includes schema/version and preserves mismatch preflight.
- [x] Training lifecycle types move into `flagquantum.training`.
- [x] Distributed training remains experimental.
- [x] API owner separately approved contract freeze.
