# Approved lazy Torch training compilation boundary for #936

Status: boundary explicitly approved by the user on 2026-11-02; contract/fake
land before domain-owned implementation. PR #1017 completed #928. This separate
approval covers the narrow #936 boundary below, not automatic backend selection.

## Approved decision

The user approved a contract-first follow-up that permits **only**
`flagquantum/runtime/training.py` to import private
`compile_training_callable` from `flagquantum/compiler/training.py`.
Land the exact architecture-checker exception and this contract/fake before
domain-owned implementation commits. This is not permission for arbitrary
Runtime-to-Compiler imports, a new numerical owner or a new public API.

The function takes an ordinary Torch training-step callable and returns a lazy
Torch compiler callable. Compiler performs generic Torch capture/optimization;
it imports neither Runtime nor Simulation and implements no quantum numerics
or optimizer formulas. The first execution remains under Runtime ownership.
Simulation continues to receive canonical Core IR through existing Module
execution and retains all quantum numerical/derivative algorithms. No Compiler
artifact or second quantum IR is passed into Simulation.

Runtime owns validation, data delivery, parameter and optimizer identities,
zero_grad/backward/step ordering, logging, callbacks, numerical refusals and
checkpointing. Compiler must not own the optimizer or replace its algorithms.
Use existing Torch compilation and optimizer operations; add no dependency,
parallel optimizer registry, loss recipe registry or benchmark-only entry point.

## Existing user workflow and unchanged contracts

The first production consumer is existing `fq.train(module, optimizer=...,
objective=..., steps=..., inputs=...)`. No new flags, decorators, private bridges
or specialized user builders are required. Manual Torch loops stay supported;
their unchanged behavior must be part of conformance, and a speedup in fq.train
alone must not be described as accelerating every manual loop.

- Old and proposed backend default: PyTorch. Explicit JAX remains authoritative.
  There is no installation-driven JAX selection or backend-policy change.
- All `Module`, `train`, `RuntimePolicy`, `ExecutionOptions`, `TrainingResult`
  signatures/defaults, exports and serialized schemas stay byte-for-byte
  compatible. Existing configuration precedence remains authoritative.
- Old checkpoints are read with existing loaders; compiled callables are
  ephemeral and rebuilt. Parameters, `.grad`, optimizer slots, dtypes and tensor
  identities remain caller-owned. No state/schema migration or release bump is
  proposed for this private implementation; public API changes need a separate
  proposal. #930's automatic-selection decision remains unapproved.
- Ordinary calls outside an eligible reusable local training context remain
  eager. Existing unsupported modes/devices/derivative requirements are never
  changed to obtain compilation. No GPU/distributed claim follows from CPU work.

## Supported first slice and precise exclusions

Start with static local CPU Torch Module training, explicitly declared batch,
fixed shape/dtype/device and ordinary scalar Torch objectives. Reuse the #928
capture/invalidation boundary. Changing parameter/input values and supported
grad masks do not create value-keyed caches. Preserve finite-angle validation,
consumed pre-update IR hashes and no-grad/inference behavior.

SGD is the initial compilation candidate. Adam must retain the ordinary eager
Torch route until its compiled float64 trajectory and optimizer slots pass the
same independent tolerance. Compilation success is not numerical conformance.
Do not change a float32 counter's checkpoint dtype or silently accept degraded
float64 arithmetic to make a candidate pass.

User hooks, subclass overrides, callbacks, stateful objectives, changing optimizer
groups/options, control flow, precision/device policy changes and checkpoint
reload must either preserve the existing observable behavior or select eager
before execution with an explicit exclusion reason. In particular, invoking an
objective twice while tracing or suppressing its side effects is forbidden.
Opaque callables are not presumed pure. The fake's hooked cases select eager;
it is not proof that arbitrary hooks or objectives are compilable.

Compilation/cache eligibility is evaluated at the existing training lifecycle,
not timed/autotuned on every update. Bound caches independently of update count.
Do not enable a default optimization until cold time-to-result and recompilation
cost are measured and a short-job exclusion is reviewed. Same-backend eager
execution after exclusion must be visible through existing free runtime
diagnostics; it must not masquerade as compiled execution. Backend/device/
representation fallback still requires the existing explicit policy.

## Review-only fake and falsification

`tests/unit/test_training_compilation_contract_fake.py` swaps one ordinary training-step callable for
a Torch compiler callable using an **identity compiler backend**. The same
public Module/optimizer consumer runs before and after. Eight manual-step cases cover SGD/
Adam, batches 1/8, user forward hooks, changing trainable features, nonuniform
weights, an independent analytic loss/VJP/update reference, six updates,
no-grad evaluation and 3+3 fresh-Module checkpoint replay with consumed IR hashes.
An additional ninth case runs the same existing public `fq.train` consumer
with eager and identity-compiler execution, including retained consumed IR hashes.
SGD without user hooks uses the fake; Adam and hooks retain explicit eager
reasons. This proves a replaceable callable boundary and refusals, not numerical
code generation, whole-graph fusion, performance or production eligibility.

The separate `reproduce_optimizer_capture.py` deliberately bypasses the Adam
exclusion to test it. Its receipts retain the actual Torch version and both
trajectories; identity compilation already changes Adam float64 arithmetic.
This is an open blocker for compiled Adam, not a passing result or an allowed
precision change. Fragmented graph capture and recompilation are retained rather
than called a whole-training graph.

From the measured repository checkout, with its environment:

```bash
python -m pytest tests/unit/test_training_compilation_contract_fake.py -q
PYTHONPATH=. python artifacts/development/torch_training_contract_20261102/reproduce_optimizer_capture.py --output /tmp/optimizer-capture.json
```

## Required production acceptance before promotion

Implement owners separately after boundary approval; independently test eager
and replacement values, nonuniform parameter/input VJPs, SGD/Adam trajectories,
optimizer state, higher derivatives where supported, all lifecycle callbacks,
hooks, training/eval/grad modes, groups and checkpoint/restart. Keep failures and
unsupported cases. Graph-break count, captured regions, derivative work,
effective route and recompile reasons must be truthful.

Reserve a held-out shape/batch/depth matrix before performance implementation.
Run the existing generated matrix and matched training protocol, then compare
the strongest available native/compiled Torch and external whole-JAX training
routes at matching dtype, batching and task quality. Report imports/setup,
first compilation, steady-state, total time to quality, peak RSS and invalidation
cost separately. Retain repeated fresh-process samples and every loss/timeout.
Do not weaken #918's proposed budgets or close #936/#918 based on this fake, a
private quantum kernel, SGD alone or a single favorable circuit.

## Reserved evaluation cases

These cases are reserved before performance implementation, separate from the
existing widths 2/4/6, depths 1/3 development matrix. Use data/circuit seed
20261102 and retain all cases, including unsupported routes and timeouts:

| Width | Depth | Declared batch | Entangling layout |
| ---: | ---: | ---: | --- |
| 3 | 2 | 1 | chain |
| 3 | 4 | 8 | ring |
| 5 | 2 | 8 | ring |
| 5 | 4 | 64 | chain |
| 7 | 2 | 64 | ring |
| 7 | 4 | 1 | chain |

Use exact complex128/float64, changing data, shared/reused nonlinear parameters,
nonuniform readout weights, both SGD and Adam, 60 updates and a checkpoint at
update 30. Preserve the existing matched classifier/Heisenberg protocol as a
separate comparison. Run three serial fresh processes per applicable route and
case; use a 120-second per-process deadline and a declared 2 GiB RSS budget.
Report unsupported precision/operations, missing dependencies, failed values or
recovery, memory refusal and timeout separately. Initial development measurements
must not be used to remove an unfavorable held-out case or change these budgets.
Include the existing no-grad/parameter/input/both derivative conformance even
where a training optimizer is inapplicable. This reservation is not a claim that
all six cases or any route already pass.

## Separately approved checkpoint version extension

On 2026-11-03 the user separately approved the narrow v2 checkpoint identity
change in `TRAINING_CHECKPOINT_IDENTITY_CONTRACT.md`. That contract supersedes
this document's checkpoint-schema preservation only for its qualified v2 writer
and strict versioned v1/v2 reader. Its contract and fake must land before the
Compiler/Runtime implementation. Other public APIs, Torch defaults and optimizer
arithmetic remain unchanged.
