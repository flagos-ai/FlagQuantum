# Approved versioned checkpoint workload identity for #936

Status: explicitly approved by the user on 2026-11-03. Land this contract and
its replacement fake before implementing Compiler/Runtime changes. This separate
approval permits the qualified v2 checkpoint writer and strict versioned reader
below; it supersedes the earlier checkpoint-preservation boundary only for this
specified versioned change. Torch backend/defaults and Adam arithmetic stay unchanged.

## Reproduced failure

`reproduce_frozen_checkpoint.py` reproduces the same CPU refusal at both the
pre-implementation producer `97b315ea02863583e88d13a4127be65133485cf6` and current
source. Three Adam updates, frozen changing inputs and an angle
`parameters[0].detach() * inputs[:, 1]` produce a checkpoint that a fresh Module
cannot restore after ordinary representative-input priming. With trainable
inputs the same workload restores. GPU CI run 41482999465 exposes the same
failure for both SGD and Adam, after their value/gradient/update checks pass.

The existing v1 signature replaces only requires_grad=True angle tensors with
shape/dtype markers. It treats a dynamic binding whose derivative is deliberately
cut as a fixed numerical constant. Its pre-update value differs from the fresh
Module's initial value. This is not a new compiled-Adam error: CUDA remains eager.

## Approved protected change

For Modules with a certifiable canonical captured builder, new checkpoints emit
`version="flagquantum.training_state.v2"`. Keep all existing envelope fields and
all public function signatures. Interpret existing `workload_signature` under the
v2 version as a hash of the canonical symbolic Core IR plus the existing binding
computation and its static constants, not current numeric parameter/input values
or requires_grad masks. The existing `ir_hash` remains the exact consumed
pre-update numeric IR hash; it must never be replaced by the workload signature.

V1 remains readable using its original strict workload validation, including its
existing refusal where a dynamic detached angle was encoded as a constant.
Never silently reinterpret a v1 hash as v2, relax constant checks, mutate a failed
restore target, or disguise a failed old archive as migrated. Uncertifiable
builders retain the existing v1 writer and explicit limitation. Version selection
is internal and visible in the existing serialized `version` field; no new user
flag, public field, backend default or dependency is proposed. Update only the
checkpoint-version portion of the protected contract after approval.

## Ownership and stable identity

- Compiler owns canonicalization of its existing CapturedModuleBuilder binding
  graph: parameter sharing, input indexing, detach/gradient cuts, operator names,
  literal constants and lifted constant snapshots. Reuse Torch FX and Core IR;
  do not create another quantum IR or registry. Normalize existing graph nodes
  deterministically rather than depending on a Python repr or object identity.
- Only Module's already approved capture consumer reads that artifact and exposes
  a private immutable workload digest to Runtime checkpoint orchestration. No new
  Runtime-to-Compiler import, public contract type or Simulation dependency.
- Runtime versions the envelope and checks its signature before mutation. Existing
  optimizer contract, RNG, precision, topology, runtime policy, tensor identities,
  step count and state restoration remain authoritative. Unknown versions fail.
- Simulation remains unchanged and consumes Core IR. No automatic JAX/backend
  selection or optimizer arithmetic changes are authorized by this proposal.

A new immutable captured constant or changed quantum topology/binding operation
must reject restore. Changing a dynamic frozen input, parameter value or grad mask
must not change v2 workload identity. Checkpoint restore still requires existing
representative-input priming where the builder needs inputs. Cross-version Torch
operator encoding must be versioned or explicitly refused; the fake does not
certify compatibility between Torch releases.

## Contract-first delivery and acceptance

First land this contract, narrow protected version delta, replacement
fake and acceptance tests in an integration PR. Implement Compiler digesting and
Runtime v1/v2 handling in separate owned commits after that baseline lands. Keep
the current failing GPU tests intact; rerun them and existing CPU/GPU/checkpoint/
error/API/architecture CI on the real implementation before merging it.

`tests/unit/test_checkpoint_identity_contract_fake.py` is a review-only replacement: its v2 digest
uses canonical Core IR, the current FX code and lifted buffers to demonstrate
identity invariants and deterministic six-step recovery. Production must use the
normalized node encoding above, not this same-environment FX-code shortcut.
The fake does not implement the production loader, RNG, distributed restoration,
cross-Torch-version migration or claim performance/closure. It must also prove
legacy v1 still refuses the original case and still restores a supported case,
and that changed binding gradients, topology and static constants remain refused.

Neither #918 nor #936 may close based on this fake or v2 checkpoint support alone.
Cold compilation/default adoption and the other performance acceptance remain.

## Runnable contract check

```bash
PYTHONPATH=. python -m pytest tests/unit/test_checkpoint_identity_contract_fake.py -q
```

The ten fake checks exercise the same existing public `fq.Module`, `fq.train`,
PyTorch optimizers and `save_checkpoint`/`load_checkpoint` consumers. They do not
change the production writer or authorize issue closure.
