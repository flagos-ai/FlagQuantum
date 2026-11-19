# Checkpoint parameter training state

Status: **approved by the API owner on 2026-11-12**. The human user approved
this exact v3 contract and replacement fake. See the
[approval record](../api-changes/FQ-TRAINING-PARAMETER-STATE-20261112.md).
Owner: Integration/API; implementation owner: Runtime. Path:
`single_device_fast_path`. Existing public signatures and default initialization
remain unchanged.

## Problem and proposed behavior

Current v1/v2 checkpoints do not record registered parameters' `requires_grad`.
A fresh callable Module can load a frozen source's v2 file with the parameter
trainable again; a symbolic Circuit can instead refuse under v1 workload
identity. Optimizer state is associated by PyTorch's parameter order, so two
equally sized groups can restore to the wrong named weights if reordered.

Qualified new files use `version="flagquantum.training_state.v3"`. Preserve all
existing envelope fields and add exactly:

```json
{
  "parameter_training": {
    "requires_grad": {
      "named_parameter_groups.encoder": false,
      "named_parameter_groups.readout": true
    },
    "optimizer_parameters": [[
      "named_parameter_groups.encoder",
      "named_parameter_groups.readout"
    ]]
  }
}
```

`requires_grad` exactly covers `module.named_parameters()` with strict boolean
values. `optimizer_parameters` is null when no optimizer was saved, otherwise
one ordered name list per optimizer group. Names refer to parameters registered
on the checkpointed module, including classical parameters when the existing
single-quantum-owner composite workflow is used. Duplicate optimizer references,
unregistered optimizer parameters and multiply registered aliases are refused
for v3 in this slice. Reusing one parameter in multiple gates remains supported.
Values, shapes and dtypes remain in the existing module state; there is no new
parameter-specification class, optimizer, registry or root export.

An optimizer can contain a frozen parameter, or intentionally omit a trainable
parameter. The two fields preserve both choices without equating them. Restore
requires the caller's optimizer groups to have the same names and order. It
does not rebuild groups, add parameters, select a different optimizer, or reset
moments. Equal group sizes alone are insufficient.

## Versioned workload identity

Keep v1 and v2 signature algorithms and reader behavior unchanged. V3 uses the
existing `workload_signature` field for a domain-separated canonical digest:

- Captured callable: the existing certified v2 capture digest, wrapped with
  `kind="captured_builder", encoding=2`. Preserve the existing normalized FX
  identity, Torch-version guard, static constants and detach semantics.
- Ordinary symbolic Circuit: its canonical source Core IR, wrapped with
  `kind="symbolic_circuit", encoding=1`. Shapes, dtype, names, expression
  structure, gate order, wires and fixed values belong to the signature; bound
  parameter values and their differentiation flags do not.

V3 Circuit qualification in the first implementation requires the ordinary
Circuit type, named symbolic parameters, default construction initial state,
and no Tensor literals in its source IR. The canonical Core codec remains
authoritative. Literal tensors, custom Circuit types and uncertifiable builders
do not gain a new identity guarantee. Existing unqualified all-trainable saves
retain their existing version and limitations. Unqualified sources containing
frozen owned parameters refuse save rather than silently omitting the new
training-state guarantee. Eligibility and refusal must be documented explicitly.

Module's existing private checkpoint-signature boundary is extended for v3;
the default v2 calculation remains unchanged for old readers. No new Compiler
import or Simulation change is admitted. Reject stale captures/templates before
writing. The existing consumed numeric `ir_hash` remains diagnostic evidence and
must not be substituted for a symbolic workload digest.

## Restore transaction

Before mutation, validate version, exact new-field structure, complete parameter
names, strict booleans, optimizer presence/type/bindings, state tensor layout,
workload, precision and rank topology using existing guards. Missing, extra,
duplicated or malformed names fail with TrainingStateError. Shape/dtype changes
and unknown versions fail closed. No implicit gradient-policy migration occurs.

On successful v3 restore, use existing state loading and apply saved
`requires_grad` flags in place, preserving registered Parameter identities and
optimizer ownership. Clear registered parameter `.grad` values to None so stale
target gradients cannot update newly frozen parameters. Checkpoints describe
complete optimizer-step boundaries, not partially accumulated gradients. Keep
existing precision, policy, RNG, seed, step and optimizer moment restoration.
Loading occurs between steps; existing live autograd graphs are not serialized
or made valid across in-place weight restoration.

On failure, restore all original state, flags, gradients and identities, together
with the existing optimizer/precision/RNG rollback. Unsupported target hooks or
custom Parameter storage must fail without leaving a partially restored model.
Custom optimizers and hooks require the existing refusal/rollback guarantees;
the review fake does not certify arbitrary third-party mutation.

## Compatibility and migration

- V1/v2 readers keep their historical validation and caller-owned grad flags.
  Old files do not contain enough information to infer a freeze policy. Users
  reapply it explicitly, as in the current executable guide.
- The new reader accepts v1/v2/v3. Old readers reject v3 as an unknown version.
- Save a new v3 file after explicit old-file recovery and representative
  execution to migrate; never relabel old bytes or reinterpret an old digest.
- Function signatures, Module state_dict layout, TrainingCheckpointRestore
  schema, default initialization, initializer copying, backend defaults and
  PyTorch optimizer arithmetic stay unchanged. No forced optimizer membership
  or automatic trainability inheritance from initializer tensors is added.
- This proposal does not authorize accelerator/distributed promotion, shared
  checkpoint aliases, elementwise freezing, gradient accumulation recovery or
  a user-facing freeze/unfreeze framework.

V3 retains the existing quantum-owner workload boundary. It does not fingerprint
arbitrary classical host code in a composite model.

## Delivery and acceptance

The approved contract and replacement fake precede the production implementation
in the review series. Qualified v3 save/load is implemented behind the existing
public interfaces; the [training guide](../guides/TRAINING_PARAMETERS.md) records
the supported profile and historical migration. Local CPU conformance covers
statevector, MPS and tensor-network trajectories, optimizer name/order refusal,
parameter flags/identities, stale gradients, rollback and v1/v2 compatibility.
This is not a claim that the full repository regression tier is green.

The original acceptance requirements are retained below.

Land the proposed contract and replacement fake before Runtime implementation.
`tests/unit/test_training_parameter_state_contract_fake.py` demonstrates only the
new metadata, identity and transactional parameter/optimizer behavior, using the
same model/optimizer training consumer with two review codecs. Existing RNG,
precision, topology, atomic file publication, cross-device/Torch compatibility,
stale-program refusal and production admission remain implementation acceptance
work; the fake must not be cited as proof of those guarantees.

After explicit approval, implement behind the current saver/loader, preserve
historical fixtures, and add public save/load conformance against the fake.
Run smoke/unit and local integration tiers, plus applicable checkpoint, error,
API, architecture and type checks. Test both frontends, flat/named parameters,
trainable/frozen/omitted optimizer membership, partial optimizer groups,
same-count reorder refusal, all-frozen inference, mid-training unfreezing,
optimizer moments, malformed payloads and injected rollback
failures. No release or performance claim follows from this proposal.

The public workflow and contract checks are executable now:

```bash
python -m examples.local.train_parameters
python -m pytest tests/integration/test_training_parameter_lifecycle.py -q
python -m pytest tests/unit/test_training_parameter_state_contract_fake.py -q
```
