# adapt vqe

Category: algorithms / state_preparation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/quantum_state_preparation/adapt_vqe/adapt_vqe.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/quantum_state_preparation/adapt_vqe/adapt_vqe.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "params": [
    0.23
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 4 physical qubits, 7 operations, depth 5.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_operation_helpers.py](../../../../../../../tests/integration/test_synthesis_qmod_operation_helpers.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `3bd4550d1ddcca756681abcc76758cd8b997a662`.

Original SHA-256: `0c68ef1bd549575de7caa9eae9ea1b541c416e30b2bca7f1492f80ed1d941801`.

Approved SHA-256: `0c68ef1bd549575de7caa9eae9ea1b541c416e30b2bca7f1492f80ed1d941801`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
