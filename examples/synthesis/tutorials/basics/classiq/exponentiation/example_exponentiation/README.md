# example exponentiation

Category: tutorials / basics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/basic_tutorials/exponentiation/example_exponentiation.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/basic_tutorials/exponentiation/example_exponentiation.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
null
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 12 physical qubits, 7163 operations, depth 5510.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_large_pauli.py](../../../../../../../tests/integration/test_synthesis_qmod_large_pauli.py)
- [validate_qmod_large_pauli.py](../../../../../../../tools/validate_qmod_large_pauli.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `f66a7f71c4e001747b3f7c4144c9c850ca8e8807`.

Original SHA-256: `70068a39d29b37ec30234de3ee46ed1f62353e6727c0b6a8c36a7ed1d5ef16df`.

Approved SHA-256: `70068a39d29b37ec30234de3ee46ed1f62353e6727c0b6a8c36a7ed1d5ef16df`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
