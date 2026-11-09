# prepare state

Category: tutorials / basics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/basic_tutorials/prepare_state/prepare_state.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/basic_tutorials/prepare_state/prepare_state.qmod'
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

Recorded resources: 5 physical qubits, 61 operations, depth 57.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_preparation_constants.py](../../../../../../../tests/integration/test_synthesis_qmod_preparation_constants.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `4a84e2058fa7c5b2b90be7ae3019853204b601f0`.

Original SHA-256: `f43e50a82c193cc79b4c20ca05342c9be47dedf63d5adc8177faf82c8119d0d1`.

Approved SHA-256: `f43e50a82c193cc79b4c20ca05342c9be47dedf63d5adc8177faf82c8119d0d1`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
