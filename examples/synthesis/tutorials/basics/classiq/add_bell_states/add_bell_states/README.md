# add bell states

Category: tutorials / basics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/basic_tutorials/add_bell_states/add_bell_states.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/basic_tutorials/add_bell_states/add_bell_states.qmod'
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

Recorded resources: 8 physical qubits, 22 operations, depth 13.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_prepare_state.py](../../../../../../../tests/integration/test_synthesis_qmod_prepare_state.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `c5a51feb0f842b76bd2171ed62939bbd1577ffab`.

Original SHA-256: `c03923daf2a5b58ea1a5c70f7834fef661418b547b60c9d1b7572c11be99acb2`.

Approved SHA-256: `c03923daf2a5b58ea1a5c70f7834fef661418b547b60c9d1b7572c11be99acb2`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
