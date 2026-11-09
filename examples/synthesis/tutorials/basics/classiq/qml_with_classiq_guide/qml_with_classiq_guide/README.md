# qml with classiq guide

Category: tutorials / basics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/basic_tutorials/qml_with_classiq_guide/qml_with_classiq_guide.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/basic_tutorials/qml_with_classiq_guide/qml_with_classiq_guide.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "input_0": 0.37,
  "weight_0": -0.81
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 1 physical qubits, 1 operations, depth 1.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_classical_real.py](../../../../../../../tests/integration/test_synthesis_qmod_classical_real.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `d0879a1decbdc9d1ef8a9045d24901b97dd090f7`.

Original SHA-256: `04191579fc322a5786d78d28175b51026da6ce12228ad7e1a1bb0d89127b466e`.

Approved SHA-256: `04191579fc322a5786d78d28175b51026da6ce12228ad7e1a1bb0d89127b466e`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
