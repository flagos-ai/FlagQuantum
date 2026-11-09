# Qmod tutorial part2

Category: tutorials / basics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/basic_tutorials/the_classiq_tutorial/Qmod_tutorial_part2.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/basic_tutorials/the_classiq_tutorial/Qmod_tutorial_part2.qmod'
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

Recorded resources: 3 physical qubits, 20 operations, depth 16.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_indexed_control.py](../../../../../../../tests/integration/test_synthesis_qmod_indexed_control.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `f2489a9e1dd6ab7459a0401a5c441a751849d4ec`.

Original SHA-256: `2330d29aca4c78675ca164737dca041ea4d4b2b0903ad6f1dc4a64ceca02f844`.

Approved SHA-256: `2330d29aca4c78675ca164737dca041ea4d4b2b0903ad6f1dc4a64ceca02f844`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
