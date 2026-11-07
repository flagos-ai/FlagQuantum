# synthesis tutorial

Category: tutorials / basics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/basic_tutorials/the_classiq_tutorial/synthesis_tutorial.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/basic_tutorials/the_classiq_tutorial/synthesis_tutorial.qmod'
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

Recorded resources: 10 physical qubits, 16 operations, depth 12.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_originals.py](../../../../../../../tests/integration/test_synthesis_qmod_originals.py)
- [test_qmod_comparisons.py](../../../../../../../tests/team/ecosystem/test_qmod_comparisons.py)
- [test_qmod_parser.py](../../../../../../../tests/team/ecosystem/test_qmod_parser.py)
- [test_qmod_tutorial_sources.py](../../../../../../../tests/team/ecosystem/test_qmod_tutorial_sources.py)

Circuit-based validation source links: 0.

## Provenance

Original Git blob: `7e9df09937d8360be71688762c4d22f309faa443`.

Original SHA-256: `4dbff47c68a449c918d30cdb63d8398de7084f02617e604d82c24191bcbfea8b`.

Approved SHA-256: `4dbff47c68a449c918d30cdb63d8398de7084f02617e604d82c24191bcbfea8b`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
