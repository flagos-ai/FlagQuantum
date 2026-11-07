# entanglement

Category: tutorials / basics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/basic_tutorials/entanglement/entanglement.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/basic_tutorials/entanglement/entanglement.qmod'
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

Recorded resources: 6 physical qubits, 6 operations, depth 6.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_static_repeat.py](../../../../../../../tests/integration/test_synthesis_qmod_static_repeat.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `05d9776aacf23ebac13fea6b805240618cd00146`.

Original SHA-256: `76216eed7b9c4bd2b6a7756de4431078ed1cb96a0574441a054599e7b0e0281b`.

Approved SHA-256: `76216eed7b9c4bd2b6a7756de4431078ed1cb96a0574441a054599e7b0e0281b`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
