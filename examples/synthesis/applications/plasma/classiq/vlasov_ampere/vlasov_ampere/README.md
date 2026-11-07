# vlasov ampere

Category: applications / plasma.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/plasma/vlasov_ampere/vlasov_ampere.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/plasma/vlasov_ampere/vlasov_ampere.qmod'
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

Recorded resources: 24 physical qubits, 4467 operations, depth 3359.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_vlasov.py](../../../../../../../tests/integration/test_synthesis_qmod_vlasov.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `058feaae75cf1f4fb760d1cbc38cf09b8585b61b`.

Original SHA-256: `260ce15bfe818903c34c7ae3a6010ceca9784ae8e41b893a50c695a7d28b9a8f`.

Approved SHA-256: `260ce15bfe818903c34c7ae3a6010ceca9784ae8e41b893a50c695a7d28b9a8f`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
