# rainbow options direct method

Category: applications / finance.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/finance/rainbow_options/rainbow_options_direct_method.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/finance/rainbow_options/rainbow_options_direct_method.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "k": 1
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 14 physical qubits, 19355 operations, depth 15770.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_rainbow_fixed_payoff.py](../../../../../../../tests/integration/test_synthesis_qmod_rainbow_fixed_payoff.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `56d904762e34733451e780bbd458b90f17625ec6`.

Original SHA-256: `6a911a20ead2fd62de750ffc3d4c4638dd0ef07d961d008a45edbc6ce00ee913`.

Approved SHA-256: `6a911a20ead2fd62de750ffc3d4c4638dd0ef07d961d008a45edbc6ce00ee913`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
