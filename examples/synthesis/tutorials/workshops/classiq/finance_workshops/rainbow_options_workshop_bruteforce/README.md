# rainbow options workshop bruteforce

Category: tutorials / workshops.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/workshops/finance_workshops/rainbow_options_workshop_bruteforce.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/workshops/finance_workshops/rainbow_options_workshop_bruteforce.qmod'
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

Recorded resources: 5 physical qubits, 354 operations, depth 306.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_rainbow_bruteforce.py](../../../../../../../tests/integration/test_synthesis_qmod_rainbow_bruteforce.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `4842be3ace906a5ff3bb19367e1fbb100cb1ccf5`.

Original SHA-256: `8175fdf84ed75b7f84862bb800de303b03e8d2f22227b2052ed838cb0a76cf75`.

Approved SHA-256: `8175fdf84ed75b7f84862bb800de303b03e8d2f22227b2052ed838cb0a76cf75`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
