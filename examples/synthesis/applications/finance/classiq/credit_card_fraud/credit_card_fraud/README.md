# credit card fraud

Category: applications / finance.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/finance/credit_card_fraud/credit_card_fraud.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/finance/credit_card_fraud/credit_card_fraud.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "data1": [
    0.17,
    -0.43,
    0.29
  ],
  "data2": [
    -0.31,
    0.57,
    0.11
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 3 physical qubits, 57 operations, depth 41.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_integer_composition.py](../../../../../../../tests/integration/test_synthesis_qmod_integer_composition.py)
- [validate_qmod_integer_composition.py](../../../../../../../tools/validate_qmod_integer_composition.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `9d6e6a52f2d301849370a2bcba708c24f249813e`.

Original SHA-256: `e51f68346a2580027b751bda6aacd9cf8c1e610ef828c8a6260203c8bd88e140`.

Approved SHA-256: `e51f68346a2580027b751bda6aacd9cf8c1e610ef828c8a6260203c8bd88e140`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
