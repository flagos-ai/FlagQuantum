# value at risk

Category: applications / finance.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/finance/value_at_risk/value_at_risk.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/finance/value_at_risk/value_at_risk.qmod'
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

Recorded resources: 8 physical qubits, 3230 operations, depth 2666.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_value_at_risk.py](../../../../../../../tests/integration/test_synthesis_qmod_value_at_risk.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `665f612912fff9525dfc276e39daf4b47a02cdf8`.

Original SHA-256: `b0bc0a0672576d2c13c771bcba7b37fd2f67366c59cd86010fa5b0ce0a230071`.

Approved SHA-256: `b0bc0a0672576d2c13c771bcba7b37fd2f67366c59cd86010fa5b0ce0a230071`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
