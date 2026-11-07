# option pricing

Category: applications / finance.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/finance/option_pricing/option_pricing.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/finance/option_pricing/option_pricing.qmod'
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

Recorded resources: 6 physical qubits, 1724 operations, depth 1497.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_option_pricing.py](../../../../../../../tests/integration/test_synthesis_qmod_option_pricing.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `737eaa2b14b54bc8163959ebd5cfa17e592b8fc8`.

Original SHA-256: `613e41b10a31835383e46d673108dd4dcfba79b707ae9ce4b6f1063bc1b52ffc`.

Approved SHA-256: `613e41b10a31835383e46d673108dd4dcfba79b707ae9ce4b6f1063bc1b52ffc`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
