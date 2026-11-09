# quantum autocallable option pricing

Category: applications / finance.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/finance/autocallable_options/quantum_autocallable_option_pricing.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/finance/autocallable_options/quantum_autocallable_option_pricing.qmod'
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

Recorded resources: 11 physical qubits, 4093 operations, depth 4083.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_autocallable_fixed_preparation.py](../../../../../../../tests/integration/test_synthesis_qmod_autocallable_fixed_preparation.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `581018f8b58c3895417701aba3d6ec0e5b9be65b`.

Original SHA-256: `eeb66f5e3446a3498ece2269b97599cc6d0ef2d8895265276ecf605872bce386`.

Approved SHA-256: `eeb66f5e3446a3498ece2269b97599cc6d0ef2d8895265276ecf605872bce386`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
