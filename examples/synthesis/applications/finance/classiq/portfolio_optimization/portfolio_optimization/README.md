# portfolio optimization

Category: applications / finance.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/finance/portfolio_optimization/portfolio_optimization.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/finance/portfolio_optimization/portfolio_optimization.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "params": [
    0.003,
    -0.006,
    0.009000000000000001,
    -0.012,
    0.015,
    -0.018000000000000002
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 12 physical qubits, 271 operations, depth 51.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_records.py](../../../../../../../tests/integration/test_synthesis_qmod_records.py)
- [validate_qmod_records.py](../../../../../../../tools/validate_qmod_records.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `136d3dc2b6f964dfcb2344b094de728262c2e3d9`.

Original SHA-256: `481f4455d4045cd5c3f6f3edc7f866bb29a67c18b2d31e90968c0768fc4167a5`.

Approved SHA-256: `481f4455d4045cd5c3f6f3edc7f866bb29a67c18b2d31e90968c0768fc4167a5`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
