# HHL portfolio

Category: applications / finance.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/finance/portfolio_optimization_hhl/HHL_portfolio.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/finance/portfolio_optimization_hhl/HHL_portfolio.qmod'
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

Recorded resources: 14 physical qubits, 13076 operations, depth 11638.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_hhl.py](../../../../../../../tests/integration/test_synthesis_qmod_hhl.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `9e851bb1afa10df4246cacdcc39dbff331d0fa1b`.

Original SHA-256: `8d099f5de2543cf54ef5af29c9bfc3d1857b854b3c490e0172a63afbfe8793d1`.

Approved SHA-256: `8d099f5de2543cf54ef5af29c9bfc3d1857b854b3c490e0172a63afbfe8793d1`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
