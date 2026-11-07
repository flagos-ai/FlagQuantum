# portfolio optimization with hhl

Category: applications / finance.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/finance/hybrid_hhl_for_portfolio_optimization/portfolio_optimization_with_hhl.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/finance/hybrid_hhl_for_portfolio_optimization/portfolio_optimization_with_hhl.qmod'
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

Recorded resources: 10 physical qubits, 429 operations, depth 402.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_hhl.py](../../../../../../../tests/integration/test_synthesis_qmod_hhl.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `89df9698331444d650e0f7325d2fcec5ec716913`.

Original SHA-256: `f18bb82be58a87a92471ceaa84a006d4bd93d7cc83d33c1e3474079853259d20`.

Approved SHA-256: `f18bb82be58a87a92471ceaa84a006d4bd93d7cc83d33c1e3474079853259d20`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
