# 3sat oracles

Category: tutorials / techniques.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/technology_demonstrations/oracle_generation/3sat_oracles.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/technology_demonstrations/oracle_generation/3sat_oracles.qmod'
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

Recorded resources: 150 physical qubits, 1033 operations, depth 629.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_wide_cnf.py](../../../../../../../tests/integration/test_synthesis_qmod_wide_cnf.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `60c11fa2050f259c1c13dbb2ee99daeba170f613`.

Original SHA-256: `9cfcd4a4caedab6f571df30893d52ea3bdf299a5438059726a3e82330f7438d2`.

Approved SHA-256: `9cfcd4a4caedab6f571df30893d52ea3bdf299a5438059726a3e82330f7438d2`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
