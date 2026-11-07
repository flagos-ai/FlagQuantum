# oracles workshop

Category: tutorials / workshops.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/workshops/oracle_workshop/oracles_workshop.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/workshops/oracle_workshop/oracles_workshop.qmod'
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

Recorded resources: 5 physical qubits, 132 operations, depth 107.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_predicate_oracle.py](../../../../../../../tests/integration/test_synthesis_qmod_predicate_oracle.py)
- [validate_qmod_predicate_oracle.py](../../../../../../../tools/validate_qmod_predicate_oracle.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `71c4c14ccee1fba9e4d36fc8cda2e83e0770e6a6`.

Original SHA-256: `b998fb0d44784390446d1c1f79c157c7007c831762709870f32e4cb0b8aa8322`.

Approved SHA-256: `b998fb0d44784390446d1c1f79c157c7007c831762709870f32e4cb0b8aa8322`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
