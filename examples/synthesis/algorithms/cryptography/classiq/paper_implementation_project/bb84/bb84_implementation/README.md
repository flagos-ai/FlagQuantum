# bb84 implementation

Category: algorithms / cryptography.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/community/paper_implementation_project/bb84/bb84_implementation.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'community/paper_implementation_project/bb84/bb84_implementation.qmod'
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

Recorded resources: 8 physical qubits, 6 operations, depth 1.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_integer_composition.py](../../../../../../../../tests/integration/test_synthesis_qmod_integer_composition.py)
- [validate_qmod_integer_composition.py](../../../../../../../../tools/validate_qmod_integer_composition.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `5e60e9ebf3efb8c4e85b9ac0a6f17d5593e251ce`.

Original SHA-256: `0fe7bec249141df2b8ae9b2f5caa60130c6ce08242920d73b5673905fac9c80e`.

Approved SHA-256: `0fe7bec249141df2b8ae9b2f5caa60130c6ce08242920d73b5673905fac9c80e`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
