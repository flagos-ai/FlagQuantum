# simon

Category: algorithms / foundational.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/foundational/simon/simon.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/foundational/simon/simon.qmod'
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

Recorded resources: 10 physical qubits, 15 operations, depth 4.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_simon.py](../../../../../../../tests/integration/test_synthesis_qmod_simon.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `8cc01f333b1f570fe76ac1a05f4be4bb9129670b`.

Original SHA-256: `0c6b69aefd6eb09ef120f4b625abe23d754a98014b338d22c53a5d7ce96dcd70`.

Approved SHA-256: `0c6b69aefd6eb09ef120f4b625abe23d754a98014b338d22c53a5d7ce96dcd70`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
