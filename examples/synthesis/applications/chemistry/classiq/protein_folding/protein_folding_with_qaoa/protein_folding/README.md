# protein folding

Category: applications / chemistry.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/chemistry/protein_folding/protein_folding_with_qaoa/protein_folding.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/chemistry/protein_folding/protein_folding_with_qaoa/protein_folding.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "params": [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.1
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 7 physical qubits, 22312 operations, depth 18286.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_records.py](../../../../../../../../tests/integration/test_synthesis_qmod_records.py)
- [validate_qmod_records.py](../../../../../../../../tools/validate_qmod_records.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `f859094258efa802ac945e8376946f5e5c68811e`.

Original SHA-256: `d4ae453aa1650a2ec9dbe2f0f0923f7316a5a733a84ec50d60360b76346cfb2b`.

Approved SHA-256: `d4ae453aa1650a2ec9dbe2f0f0923f7316a5a733a84ec50d60360b76346cfb2b`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
