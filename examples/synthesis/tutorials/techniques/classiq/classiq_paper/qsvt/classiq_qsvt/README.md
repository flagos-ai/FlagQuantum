# classiq qsvt

Category: tutorials / techniques.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/technology_demonstrations/classiq_paper/qsvt/classiq_qsvt.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/technology_demonstrations/classiq_paper/qsvt/classiq_qsvt.qmod'
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

Recorded resources: 14 physical qubits, 2798 operations, depth 2270.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_qsvt.py](../../../../../../../../tests/integration/test_synthesis_qmod_qsvt.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `2045c54435394681506fb43c0977d0deb0fca2f3`.

Original SHA-256: `db9e50097a11725bcc6a6cb4ffcaaf0ab0dcc62739d6c759721aa169303afe8e`.

Approved SHA-256: `db9e50097a11725bcc6a6cb4ffcaaf0ab0dcc62739d6c759721aa169303afe8e`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
