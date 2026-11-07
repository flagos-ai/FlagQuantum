# qct type2

Category: tutorials / library.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/qct_qst/qct_type2.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/qct_qst/qct_type2.qmod'
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

Recorded resources: 3 physical qubits, 370 operations, depth 327.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_type2_transforms.py](../../../../../../../../../tests/integration/test_synthesis_qmod_type2_transforms.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `12b820aa8fe19a039f18b825b1016892406d7dab`.

Original SHA-256: `caf6c3ac4d4ed5eaffac84538dfe8bdbecabd096fa6c5d31db02c24aed97f8ce`.

Approved SHA-256: `caf6c3ac4d4ed5eaffac84538dfe8bdbecabd096fa6c5d31db02c24aed97f8ce`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
