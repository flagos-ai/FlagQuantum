# qst type2

Category: tutorials / library.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/qct_qst/qst_type2.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/qct_qst/qst_type2.qmod'
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

Original Git blob: `2541d8808bca009faf0ccb62bb985c5db604f240`.

Original SHA-256: `0257c839e1b6c5b770f1d655ae8887cad2457094cea9be02c47a24b96301a0df`.

Approved SHA-256: `0257c839e1b6c5b770f1d655ae8887cad2457094cea9be02c47a24b96301a0df`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
