# qsvt

Category: tutorials / library.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/qsvt/qsvt.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/qsvt/qsvt.qmod'
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

Recorded resources: 13 physical qubits, 2243 operations, depth 2053.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_expanded_qsvt.py](../../../../../../../../../tests/integration/test_synthesis_qmod_expanded_qsvt.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `54330885eb0e003b1facf5a64b040078f469cf7e`.

Original SHA-256: `8d859d86ecf3d0c3c68d1ddc9707e1f382a304439cdc27ae51689581783d6457`.

Approved SHA-256: `8d859d86ecf3d0c3c68d1ddc9707e1f382a304439cdc27ae51689581783d6457`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
