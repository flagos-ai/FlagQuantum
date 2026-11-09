# qft

Category: tutorials / library.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/qft/qft.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/qft/qft.qmod'
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

Recorded resources: 4 physical qubits, 12 operations, depth 8.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_qft.py](../../../../../../../../../tests/integration/test_synthesis_qmod_qft.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `fc13d0dd2cd71754d6bf7e63b4a654753b4e33a9`.

Original SHA-256: `e846e39be758089825defcb4af4757206edf1c715cb8aab7f2d24318ff9f467e`.

Approved SHA-256: `e846e39be758089825defcb4af4757206edf1c715cb8aab7f2d24318ff9f467e`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
