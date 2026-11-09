# qpe

Category: tutorials / library.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/qpe/qpe.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/qpe/qpe.qmod'
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

Recorded resources: 4 physical qubits, 14 operations, depth 9.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_qpe.py](../../../../../../../../../tests/integration/test_synthesis_qmod_qpe.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `d93c2d144b8c6d2f67abe7de4809b566b9ba6a88`.

Original SHA-256: `5fe81fd2206951fa368f9e190b6d274202866d7991a750468c8072aad83d3700`.

Approved SHA-256: `5fe81fd2206951fa368f9e190b6d274202866d7991a750468c8072aad83d3700`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
