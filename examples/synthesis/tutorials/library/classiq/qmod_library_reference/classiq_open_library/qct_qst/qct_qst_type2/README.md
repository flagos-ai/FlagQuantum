# qct qst type2

Category: tutorials / library.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/qct_qst/qct_qst_type2.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/qct_qst/qct_qst_type2.qmod'
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

Recorded resources: 4 physical qubits, 2098 operations, depth 1802.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_type2_transforms.py](../../../../../../../../../tests/integration/test_synthesis_qmod_type2_transforms.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `73f5156921cef9a6ed427ac13c910d0d23312ad8`.

Original SHA-256: `b0820ecc1a9ebedb8c3409dbb677bbfd1bb28cc7ff4de092be6b28ee31c2755e`.

Approved SHA-256: `b0820ecc1a9ebedb8c3409dbb677bbfd1bb28cc7ff4de092be6b28ee31c2755e`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
