# encode in angle

Category: tutorials / state_preparation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/variational_data_encoding/encode_in_angle.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/variational_data_encoding/encode_in_angle.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "data": [
    0.2,
    -0.3,
    0.4,
    0.5
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 4 physical qubits, 4 operations, depth 1.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_real_arrays.py](../../../../../../../../../tests/integration/test_synthesis_qmod_real_arrays.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `3510209fabde4432f3fc39c20d6f481cd52112a6`.

Original SHA-256: `334ba1c4cee9f776171961aae9c681b5cc11380abb3800621268342ed105913e`.

Approved SHA-256: `334ba1c4cee9f776171961aae9c681b5cc11380abb3800621268342ed105913e`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
