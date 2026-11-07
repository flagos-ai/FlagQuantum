# qpe for matrix

Category: algorithms / phase_estimation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/quantum_phase_estimation/qpe_for_matrix/qpe_for_matrix.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/quantum_phase_estimation/qpe_for_matrix/qpe_for_matrix.qmod'
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

Recorded resources: 9 physical qubits, 347 operations, depth 322.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_static_qpe.py](../../../../../../../tests/integration/test_synthesis_qmod_static_qpe.py)
- [validate_qmod_static_qpe.py](../../../../../../../tools/validate_qmod_static_qpe.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `7f9e59ba7dca3260d8749cf15b43adc5b2bd2afd`.

Original SHA-256: `a782a5e6fc3cf6b7f9d7bd2ed32645da1e3b80185172081e3bfcb578f1df1433`.

Approved SHA-256: `a782a5e6fc3cf6b7f9d7bd2ed32645da1e3b80185172081e3bfcb578f1df1433`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
