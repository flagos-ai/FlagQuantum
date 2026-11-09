# quantum counting qpe

Category: algorithms / amplitude_estimation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/amplitude_amplification_and_estimation/quantum_counting/quantum_counting_qpe.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/amplitude_amplification_and_estimation/quantum_counting/quantum_counting_qpe.qmod'
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

Recorded resources: 9 physical qubits, 10453 operations, depth 8944.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_static_qpe.py](../../../../../../../tests/integration/test_synthesis_qmod_static_qpe.py)
- [validate_qmod_static_qpe.py](../../../../../../../tools/validate_qmod_static_qpe.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `8ef70fc31c9aec338fd0909b6cc00102033706df`.

Original SHA-256: `5b6999f9c73fdcca7bdcffbc9226f94c5536a540520d2420aa09a674646c5818`.

Approved SHA-256: `5b6999f9c73fdcca7bdcffbc9226f94c5536a540520d2420aa09a674646c5818`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
