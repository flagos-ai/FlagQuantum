# quantum counting iqae

Category: algorithms / amplitude_estimation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/amplitude_amplification_and_estimation/quantum_counting/quantum_counting_iqae.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/amplitude_amplification_and_estimation/quantum_counting/quantum_counting_iqae.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "k": 1
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 5 physical qubits, 781 operations, depth 665.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_amplification_iterations.py](../../../../../../../tests/integration/test_synthesis_qmod_amplification_iterations.py)
- [validate_qmod_iterations.py](../../../../../../../tools/validate_qmod_iterations.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `9368b9ce6ad4afbcdca16633316525d0e8bf2e05`.

Original SHA-256: `d4ba4f58dea107382503b71c3c4390c76e5c1978a27eb0a1f6781cf8a0a724a3`.

Approved SHA-256: `d4ba4f58dea107382503b71c3c4390c76e5c1978a27eb0a1f6781cf8a0a724a3`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
