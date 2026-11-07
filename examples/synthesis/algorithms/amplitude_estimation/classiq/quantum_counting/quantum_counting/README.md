# quantum counting

Category: algorithms / amplitude_estimation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/amplitude_amplification_and_estimation/quantum_counting/quantum_counting.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/amplitude_amplification_and_estimation/quantum_counting/quantum_counting.qmod'
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

Original Git blob: `f85e591ce8fceb9967a3117fb5c5cb5ddf3de78a`.

Original SHA-256: `1406ca98e970189c574fbcaa2b58daca07c4cb84c1a703a30fb135905c4852f7`.

Approved SHA-256: `1406ca98e970189c574fbcaa2b58daca07c4cb84c1a703a30fb135905c4852f7`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
