# variational quantum imaginary time evolution

Category: applications / optimization.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/optimization/variational_quantum_imaginary_time_evolution/variational_quantum_imaginary_time_evolution.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/optimization/variational_quantum_imaginary_time_evolution/variational_quantum_imaginary_time_evolution.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "angles": [
    0.13,
    -0.29,
    0.41,
    -0.57,
    0.68,
    -0.83
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 4 physical qubits, 28 operations, depth 18.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_single_pauli.py](../../../../../../../tests/integration/test_synthesis_qmod_single_pauli.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `45875e31fddf68e9632687358ec55da086e45dbb`.

Original SHA-256: `ff6fb42088d53455f43bdb8735479c6eba5b03b6bc5bc21910fdc45eff47f358`.

Approved SHA-256: `ff6fb42088d53455f43bdb8735479c6eba5b03b6bc5bc21910fdc45eff47f358`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
