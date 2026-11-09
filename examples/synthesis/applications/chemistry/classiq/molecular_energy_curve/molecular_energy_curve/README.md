# molecular energy curve

Category: applications / chemistry.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/chemistry/molecular_energy_curve/molecular_energy_curve.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/chemistry/molecular_energy_curve/molecular_energy_curve.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "params": [
    0.37
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 1 physical qubits, 1 operations, depth 1.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_permutations.py](../../../../../../../tests/integration/test_synthesis_qmod_permutations.py)
- [validate_qmod_permutations.py](../../../../../../../tools/validate_qmod_permutations.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `d294a3e7acbb8eb7825577473196f8981cd6812e`.

Original SHA-256: `59cbf33ec5c867a45da3e381e2b7ec01cecacf9d3dbf4a6de04b425c6cd6c850`.

Approved SHA-256: `59cbf33ec5c867a45da3e381e2b7ec01cecacf9d3dbf4a6de04b425c6cd6c850`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
