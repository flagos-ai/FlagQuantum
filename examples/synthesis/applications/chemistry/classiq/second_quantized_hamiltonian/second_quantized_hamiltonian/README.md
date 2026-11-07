# second quantized hamiltonian

Category: applications / chemistry.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/chemistry/second_quantized_hamiltonian/second_quantized_hamiltonian.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/chemistry/second_quantized_hamiltonian/second_quantized_hamiltonian.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "params": [
    0.17,
    -0.22,
    0.31,
    0.48,
    -0.51,
    0.63,
    0.72,
    -0.83,
    0.96,
    0.12,
    -0.26,
    0.37
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 4 physical qubits, 29 operations, depth 13.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_hea.py](../../../../../../../tests/integration/test_synthesis_qmod_hea.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `f83362f3334afbcb48ffe3abef668e85fc0c3282`.

Original SHA-256: `cfdb0ce4d23027d6a2907227942fefacac7d75683ab2e6174b155b06cc6759e9`.

Approved SHA-256: `cfdb0ce4d23027d6a2907227942fefacac7d75683ab2e6174b155b06cc6759e9`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
