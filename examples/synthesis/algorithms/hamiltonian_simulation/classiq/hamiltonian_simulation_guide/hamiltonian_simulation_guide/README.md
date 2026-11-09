# hamiltonian simulation guide

Category: algorithms / hamiltonian_simulation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/hamiltonian_simulation_guide.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/hamiltonian_simulation_guide.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "t": 0.37
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 2 physical qubits, 5 operations, depth 5.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_hamiltonian_simulation.py](../../../../../../../tests/integration/test_synthesis_qmod_hamiltonian_simulation.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `523a87121cc5f69215d6bf1a99d19d2c9bc9f331`.

Original SHA-256: `7d8d28ede5f3eec55995990fcb402c820e91e07dc59ee71c54b17ceabb46c23b`.

Approved SHA-256: `7d8d28ede5f3eec55995990fcb402c820e91e07dc59ee71c54b17ceabb46c23b`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
