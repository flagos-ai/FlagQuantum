# hamiltonian simulation guide trotter

Category: algorithms / hamiltonian_simulation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/hamiltonian_simulation_guide_trotter.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/hamiltonian_simulation_guide_trotter.qmod'
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

Recorded resources: 2 physical qubits, 50 operations, depth 40.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_hamiltonian_simulation.py](../../../../../../../tests/integration/test_synthesis_qmod_hamiltonian_simulation.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `cf13420cf09c2303b39f07d77dae9066392371c2`.

Original SHA-256: `fa9c62491a5b68e9e0a9379109af01cc2d2093a6f57a6717336017c1efdbcb04`.

Approved SHA-256: `fa9c62491a5b68e9e0a9379109af01cc2d2093a6f57a6717336017c1efdbcb04`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
