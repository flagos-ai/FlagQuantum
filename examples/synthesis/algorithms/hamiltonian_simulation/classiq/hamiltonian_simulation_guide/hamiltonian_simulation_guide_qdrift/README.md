# hamiltonian simulation guide qdrift

Category: algorithms / hamiltonian_simulation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/hamiltonian_simulation_guide_qdrift.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/hamiltonian_simulation_guide_qdrift.qmod'
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

Recorded resources: 2 physical qubits, 225 operations, depth 197.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_hamiltonian_simulation.py](../../../../../../../tests/integration/test_synthesis_qmod_hamiltonian_simulation.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `b0a9536f8aecdbf011467984598b9d564ee4f9ef`.

Original SHA-256: `6e67fc80d979fd09f3ae45961688ce947dd397bb18e2f10f678a51d8a9f44f17`.

Approved SHA-256: `6e67fc80d979fd09f3ae45961688ce947dd397bb18e2f10f678a51d8a9f44f17`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
