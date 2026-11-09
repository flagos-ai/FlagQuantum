# bpde

Category: applications / differential_equations.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/community/paper_implementation_project/bpde/bpde.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | [Corresponding native program](../../../../../../algorithms/hamiltonian_simulation/controlled_evolution/main.py) |
| Circuit comparison | [Runnable three-form comparison](../../../../../../algorithms/hamiltonian_simulation/controlled_evolution/comparisons/circuit.py) |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'community/paper_implementation_project/bpde/bpde.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "delta_e": 0.37
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 2 physical qubits, 71 operations, depth 70.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

Run the native program and its three-form comparison:

```sh
python -m examples.synthesis.algorithms.hamiltonian_simulation.controlled_evolution.main
python -m examples.synthesis.algorithms.hamiltonian_simulation.controlled_evolution.comparisons.circuit
```

## Existing validation sources

- [test_synthesis_qmod_controlled_evolution.py](../../../../../../../../tests/integration/test_synthesis_qmod_controlled_evolution.py)
- [validate_qmod_controlled_evolution.py](../../../../../../../../tools/validate_qmod_controlled_evolution.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `5bd44ca6b2f4f78f1cc195dae83df00da370db49`.

Original SHA-256: `4413d7ceb261befcd2989100f01e37742d588a82e78d34234bd2f6b0907b7dae`.

Approved SHA-256: `4413d7ceb261befcd2989100f01e37742d588a82e78d34234bd2f6b0907b7dae`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
