# discrete poisson solver

Category: algorithms / quantum_differential_equations_solvers.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/quantum_differential_equations_solvers/discrete_poisson_solver/discrete_poisson_solver.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/quantum_differential_equations_solvers/discrete_poisson_solver/discrete_poisson_solver.qmod'
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

Recorded resources: 13 physical qubits, 5632 operations, depth 2657.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_discrete_poisson.py](../../../../../../../tests/integration/test_synthesis_qmod_discrete_poisson.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `21f596cadf41381e87007196f227d1eda34a5a89`.

Original SHA-256: `c26326402f1d9970e4e1b592b3b4665e419b96adf756ed0a43d5d63047e4a211`.

Approved SHA-256: `c26326402f1d9970e4e1b592b3b4665e419b96adf756ed0a43d5d63047e4a211`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
