# qsvt matrix inversion

Category: algorithms / quantum_linear_solvers.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/quantum_linear_solvers/qsvt_matrix_inversion/qsvt_matrix_inversion.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/quantum_linear_solvers/qsvt_matrix_inversion/qsvt_matrix_inversion.qmod'
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

Recorded resources: 4 physical qubits, 67915 operations, depth 59872.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_qsvt_inversion.py](../../../../../../../tests/integration/test_synthesis_qmod_qsvt_inversion.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `a500b67faef23903562cab490002b0f40c35e398`.

Original SHA-256: `f72e8f606f21ea3d88f7918fc913f1afdd8474105485bb48ef8c20b1a971afff`.

Approved SHA-256: `f72e8f606f21ea3d88f7918fc913f1afdd8474105485bb48ef8c20b1a971afff`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
