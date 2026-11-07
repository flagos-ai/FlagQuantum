# time marching

Category: algorithms / quantum_differential_equations_solvers.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/quantum_differential_equations_solvers/time_marching/time_marching.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/quantum_differential_equations_solvers/time_marching/time_marching.qmod'
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

Recorded resources: 15 physical qubits, 5118 operations, depth 4355.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_time_marching.py](../../../../../../../tests/integration/test_synthesis_qmod_time_marching.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `06c6b486196e804d90986cf04174bb8aff8cd941`.

Original SHA-256: `bd82222da499fc6fc5f07896173a88cadea46b68935df57d2374d18671ce31fe`.

Approved SHA-256: `bd82222da499fc6fc5f07896173a88cadea46b68935df57d2374d18671ce31fe`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
