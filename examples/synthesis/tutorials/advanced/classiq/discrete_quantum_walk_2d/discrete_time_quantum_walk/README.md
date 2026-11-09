# discrete time quantum walk

Category: tutorials / advanced.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/advanced_tutorials/discrete_quantum_walk_2d/discrete_time_quantum_walk.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/advanced_tutorials/discrete_quantum_walk_2d/discrete_time_quantum_walk.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "t": 2
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 15 physical qubits, 198 operations, depth 186.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_grover_walk.py](../../../../../../../tests/integration/test_synthesis_qmod_grover_walk.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `563f5eb9e436f602d0373b0178b316436a157ca3`.

Original SHA-256: `5b57d9d15e640714aa04d95aaa54c46d320dc7b31b0c8e8ea30c17b5f7dcc3ce`.

Approved SHA-256: `5b57d9d15e640714aa04d95aaa54c46d320dc7b31b0c8e8ea30c17b5f7dcc3ce`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
