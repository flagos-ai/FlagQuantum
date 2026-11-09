# discrete quantum walk

Category: tutorials / advanced.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/advanced_tutorials/discrete_quantum_walk/discrete_quantum_walk.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/advanced_tutorials/discrete_quantum_walk/discrete_quantum_walk.qmod'
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

Recorded resources: 16 physical qubits, 150 operations, depth 142.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_permutations.py](../../../../../../../tests/integration/test_synthesis_qmod_permutations.py)
- [validate_qmod_permutations.py](../../../../../../../tools/validate_qmod_permutations.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `27185f41232ce8872e17110a0b46c53c43825ad0`.

Original SHA-256: `b76296df49abedb01568a603e3cabf0784bd05794ae8746d468b2acba897167b`.

Approved SHA-256: `b76296df49abedb01568a603e3cabf0784bd05794ae8746d468b2acba897167b`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
