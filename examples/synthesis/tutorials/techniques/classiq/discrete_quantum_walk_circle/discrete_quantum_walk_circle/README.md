# discrete quantum walk circle

Category: tutorials / techniques.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/technology_demonstrations/discrete_quantum_walk_circle/discrete_quantum_walk_circle.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/technology_demonstrations/discrete_quantum_walk_circle/discrete_quantum_walk_circle.qmod'
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

Recorded resources: 17 physical qubits, 168 operations, depth 160.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_coherent_walk.py](../../../../../../../tests/integration/test_synthesis_qmod_coherent_walk.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `899de47fed6104329c0afa8ef0bb5b606dc23af1`.

Original SHA-256: `3f3c42c6a51ad3c3004914c9d1299f618ef25b4df0f858714039927f148bff1e`.

Approved SHA-256: `3f3c42c6a51ad3c3004914c9d1299f618ef25b4df0f858714039927f148bff1e`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
