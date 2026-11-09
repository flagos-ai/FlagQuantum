# quantum compression algorithm for symmetric states

Category: algorithms / state_preparation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/community/paper_implementation_project/quantum_compression_algorithm_for_symmetric_states/quantum_compression_algorithm_for_symmetric_states.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'community/paper_implementation_project/quantum_compression_algorithm_for_symmetric_states/quantum_compression_algorithm_for_symmetric_states.qmod'
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

Recorded resources: 3 physical qubits, 406 operations, depth 355.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_indexed_control.py](../../../../../../../../tests/integration/test_synthesis_qmod_indexed_control.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `6d15169fda6a7eaec81ed2a2fb6be8f711c7780d`.

Original SHA-256: `79c03c4b8dca7c5c37ae9cbb48d62b5f88e2e71d27441bc8c252eb9507576f62`.

Approved SHA-256: `79c03c4b8dca7c5c37ae9cbb48d62b5f88e2e71d27441bc8c252eb9507576f62`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
