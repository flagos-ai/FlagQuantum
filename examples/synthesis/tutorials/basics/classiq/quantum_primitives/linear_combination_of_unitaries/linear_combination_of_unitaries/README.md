# linear combination of unitaries

Category: tutorials / basics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/basic_tutorials/quantum_primitives/linear_combination_of_unitaries/linear_combination_of_unitaries.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/basic_tutorials/quantum_primitives/linear_combination_of_unitaries/linear_combination_of_unitaries.qmod'
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

Recorded resources: 4 physical qubits, 264 operations, depth 228.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_reversible_scopes.py](../../../../../../../../tests/integration/test_synthesis_qmod_reversible_scopes.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `5c79a60bcead7e8ed1ab65dad0cdb2462ebec14c`.

Original SHA-256: `cf263dae4a2539543029545a0a50d262e100ba81e9c303a1bd07a25a75580256`.

Approved SHA-256: `cf263dae4a2539543029545a0a50d262e100ba81e9c303a1bd07a25a75580256`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
