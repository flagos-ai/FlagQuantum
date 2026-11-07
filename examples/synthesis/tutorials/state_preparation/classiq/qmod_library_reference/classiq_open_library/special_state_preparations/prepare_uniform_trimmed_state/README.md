# prepare uniform trimmed state

Category: tutorials / state_preparation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_uniform_trimmed_state.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | [Corresponding native program](main.py) |
| Circuit comparison | [Runnable three-form comparison](comparisons/circuit.py) |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_uniform_trimmed_state.qmod'
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

Recorded resources: 4 physical qubits, 54 operations, depth 44.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

Run the native program and its three-form comparison:

```sh
python -m examples.synthesis.tutorials.state_preparation.classiq.qmod_library_reference.classiq_open_library.special_state_preparations.prepare_uniform_trimmed_state.main
python -m examples.synthesis.tutorials.state_preparation.classiq.qmod_library_reference.classiq_open_library.special_state_preparations.prepare_uniform_trimmed_state.comparisons.circuit
```

## Existing validation sources

- [test_synthesis_qmod_uniform.py](../../../../../../../../../tests/integration/test_synthesis_qmod_uniform.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `77d6bdea6739acd5a889c6f0f62484652198faad`.

Original SHA-256: `e4dab9a7b36a10f1d1f0dd84da4b761723afc364269bf90e6a39faab27a02da3`.

Approved SHA-256: `e4dab9a7b36a10f1d1f0dd84da4b761723afc364269bf90e6a39faab27a02da3`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
