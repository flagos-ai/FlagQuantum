# prepare bell state

Category: tutorials / state_preparation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_bell_state.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | [Corresponding native program](main.py) |
| Circuit comparison | [Runnable three-form comparison](comparisons/circuit.py) |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_bell_state.qmod'
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

Recorded resources: 2 physical qubits, 3 operations, depth 3.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

Run the native program and its three-form comparison:

```sh
python -m examples.synthesis.tutorials.state_preparation.classiq.qmod_library_reference.classiq_open_library.special_state_preparations.prepare_bell_state.main
python -m examples.synthesis.tutorials.state_preparation.classiq.qmod_library_reference.classiq_open_library.special_state_preparations.prepare_bell_state.comparisons.circuit
```

## Existing validation sources

- [test_synthesis_native_catalog.py](../../../../../../../../../tests/integration/test_synthesis_native_catalog.py)
- [test_synthesis_qmod_bell.py](../../../../../../../../../tests/integration/test_synthesis_qmod_bell.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `a848085bc035d413768249b115b2707152833a0d`.

Original SHA-256: `6ef951d1c554ef34ea1fc466c084a704b4d1b95e9b59cce535ac53fe330fc995`.

Approved SHA-256: `6ef951d1c554ef34ea1fc466c084a704b4d1b95e9b59cce535ac53fe330fc995`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
