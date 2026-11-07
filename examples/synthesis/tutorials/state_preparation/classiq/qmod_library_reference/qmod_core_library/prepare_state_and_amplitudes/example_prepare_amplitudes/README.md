# example prepare amplitudes

Category: tutorials / state_preparation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/qmod_core_library/prepare_state_and_amplitudes/example_prepare_amplitudes.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | [Corresponding native program](main.py) |
| Circuit comparison | [Runnable three-form comparison](comparisons/circuit.py) |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/qmod_core_library/prepare_state_and_amplitudes/example_prepare_amplitudes.qmod'
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

Recorded resources: 3 physical qubits, 13 operations, depth 11.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

Run the native program and its three-form comparison:

```sh
python -m examples.synthesis.tutorials.state_preparation.classiq.qmod_library_reference.qmod_core_library.prepare_state_and_amplitudes.example_prepare_amplitudes.main
python -m examples.synthesis.tutorials.state_preparation.classiq.qmod_library_reference.qmod_core_library.prepare_state_and_amplitudes.example_prepare_amplitudes.comparisons.circuit
```

## Existing validation sources

- [test_synthesis_native_catalog.py](../../../../../../../../../tests/integration/test_synthesis_native_catalog.py)
- [test_synthesis_qmod_amplitudes.py](../../../../../../../../../tests/integration/test_synthesis_qmod_amplitudes.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `fa7563598bbe60e6ea37a6f5dc888b11593b8744`.

Original SHA-256: `56f1962a04ea294d5b928d1a33c34540f6410ef016fa4ee1bc452f07cae36baa`.

Approved SHA-256: `56f1962a04ea294d5b928d1a33c34540f6410ef016fa4ee1bc452f07cae36baa`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
