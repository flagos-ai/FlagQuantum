# PHASE

Category: tutorials / gates.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/qmod_core_library/standard_gates/PHASE.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | [Corresponding native program](main.py) |
| Circuit comparison | [Runnable three-form comparison](comparisons/circuit.py) |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/qmod_core_library/standard_gates/PHASE.qmod'
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

Recorded resources: 1 physical qubits, 1 operations, depth 1.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

Run the native program and its three-form comparison:

```sh
python -m examples.synthesis.tutorials.gates.classiq.qmod_library_reference.qmod_core_library.standard_gates.PHASE.main
python -m examples.synthesis.tutorials.gates.classiq.qmod_library_reference.qmod_core_library.standard_gates.PHASE.comparisons.circuit
```

## Existing validation sources

- [test_synthesis_native_catalog.py](../../../../../../../../../tests/integration/test_synthesis_native_catalog.py)
- [test_synthesis_qmod_controlled_evolution.py](../../../../../../../../../tests/integration/test_synthesis_qmod_controlled_evolution.py)
- [test_synthesis_qmod_phase_gate.py](../../../../../../../../../tests/integration/test_synthesis_qmod_phase_gate.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `8163d1ad91e916fcc5fa6c5c5e71f61d934efbed`.

Original SHA-256: `8cd253f41e1ae55f32592e1a8af20c09409d970e365d4e84b5e0eed06bcca035`.

Approved SHA-256: `8cd253f41e1ae55f32592e1a8af20c09409d970e365d4e84b5e0eed06bcca035`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
