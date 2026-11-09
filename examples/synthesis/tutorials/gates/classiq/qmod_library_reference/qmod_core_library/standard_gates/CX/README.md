# CX

Category: tutorials / gates.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/qmod_core_library/standard_gates/CX.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | [Corresponding native program](main.py) |
| Circuit comparison | [Runnable three-form comparison](comparisons/circuit.py) |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/qmod_core_library/standard_gates/CX.qmod'
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

Recorded resources: 2 physical qubits, 1 operations, depth 1.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

Run the native program and its three-form comparison:

```sh
python -m examples.synthesis.tutorials.gates.classiq.qmod_library_reference.qmod_core_library.standard_gates.CX.main
python -m examples.synthesis.tutorials.gates.classiq.qmod_library_reference.qmod_core_library.standard_gates.CX.comparisons.circuit
```

## Existing validation sources

- [test_synthesis_native_catalog.py](../../../../../../../../../tests/integration/test_synthesis_native_catalog.py)
- [test_synthesis_qmod_pair_gates.py](../../../../../../../../../tests/integration/test_synthesis_qmod_pair_gates.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `3e4f9c88d041931f66405dd805c36cb51350ac54`.

Original SHA-256: `d1137ef45b570e6283be17fc616bd836387f54b71482e33b0fba9582fa183bf3`.

Approved SHA-256: `d1137ef45b570e6283be17fc616bd836387f54b71482e33b0fba9582fa183bf3`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
