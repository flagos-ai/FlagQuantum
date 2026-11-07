# RZ

Category: tutorials / gates.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/qmod_core_library/standard_gates/RZ.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | [Corresponding native program](main.py) |
| Circuit comparison | [Runnable three-form comparison](comparisons/circuit.py) |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/qmod_core_library/standard_gates/RZ.qmod'
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
python -m examples.synthesis.tutorials.gates.classiq.qmod_library_reference.qmod_core_library.standard_gates.RZ.main
python -m examples.synthesis.tutorials.gates.classiq.qmod_library_reference.qmod_core_library.standard_gates.RZ.comparisons.circuit
```

## Existing validation sources

- [test_synthesis_native_catalog.py](../../../../../../../../../tests/integration/test_synthesis_native_catalog.py)
- [test_synthesis_qmod_rz.py](../../../../../../../../../tests/integration/test_synthesis_qmod_rz.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `d48bbb3be30677d16f80ceff216521f054cad3de`.

Original SHA-256: `5f9c03d460062801ae7d86cb78a20e87fa563ea40824ab8be76a09b0100d801c`.

Approved SHA-256: `5f9c03d460062801ae7d86cb78a20e87fa563ea40824ab8be76a09b0100d801c`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
