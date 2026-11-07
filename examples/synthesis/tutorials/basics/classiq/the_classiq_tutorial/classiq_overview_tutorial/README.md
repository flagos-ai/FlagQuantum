# classiq overview tutorial

Category: tutorials / basics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/basic_tutorials/the_classiq_tutorial/classiq_overview_tutorial.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | [Corresponding native program](main.py) |
| Circuit comparison | [Runnable three-form comparison](comparisons/circuit.py) |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/basic_tutorials/the_classiq_tutorial/classiq_overview_tutorial.qmod'
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

Recorded resources: 10 physical qubits, 16 operations, depth 12.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

Run the native program and its three-form comparison:

```sh
python -m examples.synthesis.tutorials.basics.classiq.the_classiq_tutorial.classiq_overview_tutorial.main
python -m examples.synthesis.tutorials.basics.classiq.the_classiq_tutorial.classiq_overview_tutorial.comparisons.circuit
```

## Existing validation sources

- [test_synthesis_native_catalog.py](../../../../../../../tests/integration/test_synthesis_native_catalog.py)
- [test_qmod_tutorial_sources.py](../../../../../../../tests/team/ecosystem/test_qmod_tutorial_sources.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `1f476102f49ad56009d5329a8f180c1596be61dc`.

Original SHA-256: `5e28d56fea67b203a6e0e0b44ef0fa8998e66ae477fc62a4b02d2c5c8bcd30d1`.

Approved SHA-256: `5e28d56fea67b203a6e0e0b44ef0fa8998e66ae477fc62a4b02d2c5c8bcd30d1`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
