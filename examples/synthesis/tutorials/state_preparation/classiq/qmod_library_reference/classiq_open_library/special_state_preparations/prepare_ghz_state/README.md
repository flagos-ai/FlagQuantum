# prepare ghz state

Category: tutorials / state_preparation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_ghz_state.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | [Corresponding native program](main.py) |
| Circuit comparison | [Runnable three-form comparison](comparisons/circuit.py) |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_ghz_state.qmod'
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

Recorded resources: 5 physical qubits, 5 operations, depth 5.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

Run the native program and its three-form comparison:

```sh
python -m examples.synthesis.tutorials.state_preparation.classiq.qmod_library_reference.classiq_open_library.special_state_preparations.prepare_ghz_state.main
python -m examples.synthesis.tutorials.state_preparation.classiq.qmod_library_reference.classiq_open_library.special_state_preparations.prepare_ghz_state.comparisons.circuit
```

## Existing validation sources

- [test_synthesis_native_catalog.py](../../../../../../../../../tests/integration/test_synthesis_native_catalog.py)
- [test_synthesis_qmod_ghz.py](../../../../../../../../../tests/integration/test_synthesis_qmod_ghz.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `e8666b8b49d1de36de614a8f2897afb869cbdbe7`.

Original SHA-256: `7c0d3be2c11fbca40a24f617d53a0257f375cec4e05ce0102d73ea45ac0b5992`.

Approved SHA-256: `7c0d3be2c11fbca40a24f617d53a0257f375cec4e05ce0102d73ea45ac0b5992`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
