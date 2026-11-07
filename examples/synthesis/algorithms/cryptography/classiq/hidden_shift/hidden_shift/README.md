# hidden shift

Category: algorithms / cryptography.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/number_theory_and_cryptography/hidden_shift/hidden_shift.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | [Corresponding native program](../../../../hidden_shift/quantum_callbacks/main.py) |
| Circuit comparison | [Runnable three-form comparison](../../../../hidden_shift/quantum_callbacks/comparisons/circuit.py) |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/number_theory_and_cryptography/hidden_shift/hidden_shift.qmod'
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

Recorded resources: 4 physical qubits, 83 operations, depth 66.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

Run the native program and its three-form comparison:

```sh
python -m examples.synthesis.algorithms.hidden_shift.quantum_callbacks.main
python -m examples.synthesis.algorithms.hidden_shift.quantum_callbacks.comparisons.circuit
```

## Existing validation sources

- [test_synthesis_qmod_quantum_callbacks.py](../../../../../../../tests/integration/test_synthesis_qmod_quantum_callbacks.py)
- [validate_qmod_quantum_callbacks.py](../../../../../../../tools/validate_qmod_quantum_callbacks.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `476d80c851ffa24eb5375c70e9a3623fd9cc1af6`.

Original SHA-256: `13edfda2a42c77f76da852fb6d568c0df785f6991eaa6f79863d87f5cd499830`.

Approved SHA-256: `13edfda2a42c77f76da852fb6d568c0df785f6991eaa6f79863d87f5cd499830`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
