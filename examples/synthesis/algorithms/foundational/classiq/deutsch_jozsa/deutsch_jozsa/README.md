# deutsch jozsa

Category: algorithms / foundational.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/foundational/deutsch_jozsa/deutsch_jozsa.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/foundational/deutsch_jozsa/deutsch_jozsa.qmod'
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

Recorded resources: 3 physical qubits, 1 operations, depth 1.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_quantum_callbacks.py](../../../../../../../tests/integration/test_synthesis_qmod_quantum_callbacks.py)
- [validate_qmod_quantum_callbacks.py](../../../../../../../tools/validate_qmod_quantum_callbacks.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `9b2bd0276d61651a05af8756902c7a7ff5dfd968`.

Original SHA-256: `bff22e530fb7b716bd30f846cc20efa9293c45072c2c89aa3f6bbc6e6e3b903e`.

Approved SHA-256: `bff22e530fb7b716bd30f846cc20efa9293c45072c2c89aa3f6bbc6e6e3b903e`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
