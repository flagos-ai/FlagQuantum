# hadamard test

Category: algorithms / primitives.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/quantum_primitives/hadamard_test/hadamard_test.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/quantum_primitives/hadamard_test/hadamard_test.qmod'
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

Recorded resources: 5 physical qubits, 51 operations, depth 46.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_hadamard_test.py](../../../../../../../tests/integration/test_synthesis_qmod_hadamard_test.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `1759331eb973dc306eed7a9e17c25efb9cf4b61f`.

Original SHA-256: `f7e5b409fe56c5dd4de9f23d6127d3457b103b4029ebd67836947e5e28eb4709`.

Approved SHA-256: `f7e5b409fe56c5dd4de9f23d6127d3457b103b4029ebd67836947e5e28eb4709`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
