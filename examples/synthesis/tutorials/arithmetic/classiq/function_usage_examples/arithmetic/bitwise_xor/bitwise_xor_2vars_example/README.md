# bitwise xor 2vars example

Category: tutorials / arithmetic.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/function_usage_examples/arithmetic/bitwise_xor/bitwise_xor_2vars_example.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/function_usage_examples/arithmetic/bitwise_xor/bitwise_xor_2vars_example.qmod'
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

Recorded resources: 18 physical qubits, 24 operations, depth 6.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_unsigned_bitwise.py](../../../../../../../../../tests/integration/test_synthesis_qmod_unsigned_bitwise.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `e9247706ed18cce9a048f2b046b8752c924643e1`.

Original SHA-256: `162f3d7d09c7d31f02afd80bef8d9f017f5181f55adce65a0d7a7120efaf896a`.

Approved SHA-256: `162f3d7d09c7d31f02afd80bef8d9f017f5181f55adce65a0d7a7120efaf896a`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
