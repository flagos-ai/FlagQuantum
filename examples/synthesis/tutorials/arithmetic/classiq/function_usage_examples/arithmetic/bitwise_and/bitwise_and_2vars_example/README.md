# bitwise and 2vars example

Category: tutorials / arithmetic.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/function_usage_examples/arithmetic/bitwise_and/bitwise_and_2vars_example.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/function_usage_examples/arithmetic/bitwise_and/bitwise_and_2vars_example.qmod'
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

Recorded resources: 9 physical qubits, 6 operations, depth 2.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_unsigned_bitwise.py](../../../../../../../../../tests/integration/test_synthesis_qmod_unsigned_bitwise.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `c76a6ecb65c3850ec4efb57a077f2829c24a2c5d`.

Original SHA-256: `19606f976123747e326be525173bfa132a289bee89b3c17d0697af8c71120358`.

Approved SHA-256: `19606f976123747e326be525173bfa132a289bee89b3c17d0697af8c71120358`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
