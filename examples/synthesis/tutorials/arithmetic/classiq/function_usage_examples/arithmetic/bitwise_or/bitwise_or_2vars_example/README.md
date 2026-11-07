# bitwise or 2vars example

Category: tutorials / arithmetic.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/function_usage_examples/arithmetic/bitwise_or/bitwise_or_2vars_example.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/function_usage_examples/arithmetic/bitwise_or/bitwise_or_2vars_example.qmod'
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

Recorded resources: 18 physical qubits, 30 operations, depth 8.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_unsigned_bitwise.py](../../../../../../../../../tests/integration/test_synthesis_qmod_unsigned_bitwise.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `620705df415ca6e09fe5ec3b92df176b3f54ac69`.

Original SHA-256: `d49707bafc06f659cbddaae72fe0663e7e2ddb1c74619237d5ebeae01d525153`.

Approved SHA-256: `d49707bafc06f659cbddaae72fe0663e7e2ddb1c74619237d5ebeae01d525153`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
