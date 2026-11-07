# maximum integer example

Category: tutorials / arithmetic.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/function_usage_examples/arithmetic/extremum/maximum_integer_example.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/function_usage_examples/arithmetic/extremum/maximum_integer_example.qmod'
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

Recorded resources: 7 physical qubits, 12 operations, depth 7.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_extrema.py](../../../../../../../../../tests/integration/test_synthesis_qmod_extrema.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `3fb6caa2b94624cac6022ca0fc9e6566f8144298`.

Original SHA-256: `a68827f07d67d85353e0bbbf1dade7382ba14cd2b961c0c25e9ddda3b9c7aedb`.

Approved SHA-256: `a68827f07d67d85353e0bbbf1dade7382ba14cd2b961c0c25e9ddda3b9c7aedb`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
