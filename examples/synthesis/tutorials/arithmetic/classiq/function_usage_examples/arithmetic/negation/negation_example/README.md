# negation example

Category: tutorials / arithmetic.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/function_usage_examples/arithmetic/negation/negation_example.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/function_usage_examples/arithmetic/negation/negation_example.qmod'
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

Recorded resources: 20 physical qubits, 73 operations, depth 52.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_subtraction.py](../../../../../../../../../tests/integration/test_synthesis_qmod_subtraction.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `e6b8030e522ec664c34108fe8544829eef05ec89`.

Original SHA-256: `f3c18cce1355867a618573888ed01e62a1fbdff5a3d931f52417ed7360a1fac7`.

Approved SHA-256: `f3c18cce1355867a618573888ed01e62a1fbdff5a3d931f52417ed7360a1fac7`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
