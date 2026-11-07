# subtraction float example

Category: tutorials / arithmetic.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/function_usage_examples/arithmetic/subtraction/subtraction_float_example.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/function_usage_examples/arithmetic/subtraction/subtraction_float_example.qmod'
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

Recorded resources: 20 physical qubits, 75 operations, depth 51.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_subtraction.py](../../../../../../../../../tests/integration/test_synthesis_qmod_subtraction.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `ba263fb49122a235a1c0d9f992feb384be828b39`.

Original SHA-256: `0175e6eadc55bc9d8270a2d9b83fe58fa658e7cbede123ce8ae569045c83788d`.

Approved SHA-256: `0175e6eadc55bc9d8270a2d9b83fe58fa658e7cbede123ce8ae569045c83788d`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
