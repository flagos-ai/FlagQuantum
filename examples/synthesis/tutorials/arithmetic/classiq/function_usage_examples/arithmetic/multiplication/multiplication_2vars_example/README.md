# multiplication 2vars example

Category: tutorials / arithmetic.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/function_usage_examples/arithmetic/multiplication/multiplication_2vars_example.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | [Corresponding native program](main.py) |
| Circuit comparison | [Runnable three-form comparison](comparisons/circuit.py) |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/function_usage_examples/arithmetic/multiplication/multiplication_2vars_example.qmod'
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

Recorded resources: 24 physical qubits, 275 operations, depth 214.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

Run the native program and its three-form comparison:

```sh
python -m examples.synthesis.tutorials.arithmetic.classiq.function_usage_examples.arithmetic.multiplication.multiplication_2vars_example.main
python -m examples.synthesis.tutorials.arithmetic.classiq.function_usage_examples.arithmetic.multiplication.multiplication_2vars_example.comparisons.circuit
```

## Existing validation sources

- [test_synthesis_native_catalog.py](../../../../../../../../../tests/integration/test_synthesis_native_catalog.py)
- [test_synthesis_qmod_originals.py](../../../../../../../../../tests/integration/test_synthesis_qmod_originals.py)
- [test_qmod_comparisons.py](../../../../../../../../../tests/team/ecosystem/test_qmod_comparisons.py)
- [test_qmod_parser.py](../../../../../../../../../tests/team/ecosystem/test_qmod_parser.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `2282dd8e5587da689c795f04cac2eaf7a1f2a569`.

Original SHA-256: `5322769f60fad0608e6c2446fb7b04d85828a277e00f401d1baed3b562016233`.

Approved SHA-256: `5322769f60fad0608e6c2446fb7b04d85828a277e00f401d1baed3b562016233`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
