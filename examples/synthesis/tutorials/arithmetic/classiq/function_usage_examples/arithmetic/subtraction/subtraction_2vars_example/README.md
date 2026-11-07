# subtraction 2vars example

Category: tutorials / arithmetic.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/function_usage_examples/arithmetic/subtraction/subtraction_2vars_example.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/function_usage_examples/arithmetic/subtraction/subtraction_2vars_example.qmod'
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

Recorded resources: 19 physical qubits, 69 operations, depth 51.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_subtraction.py](../../../../../../../../../tests/integration/test_synthesis_qmod_subtraction.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `5ba63d99f072d7af128c0fe89744f38c256b4ea5`.

Original SHA-256: `d56d23b9aef1e0693532703ba68f891d3c1ef24b74db950622507ccb84efc8b4`.

Approved SHA-256: `d56d23b9aef1e0693532703ba68f891d3c1ef24b74db950622507ccb84efc8b4`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
