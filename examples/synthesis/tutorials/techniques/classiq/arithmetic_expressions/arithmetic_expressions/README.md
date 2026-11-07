# arithmetic expressions

Category: tutorials / techniques.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/technology_demonstrations/arithmetic_expressions/arithmetic_expressions.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/technology_demonstrations/arithmetic_expressions/arithmetic_expressions.qmod'
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

Recorded resources: 5 physical qubits, 8 operations, depth 6.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_extrema.py](../../../../../../../tests/integration/test_synthesis_qmod_extrema.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `77672bda8b2754f9a8102d988b6a5ca2a8cec420`.

Original SHA-256: `9645adb6017fc3b4d61af89e6e4f9e13a4c8ed08dfea05e89a0d0a542168897c`.

Approved SHA-256: `9645adb6017fc3b4d61af89e6e4f9e13a4c8ed08dfea05e89a0d0a542168897c`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
