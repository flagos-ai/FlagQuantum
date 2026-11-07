# X

Category: tutorials / gates.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/qmod_core_library/standard_gates/X.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/qmod_core_library/standard_gates/X.qmod'
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

Recorded resources: 1 physical qubits, 1 operations, depth 1.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_qmod_parser_execution.py](../../../../../../../../../tests/integration/test_qmod_parser_execution.py)
- [test_synthesis_qmod_originals.py](../../../../../../../../../tests/integration/test_synthesis_qmod_originals.py)
- [test_qmod_comparisons.py](../../../../../../../../../tests/team/ecosystem/test_qmod_comparisons.py)
- [test_qmod_parser.py](../../../../../../../../../tests/team/ecosystem/test_qmod_parser.py)

Circuit-based validation source links: 0.

## Provenance

Original Git blob: `51df4959dfff80b44df7e288f82778349c10f48d`.

Original SHA-256: `36814f1f719f430310703d4ab137879477539dd0d490d043ae5dbabf1532f825`.

Approved SHA-256: `36814f1f719f430310703d4ab137879477539dd0d490d043ae5dbabf1532f825`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
