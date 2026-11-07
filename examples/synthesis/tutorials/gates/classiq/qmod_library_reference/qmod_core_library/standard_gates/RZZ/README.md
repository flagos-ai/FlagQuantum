# RZZ

Category: tutorials / gates.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/qmod_core_library/standard_gates/RZZ.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/qmod_core_library/standard_gates/RZZ.qmod'
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

Recorded resources: 2 physical qubits, 1 operations, depth 1.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_rzz.py](../../../../../../../../../tests/integration/test_synthesis_qmod_rzz.py)

Circuit-based validation source links: 0.

## Provenance

Original Git blob: `3d19d2b567f23800acee8db04dc348d79865f577`.

Original SHA-256: `68c5550c8877b6b6a5aab78a2c5642cef363d3891476900c7b8738595e6c659d`.

Approved SHA-256: `68c5550c8877b6b6a5aab78a2c5642cef363d3891476900c7b8738595e6c659d`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
