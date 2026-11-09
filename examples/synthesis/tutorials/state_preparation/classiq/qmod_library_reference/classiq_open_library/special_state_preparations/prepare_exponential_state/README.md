# prepare exponential state

Category: tutorials / state_preparation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_exponential_state.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_exponential_state.qmod'
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

Recorded resources: 5 physical qubits, 5 operations, depth 1.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_exponential_state.py](../../../../../../../../../tests/integration/test_synthesis_qmod_exponential_state.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `8ebfc5c90d89de0f293538f2d1a1afe211d7ea75`.

Original SHA-256: `9b0f65a58d107bbe25217e5a6ff9f08c2eb1a55a3ca751f09c3aae9c85b09730`.

Approved SHA-256: `9b0f65a58d107bbe25217e5a6ff9f08c2eb1a55a3ca751f09c3aae9c85b09730`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
