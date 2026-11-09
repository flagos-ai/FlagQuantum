# unitary

Category: tutorials / library.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/qmod_core_library/unitary/unitary.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/qmod_core_library/unitary/unitary.qmod'
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

Recorded resources: 2 physical qubits, 9 operations, depth 7.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_unitary.py](../../../../../../../../../tests/integration/test_synthesis_qmod_unitary.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `5ddd508e6676254ab743256583b1dc2b7e67670a`.

Original SHA-256: `c9b50c054efea25b572133b2d9a9c7966df7d5d84adf6c111ad73221ad2b101b`.

Approved SHA-256: `c9b50c054efea25b572133b2d9a9c7966df7d5d84adf6c111ad73221ad2b101b`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
