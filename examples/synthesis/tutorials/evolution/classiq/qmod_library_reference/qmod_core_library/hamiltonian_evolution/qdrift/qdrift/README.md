# qdrift

Category: tutorials / evolution.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/qmod_core_library/hamiltonian_evolution/qdrift/qdrift.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/qmod_core_library/hamiltonian_evolution/qdrift/qdrift.qmod'
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

Recorded resources: 2 physical qubits, 8 operations, depth 6.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

A case-specific validation source has not yet been linked in this catalog.

Circuit-based validation source links: 0.

## Provenance

Original Git blob: `ce93fff736a7c9a5ab3bb2a7a68bd08de8ccefa6`.

Original SHA-256: `9decab980923df21b25ca7210758fba218e165a72351481023c0370a36361e00`.

Approved SHA-256: `9decab980923df21b25ca7210758fba218e165a72351481023c0370a36361e00`.

[Classiq MIT license](../../../../../../../../CLASSIQ_LICENSE.txt)
