# SWAP

Category: tutorials / gates.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/qmod_core_library/standard_gates/SWAP.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/qmod_core_library/standard_gates/SWAP.qmod'
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

- [test_synthesis_qmod_pair_gates.py](../../../../../../../../../tests/integration/test_synthesis_qmod_pair_gates.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `0a733958bc25b479d7f679f41e0e2d7ba793a50a`.

Original SHA-256: `2e37c3cefa75dd52af1292b4790b5698654428e4ad6f3b6272afbbf90e537ad2`.

Approved SHA-256: `2e37c3cefa75dd52af1292b4790b5698654428e4ad6f3b6272afbbf90e537ad2`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
