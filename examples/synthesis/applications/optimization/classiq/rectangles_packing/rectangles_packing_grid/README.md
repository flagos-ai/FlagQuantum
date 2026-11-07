# rectangles packing grid

Category: applications / optimization.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/optimization/rectangles_packing/rectangles_packing_grid.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/optimization/rectangles_packing/rectangles_packing_grid.qmod'
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

Recorded resources: 17 physical qubits, 2574 operations, depth 636.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_cscope.py](../../../../../../../tests/integration/test_synthesis_qmod_cscope.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `149826034d817b538b3e64c9f6524337acc84d71`.

Original SHA-256: `5f8f6c8e5d84aa2897cf4e7b52b01b300b4ef01d86907c91c414388d5959f9a0`.

Approved SHA-256: `5f8f6c8e5d84aa2897cf4e7b52b01b300b4ef01d86907c91c414388d5959f9a0`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
