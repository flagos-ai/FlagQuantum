# execution tutorial part2

Category: tutorials / basics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/basic_tutorials/the_classiq_tutorial/execution_tutorial_part2.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/basic_tutorials/the_classiq_tutorial/execution_tutorial_part2.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "angles": [
    0.2,
    -0.3
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 2 physical qubits, 3 operations, depth 3.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_real_arrays.py](../../../../../../../tests/integration/test_synthesis_qmod_real_arrays.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `c53f7fdf9290e4f172fa99878a6bf1a2cf061244`.

Original SHA-256: `a5d068e4e3aafd7280db8560f9b0ee81339ce8cd631c026ee576f1660efc72a0`.

Approved SHA-256: `a5d068e4e3aafd7280db8560f9b0ee81339ce8cd631c026ee576f1660efc72a0`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
