# algo design QCE tutorial part II

Category: tutorials / workshops.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/workshops/algo_design_QCE_tutorial/algo_design_QCE_tutorial_part_II.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/workshops/algo_design_QCE_tutorial/algo_design_QCE_tutorial_part_II.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "params": [
    0.003,
    -0.006,
    0.009000000000000001,
    -0.012,
    0.015,
    -0.018000000000000002
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 5 physical qubits, 856 operations, depth 721.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_guarded_phase.py](../../../../../../../tests/integration/test_synthesis_qmod_guarded_phase.py)
- [validate_qmod_knapsack.py](../../../../../../../tools/validate_qmod_knapsack.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `5f536d451355bf821353c1772d85e4d508f69be5`.

Original SHA-256: `777232ccf90d7004091b1d0648c9afdc9e72d2414e4d0fd7bc7c396538ed0a5a`.

Approved SHA-256: `777232ccf90d7004091b1d0648c9afdc9e72d2414e4d0fd7bc7c396538ed0a5a`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
