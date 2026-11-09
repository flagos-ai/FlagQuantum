# grover max cut

Category: algorithms / grover.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/search_and_optimization/grover/grover_max_cut.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/search_and_optimization/grover/grover_max_cut.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "r": 2
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 7 physical qubits, 148 operations, depth 122.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_grover.py](../../../../../../../tests/integration/test_synthesis_qmod_grover.py)
- [validate_qmod_grover.py](../../../../../../../tools/validate_qmod_grover.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `defaa1ddf4de9c9aa1b8847bfd6842d33fb80f08`.

Original SHA-256: `0138bb2f8a6d85c54dd3d7953c818cc856b0c125dafd3201b438eb5d25529e53`.

Approved SHA-256: `0138bb2f8a6d85c54dd3d7953c818cc856b0c125dafd3201b438eb5d25529e53`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
