# 3 sat grover

Category: algorithms / grover.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/search_and_optimization/grover/3_sat_grover.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/search_and_optimization/grover/3_sat_grover.qmod'
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

Recorded resources: 5 physical qubits, 46 operations, depth 33.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_grover.py](../../../../../../../tests/integration/test_synthesis_qmod_grover.py)
- [validate_qmod_grover.py](../../../../../../../tools/validate_qmod_grover.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `f995a51dc48bfe6930178457e33aa1802a48459c`.

Original SHA-256: `c33d090e750dd4b3fcae7d2bd26d72d655297f27e7ce45ab50a697a6e6c58e72`.

Approved SHA-256: `c33d090e750dd4b3fcae7d2bd26d72d655297f27e7ce45ab50a697a6e6c58e72`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
