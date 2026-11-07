# hybrid qnn for subset majority

Category: algorithms / machine_learning.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/QML/hybrid_qnn/hybrid_qnn_for_subset_majority.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/QML/hybrid_qnn/hybrid_qnn_for_subset_majority.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "input_vec": [
    0.2,
    -0.3,
    0.4,
    -0.5
  ],
  "weight": [
    0.1,
    0.2,
    0.30000000000000004
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 2 physical qubits, 5 operations, depth 3.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_rzz_views.py](../../../../../../../tests/integration/test_synthesis_qmod_rzz_views.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `7c90600dc38a1edc6b0ef236d90e28707253c1d4`.

Original SHA-256: `e7070708b270852ae1301029ebf4af23114487f8e1b7f21b3b14f4b54f781b6e`.

Approved SHA-256: `e7070708b270852ae1301029ebf4af23114487f8e1b7f21b3b14f4b54f781b6e`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
