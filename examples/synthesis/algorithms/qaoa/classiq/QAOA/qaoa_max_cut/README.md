# qaoa max cut

Category: algorithms / qaoa.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/search_and_optimization/QAOA/qaoa_max_cut.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/search_and_optimization/QAOA/qaoa_max_cut.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "params": [
    0.1,
    0.2,
    0.30000000000000004,
    0.4,
    0.5,
    0.6000000000000001,
    0.7000000000000001,
    0.8
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 5 physical qubits, 64 operations, depth 25.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_qaoa_phase.py](../../../../../../../tests/integration/test_synthesis_qmod_qaoa_phase.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `443e8e8117b21375b608640c5cd54b5d34b720f1`.

Original SHA-256: `28d6de28d50734a8b0c27e85bd81f3ca047af754b7b2e18284865b02ce178e32`.

Approved SHA-256: `28d6de28d50734a8b0c27e85bd81f3ca047af754b7b2e18284865b02ce178e32`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
