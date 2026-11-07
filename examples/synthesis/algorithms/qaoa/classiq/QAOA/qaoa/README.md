# qaoa

Category: algorithms / qaoa.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/search_and_optimization/QAOA/qaoa.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/search_and_optimization/QAOA/qaoa.qmod'
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

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `7cc0e92318f042f1d16b31345f8b3a0f6139dac6`.

Original SHA-256: `583f2319069ff7687248c631c1c26dcc7b63550dbcc74ba0363dd0b0da0bb29b`.

Approved SHA-256: `583f2319069ff7687248c631c1c26dcc7b63550dbcc74ba0363dd0b0da0bb29b`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
