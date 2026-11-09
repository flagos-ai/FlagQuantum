# max induced k color subgraph

Category: applications / optimization.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/optimization/max_induced_k_color_subgraph/max_induced_k_color_subgraph.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/optimization/max_induced_k_color_subgraph/max_induced_k_color_subgraph.qmod'
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
    -0.018000000000000002,
    0.021,
    -0.024,
    0.027,
    -0.03,
    0.033,
    -0.036000000000000004,
    0.039,
    -0.042,
    0.045,
    -0.048
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 12 physical qubits, 33592 operations, depth 26085.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_records.py](../../../../../../../tests/integration/test_synthesis_qmod_records.py)
- [validate_qmod_records.py](../../../../../../../tools/validate_qmod_records.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `ae9da944b8adcd7003315172401656ad43810e11`.

Original SHA-256: `6719e725fd8b261d54ac1d556d51fc2679bbf684c2bc612b5f1f08d24a1cb1d2`.

Approved SHA-256: `6719e725fd8b261d54ac1d556d51fc2679bbf684c2bc612b5f1f08d24a1cb1d2`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
