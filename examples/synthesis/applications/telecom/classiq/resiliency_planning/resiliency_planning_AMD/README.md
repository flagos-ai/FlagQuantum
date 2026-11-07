# resiliency planning AMD

Category: applications / telecom.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/telecom/resiliency_planning/resiliency_planning_AMD.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/telecom/resiliency_planning/resiliency_planning_AMD.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "params": [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.1,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
    0.16,
    0.17,
    0.18,
    0.19,
    0.2,
    0.21,
    0.22,
    0.23,
    0.24,
    0.25,
    0.26,
    0.27,
    0.28,
    0.29,
    0.3,
    0.31,
    0.32,
    0.33,
    0.34,
    0.35000000000000003,
    0.36,
    0.37,
    0.38,
    0.39,
    0.4
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 24 physical qubits, 2201 operations, depth 269.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_records.py](../../../../../../../tests/integration/test_synthesis_qmod_records.py)
- [test_synthesis_qmod_sparse_phase.py](../../../../../../../tests/integration/test_synthesis_qmod_sparse_phase.py)
- [validate_qmod_sparse_phase.py](../../../../../../../tools/validate_qmod_sparse_phase.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `0034d5aba2801be900a83a9171b36c1c0a5c77e8`.

Original SHA-256: `7742c197895a44808e86f4282fdfda5289b1e6030e0900d05b01ad07a622f7d0`.

Approved SHA-256: `7742c197895a44808e86f4282fdfda5289b1e6030e0900d05b01ad07a622f7d0`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
