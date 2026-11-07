# set partition

Category: applications / optimization.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/optimization/set_partition/set_partition.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/optimization/set_partition/set_partition.qmod'
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
    0.06
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 10 physical qubits, 196 operations, depth 43.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_records.py](../../../../../../../tests/integration/test_synthesis_qmod_records.py)
- [validate_qmod_records.py](../../../../../../../tools/validate_qmod_records.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `0fc084450cfaf46053c63cebcea58f923d137ef3`.

Original SHA-256: `9f81dad6502af4db63851b572bad3dac6b50ec69a0b01cd7a888c78bf5ee215e`.

Approved SHA-256: `9f81dad6502af4db63851b572bad3dac6b50ec69a0b01cd7a888c78bf5ee215e`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
