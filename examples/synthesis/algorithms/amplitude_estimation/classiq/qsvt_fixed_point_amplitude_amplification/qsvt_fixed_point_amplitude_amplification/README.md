# qsvt fixed point amplitude amplification

Category: algorithms / amplitude_estimation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/amplitude_amplification_and_estimation/qsvt_fixed_point_amplitude_amplification/qsvt_fixed_point_amplitude_amplification.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/amplitude_amplification_and_estimation/qsvt_fixed_point_amplitude_amplification/qsvt_fixed_point_amplitude_amplification.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
null
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 10 physical qubits, 20542 operations, depth 16850.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_fixed_point_qsvt.py](../../../../../../../tests/integration/test_synthesis_qmod_fixed_point_qsvt.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `caf41a36db0fae33bc507f3bab4c39f79329b5ef`.

Original SHA-256: `33f6ac40cf27c7a1014ca6ca973947d1deea48283f88e766dd01a4f5e054d380`.

Approved SHA-256: `33f6ac40cf27c7a1014ca6ca973947d1deea48283f88e766dd01a4f5e054d380`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
