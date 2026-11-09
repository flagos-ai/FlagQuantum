# qmc user defined

Category: algorithms / amplitude_estimation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/amplitude_amplification_and_estimation/qmc_user_defined/qmc_user_defined.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/amplitude_amplification_and_estimation/qmc_user_defined/qmc_user_defined.qmod'
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

Recorded resources: 7 physical qubits, 1819 operations, depth 1607.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_user_defined_qmc.py](../../../../../../../tests/integration/test_synthesis_qmod_user_defined_qmc.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `39582518486519a476e388ee5a739bfb45d6c331`.

Original SHA-256: `75823b088f4e730ff750c3bdb7c8b4a2f1d5a8dfdb22e14a3111e5593555a20d`.

Approved SHA-256: `75823b088f4e730ff750c3bdb7c8b4a2f1d5a8dfdb22e14a3111e5593555a20d`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
