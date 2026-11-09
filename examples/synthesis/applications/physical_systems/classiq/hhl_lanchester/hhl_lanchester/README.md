# hhl lanchester

Category: applications / physical_systems.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/physical_systems/hhl_lanchester/hhl_lanchester.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/physical_systems/hhl_lanchester/hhl_lanchester.qmod'
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

Recorded resources: 16 physical qubits, 20213 operations, depth 18016.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_lanchester.py](../../../../../../../tests/integration/test_synthesis_qmod_lanchester.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `dd0db03afb47b47d91700b06749a3ffe2a71d493`.

Original SHA-256: `c352c8cfc2834da2a086ccda08d95d25c97816dc5daa767ad026a2838b9a8dfd`.

Approved SHA-256: `c352c8cfc2834da2a086ccda08d95d25c97816dc5daa767ad026a2838b9a8dfd`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
