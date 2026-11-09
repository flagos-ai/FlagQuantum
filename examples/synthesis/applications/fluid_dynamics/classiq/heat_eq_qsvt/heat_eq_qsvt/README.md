# heat eq qsvt

Category: applications / fluid_dynamics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/CFD/heat_eq_qsvt/heat_eq_qsvt.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/CFD/heat_eq_qsvt/heat_eq_qsvt.qmod'
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

Recorded resources: 11 physical qubits, 267 operations, depth 202.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_qsvt.py](../../../../../../../tests/integration/test_synthesis_qmod_qsvt.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `07aad5fc6528d6dfe82c9ea0681179f17ebc5734`.

Original SHA-256: `859f73381e5697352c20ed963cfde455c96381108b318d25813f4d6bcf820ef6`.

Approved SHA-256: `859f73381e5697352c20ed963cfde455c96381108b318d25813f4d6bcf820ef6`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
