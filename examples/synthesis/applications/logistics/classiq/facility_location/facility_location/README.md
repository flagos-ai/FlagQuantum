# facility location

Category: applications / logistics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/logistics/facility_location/facility_location.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/logistics/facility_location/facility_location.qmod'
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
    -0.03
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 17 physical qubits, 444 operations, depth 177.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_phase_factors.py](../../../../../../../tests/integration/test_synthesis_qmod_phase_factors.py)
- [validate_qmod_phase_factors.py](../../../../../../../tools/validate_qmod_phase_factors.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `c0c72d9e57c8d482cf956816f6d23b86435c8194`.

Original SHA-256: `bf193efb21cb352c03fe6fe55405728eefe8c6e3d597ba0bb6ce2a9cb432a02c`.

Approved SHA-256: `bf193efb21cb352c03fe6fe55405728eefe8c6e3d597ba0bb6ce2a9cb432a02c`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
