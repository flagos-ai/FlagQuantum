# quantum sawtooth map

Category: applications / physical_systems.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/physical_systems/quantum_chaos/quantum_sawtooth_map.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/physical_systems/quantum_chaos/quantum_sawtooth_map.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "kick_mag": 0.43,
  "num_kicks": 2
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 3 physical qubits, 49 operations, depth 36.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_fourier_phases.py](../../../../../../../tests/integration/test_synthesis_qmod_fourier_phases.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `3d08cbb6f3727535e46851c4d54b2c8bd6a112b0`.

Original SHA-256: `36a17564a2c366044fc84d36e2a017aeb645ef86bc27445d4344ce19fc34b1b7`.

Approved SHA-256: `36a17564a2c366044fc84d36e2a017aeb645ef86bc27445d4344ce19fc34b1b7`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
