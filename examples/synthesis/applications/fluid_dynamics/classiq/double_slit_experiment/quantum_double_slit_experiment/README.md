# quantum double slit experiment

Category: applications / fluid_dynamics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/CFD/double_slit_experiment/quantum_double_slit_experiment.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/CFD/double_slit_experiment/quantum_double_slit_experiment.qmod'
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

Recorded resources: 24 physical qubits, 66509 operations, depth 48718.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_double_slit_gate_tape.py](../../../../../../../tests/integration/test_synthesis_qmod_double_slit_gate_tape.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `3cf844c4844217a622adeefdfa38255f480ec95b`.

Original SHA-256: `ec6de8a9d82cad74f9c98aa3255728a652d275194f77e6bdee2d9e3717d59736`.

Approved SHA-256: `ec6de8a9d82cad74f9c98aa3255728a652d275194f77e6bdee2d9e3717d59736`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
