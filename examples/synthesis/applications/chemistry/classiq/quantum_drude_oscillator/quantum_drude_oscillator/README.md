# quantum drude oscillator

Category: applications / chemistry.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/chemistry/quantum_drude_oscillator/quantum_drude_oscillator.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/chemistry/quantum_drude_oscillator/quantum_drude_oscillator.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "params": [
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.12,
    0.14,
    0.16,
    0.18,
    0.2,
    0.22,
    0.24,
    0.26,
    0.28,
    0.3,
    0.32,
    0.34,
    0.36,
    0.38,
    0.4,
    0.42,
    0.44,
    0.46,
    0.48,
    0.5,
    0.52,
    0.54,
    0.56,
    0.58,
    0.6,
    0.62,
    0.64,
    0.66,
    0.68,
    0.7,
    0.72,
    0.74,
    0.76,
    0.78,
    0.8
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 6 physical qubits, 90 operations, depth 23.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_real_helpers.py](../../../../../../../tests/integration/test_synthesis_qmod_real_helpers.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `3a0aa8228232a15814868cbf0595f2f45de0cac3`.

Original SHA-256: `bc376abfa2ddbc8250b6fc720a2c8578853e526aad66ce29c8729f05e9a36ad6`.

Approved SHA-256: `bc376abfa2ddbc8250b6fc720a2c8578853e526aad66ce29c8729f05e9a36ad6`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
