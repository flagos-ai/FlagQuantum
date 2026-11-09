# qlbm

Category: applications / fluid_dynamics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/CFD/qlbm/qlbm.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/CFD/qlbm/qlbm.qmod' --revision approved
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

Original result: `frontend_refused`. Approved result: `execution_smoke_passed`.

Recorded resources: 15 physical qubits, 15704 operations, depth 9109.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Approved numerical revision

Use equal amplitudes 0.5 for the uniform four-state preparation.

[Approved source](comparisons/accepted.qmod)

The original remains byte-identical and is expected to be refused.
Run the command without `--revision approved` to inspect that refusal.

Recorded original diagnostic:

```text
qmod.amplitudes: ../synth-proposal/classiq-qmod-audit/qmod/applications/CFD/qlbm/qlbm.qmod:13:30: inplace preparation requires a normalized finite real vector and a nonnegative finite bound
```

## Existing validation sources

A case-specific validation source has not yet been linked in this catalog.

Circuit-based validation source links: 0.

## Provenance

Original Git blob: `0dcedc96dfb4489b6660cb803cb83d7a0566099d`.

Original SHA-256: `5a3ac4d0d71a5841a8d65eb4ff35f015bd9d605ff83d265c358b1023d403eebf`.

Approved SHA-256: `8cb0e28d772910f81a5369f0a51a691be20a6b3b24eb3a9bcaee25f9bf6e0d36`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
