# stateprep guassian using qsvt

Category: algorithms / state_preparation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/community/paper_implementation_project/quantum_state_preparation_without_coherent_arithmetic/stateprep_guassian_using_qsvt.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'community/paper_implementation_project/quantum_state_preparation_without_coherent_arithmetic/stateprep_guassian_using_qsvt.qmod' --revision approved
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

Recorded resources: 17 physical qubits, 3194 operations, depth 2873.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Approved numerical revision

Use the marked overlap of the unchanged phase sequences.

[Approved source](comparisons/accepted.qmod)

The original remains byte-identical and is expected to be refused.
Run the command without `--revision approved` to inspect that refusal.

Recorded original diagnostic:

```text
qmod.amplification.amplitude: ../synth-proposal/classiq-qmod-audit/qmod/community/paper_implementation_project/quantum_state_preparation_without_coherent_arithmetic/stateprep_guassian_using_qsvt.qmod:105:33: supplied amplitude 0.39579583583035999 disagrees with proven amplitude 0.39191102421483637; original parameters are not corrected
```

## Existing validation sources

A case-specific validation source has not yet been linked in this catalog.

Circuit-based validation source links: 0.

## Provenance

Original Git blob: `55557082fc4d254ee9ba37d8a4a5bb8d2a3c7a00`.

Original SHA-256: `3bee74cf6e620f933c033a51933fb7adb4cfc2e1ecdc74336e51cce857055d51`.

Approved SHA-256: `28d5beca520401249ad439cf3ff3e67e62da26edad6cbd1ba8aeab97f73e82f2`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
