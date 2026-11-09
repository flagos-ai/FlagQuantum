# qpe for molecules

Category: applications / chemistry.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/chemistry/qpe_for_molecules/qpe_for_molecules.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/chemistry/qpe_for_molecules/qpe_for_molecules.qmod'
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

Recorded resources: 13 physical qubits, 16931 operations, depth 15488.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_molecule_qpe.py](../../../../../../../tests/integration/test_synthesis_qmod_molecule_qpe.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `06b228153e7bab2d562c253fad3fe6a5d150bb38`.

Original SHA-256: `9dbdf81f075934e8179772046a8dd072f71c4f0edaf431161a8c72f779a49902`.

Approved SHA-256: `9dbdf81f075934e8179772046a8dd072f71c4f0edaf431161a8c72f779a49902`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
