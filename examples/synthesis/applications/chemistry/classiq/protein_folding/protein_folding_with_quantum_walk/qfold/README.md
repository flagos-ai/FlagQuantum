# qfold

Category: applications / chemistry.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/chemistry/protein_folding/protein_folding_with_quantum_walk/qfold.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/chemistry/protein_folding/protein_folding_with_quantum_walk/qfold.qmod'
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

Recorded resources: 4 physical qubits, 87 operations, depth 69.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_qfold.py](../../../../../../../../tests/integration/test_synthesis_qmod_qfold.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `c49f7ef277e66371d1b12fcce8bb12f28e16336e`.

Original SHA-256: `b456350a74f0e34f7ee6206c224dc432ad43aa4b90be158076c8502ca8e2373f`.

Approved SHA-256: `b456350a74f0e34f7ee6206c224dc432ad43aa4b90be158076c8502ca8e2373f`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
