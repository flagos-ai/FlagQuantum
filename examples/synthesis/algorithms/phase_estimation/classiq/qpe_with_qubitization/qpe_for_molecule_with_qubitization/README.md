# qpe for molecule with qubitization

Category: algorithms / phase_estimation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/quantum_phase_estimation/qpe_with_qubitization/qpe_for_molecule_with_qubitization.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/quantum_phase_estimation/qpe_with_qubitization/qpe_for_molecule_with_qubitization.qmod'
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

Recorded resources: 15 physical qubits, 54289 operations, depth 45256.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_qubitization_qpe.py](../../../../../../../tests/integration/test_synthesis_qmod_qubitization_qpe.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `3ee281fe062bd7e4c1aeba92a47458cde68ae060`.

Original SHA-256: `3bc5118a82cd3bc66df0c01f31ee489e8acecdfa4ce8ea96344972fb86988943`.

Approved SHA-256: `3bc5118a82cd3bc66df0c01f31ee489e8acecdfa4ce8ea96344972fb86988943`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
