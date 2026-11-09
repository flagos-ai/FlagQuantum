# classiq discrete quantum walk

Category: tutorials / techniques.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/technology_demonstrations/classiq_paper/quantum_walk/classiq_discrete_quantum_walk.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/technology_demonstrations/classiq_paper/quantum_walk/classiq_discrete_quantum_walk.qmod'
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

Recorded resources: 10 physical qubits, 46 operations, depth 46.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_coherent_walk.py](../../../../../../../../tests/integration/test_synthesis_qmod_coherent_walk.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `fa16163c96db9462af9e9e1b21cee28811103776`.

Original SHA-256: `23bf1e8f976a44b5e515f13f4c6e24ebb1f0f50bb8c1ec2d4829bd8dfd8abf19`.

Approved SHA-256: `23bf1e8f976a44b5e515f13f4c6e24ebb1f0f50bb8c1ec2d4829bd8dfd8abf19`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
