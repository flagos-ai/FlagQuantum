# quantum teleportation

Category: algorithms / foundational.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/foundational/quantum_teleportation/quantum_teleportation.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/foundational/quantum_teleportation/quantum_teleportation.qmod'
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

Recorded resources: 3 physical qubits, 7 operations, depth 5.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_teleportation.py](../../../../../../../tests/integration/test_synthesis_qmod_teleportation.py)

Circuit-based validation source links: 0.

## Provenance

Original Git blob: `0166d196387730ae40ec04d9f3651b273521a88c`.

Original SHA-256: `7b30cdd5fb3ab8df192f36d9859b4db6b091fb3b5c50604aaa41e17e4b08d40c`.

Approved SHA-256: `7b30cdd5fb3ab8df192f36d9859b4db6b091fb3b5c50604aaa41e17e4b08d40c`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
