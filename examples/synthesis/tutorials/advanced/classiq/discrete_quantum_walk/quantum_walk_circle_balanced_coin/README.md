# quantum walk circle balanced coin

Category: tutorials / advanced.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/advanced_tutorials/discrete_quantum_walk/quantum_walk_circle_balanced_coin.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/advanced_tutorials/discrete_quantum_walk/quantum_walk_circle_balanced_coin.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "t": 2
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 12 physical qubits, 159 operations, depth 153.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_circle_walk.py](../../../../../../../tests/integration/test_synthesis_qmod_circle_walk.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `c47f1b7bee16a2d7cb1bcb028423767b5bd7a3df`.

Original SHA-256: `6e2678a6619300633ece55616378183aa624f9deb78511b7b77744cddb8206dc`.

Approved SHA-256: `6e2678a6619300633ece55616378183aa624f9deb78511b7b77744cddb8206dc`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
