# vqls with lcu

Category: algorithms / quantum_linear_solvers.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/quantum_linear_solvers/vqls/vqls_with_lcu.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/quantum_linear_solvers/vqls/vqls_with_lcu.qmod'
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
    0.18
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 3 physical qubits, 15 operations, depth 9.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_real_helpers.py](../../../../../../../tests/integration/test_synthesis_qmod_real_helpers.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `5cf9bac60fb7ac51c02147e1bbe79b8699c643ac`.

Original SHA-256: `9dcfb911d4df8d719b95331e5a1291158a33a91de7efa64f2e7d6ef33b14a0ae`.

Approved SHA-256: `9dcfb911d4df8d719b95331e5a1291158a33a91de7efa64f2e7d6ef33b14a0ae`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
