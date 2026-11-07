# gm qaoa

Category: algorithms / grover.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/search_and_optimization/grover_mixer_qaoa/gm_qaoa.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/search_and_optimization/grover_mixer_qaoa/gm_qaoa.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "params": [
    0.13,
    0.47,
    -0.39,
    0.11,
    0.9,
    -0.29
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 5 physical qubits, 172 operations, depth 151.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_gm_qaoa.py](../../../../../../../tests/integration/test_synthesis_qmod_gm_qaoa.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `fe9b15ecdf807969a3dfc2a45ec1d52ed2d69427`.

Original SHA-256: `2cc21d5787bc600ce85865478190a8ae8ab1f7a3d581cfe99b35ca8524786d4f`.

Approved SHA-256: `2cc21d5787bc600ce85865478190a8ae8ab1f7a3d581cfe99b35ca8524786d4f`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
