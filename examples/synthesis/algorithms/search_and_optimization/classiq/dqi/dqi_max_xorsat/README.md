# dqi max xorsat

Category: algorithms / search_and_optimization.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/search_and_optimization/dqi/dqi_max_xorsat.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/search_and_optimization/dqi/dqi_max_xorsat.qmod'
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

Recorded resources: 14 physical qubits, 3802 operations, depth 1988.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_dqi_consuming.py](../../../../../../../tests/integration/test_synthesis_qmod_dqi_consuming.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `5c1cc2da7166b2a182a751b64a4f4febed899b64`.

Original SHA-256: `fe304ca66cbc23e7628e0358cc7d101b76b59a06b34c50162dae997b93bd613e`.

Approved SHA-256: `fe304ca66cbc23e7628e0358cc7d101b76b59a06b34c50162dae997b93bd613e`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
