# hhl

Category: algorithms / quantum_linear_solvers.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/quantum_linear_solvers/hhl/hhl.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/quantum_linear_solvers/hhl/hhl.qmod'
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

Recorded resources: 7 physical qubits, 207 operations, depth 150.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_compact_hhl.py](../../../../../../../tests/integration/test_synthesis_qmod_compact_hhl.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `2d3e6f4350ae70cacc1dddb44a16a5a9d046dfa6`.

Original SHA-256: `08271b3c67726af0381e77134ff16c1945ce2707dc7f1d53329193e351bd2d2b`.

Approved SHA-256: `08271b3c67726af0381e77134ff16c1945ce2707dc7f1d53329193e351bd2d2b`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
