# gqsp

Category: algorithms / primitives.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/quantum_primitives/gqsp/gqsp.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/quantum_primitives/gqsp/gqsp.qmod'
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

Recorded resources: 8 physical qubits, 41 operations, depth 34.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_gqsp.py](../../../../../../../tests/integration/test_synthesis_qmod_gqsp.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `849725045c514e5a07922cc715a9436dbdae0973`.

Original SHA-256: `6f051bfbd054d9511f4166d8678202841ecaa457cf2cb0d32be0c994d449f3d3`.

Approved SHA-256: `6f051bfbd054d9511f4166d8678202841ecaa457cf2cb0d32be0c994d449f3d3`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
