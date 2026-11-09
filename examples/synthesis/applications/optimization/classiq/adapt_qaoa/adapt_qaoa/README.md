# adapt qaoa

Category: applications / optimization.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/optimization/adapt_qaoa/adapt_qaoa.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/optimization/adapt_qaoa/adapt_qaoa.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "params": [
    0.1,
    0.2
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 8 physical qubits, 33 operations, depth 10.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_qaoa_phase.py](../../../../../../../tests/integration/test_synthesis_qmod_qaoa_phase.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `a54cf3838bc16711a186884559042fab41d175d9`.

Original SHA-256: `e1c3a471e0b31b501c3f110031d43738446d2befff8eae49948875f0b24accd0`.

Approved SHA-256: `e1c3a471e0b31b501c3f110031d43738446d2befff8eae49948875f0b24accd0`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
