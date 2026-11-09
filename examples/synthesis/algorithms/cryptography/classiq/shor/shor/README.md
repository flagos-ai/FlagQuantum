# shor

Category: algorithms / cryptography.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/number_theory_and_cryptography/shor/shor.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/number_theory_and_cryptography/shor/shor.qmod'
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

Recorded resources: 18 physical qubits, 2108 operations, depth 1844.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_shor.py](../../../../../../../tests/integration/test_synthesis_qmod_shor.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `ec2f81f662d608b54169df1d49a1851f2152fdbd`.

Original SHA-256: `938e365f085aec5378693c5c466261ca0ccc9e8b5720979bce7da10a9e4516b0`.

Approved SHA-256: `938e365f085aec5378693c5c466261ca0ccc9e8b5720979bce7da10a9e4516b0`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
