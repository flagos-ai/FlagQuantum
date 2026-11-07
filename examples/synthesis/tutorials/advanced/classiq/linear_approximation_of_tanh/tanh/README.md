# tanh

Category: tutorials / advanced.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/advanced_tutorials/linear_approximation_of_tanh/tanh.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/advanced_tutorials/linear_approximation_of_tanh/tanh.qmod'
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

Recorded resources: 10 physical qubits, 43 operations, depth 36.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_tanh.py](../../../../../../../tests/integration/test_synthesis_qmod_tanh.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `f02188eb157d0d04c3f627c561ca033480e9e484`.

Original SHA-256: `dd18b55bd39c68c61326df9f2df98e597a720d5acb0e59f688f05d720840d27a`.

Approved SHA-256: `dd18b55bd39c68c61326df9f2df98e597a720d5acb0e59f688f05d720840d27a`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
