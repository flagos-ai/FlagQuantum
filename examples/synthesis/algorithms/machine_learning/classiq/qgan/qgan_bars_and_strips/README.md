# qgan bars and strips

Category: algorithms / machine_learning.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/QML/qgan/qgan_bars_and_strips.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/QML/qgan/qgan_bars_and_strips.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "input_vec": [
    0.2,
    -0.3,
    0.4,
    -0.5
  ],
  "weight": [
    0.1,
    0.2,
    0.30000000000000004,
    0.4,
    0.5,
    0.6000000000000001,
    0.7000000000000001,
    0.8
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 4 physical qubits, 11 operations, depth 5.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_rzz_views.py](../../../../../../../tests/integration/test_synthesis_qmod_rzz_views.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `3b30db5c0cf90075e8a0fa9b0307a57e6903b9b6`.

Original SHA-256: `00008898f2081de7f9ccf5109a029023df940b048752b76f49f6bf1cea7657d8`.

Approved SHA-256: `00008898f2081de7f9ccf5109a029023df940b048752b76f49f6bf1cea7657d8`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
