# hhl workshop

Category: tutorials / workshops.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/workshops/hhl_workshop/hhl_workshop.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/workshops/hhl_workshop/hhl_workshop.qmod'
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

Recorded resources: 7 physical qubits, 417 operations, depth 400.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_hhl.py](../../../../../../../tests/integration/test_synthesis_qmod_hhl.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `2ab462b0a7eaa806bb84281f9ce3d13f4661a6cc`.

Original SHA-256: `1969cd5e613a8da44f843b4e04b6c0cca4aa3796bb987d34ab6251170fd44edd`.

Approved SHA-256: `1969cd5e613a8da44f843b4e04b6c0cca4aa3796bb987d34ab6251170fd44edd`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
