# hardware aware mcx

Category: tutorials / techniques.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/technology_demonstrations/hardware_aware_mcx/hardware_aware_mcx.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/technology_demonstrations/hardware_aware_mcx/hardware_aware_mcx.qmod'
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

Recorded resources: 22 physical qubits, 41 operations, depth 41.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_array_control.py](../../../../../../../tests/integration/test_synthesis_qmod_array_control.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `75c0e84394ef5370d7210d6a68c2b31f01537e20`.

Original SHA-256: `68af2bf554d710843078bce0936a0ed12b1ba1020981aac4d6ef8465851ebc3d`.

Approved SHA-256: `68af2bf554d710843078bce0936a0ed12b1ba1020981aac4d6ef8465851ebc3d`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
