# qmci pi estimation

Category: tutorials / basics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/basic_tutorials/qmci_pi_estimation/qmci_pi_estimation.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/basic_tutorials/qmci_pi_estimation/qmci_pi_estimation.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "k": 1
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 3 physical qubits, 15 operations, depth 11.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_amplification_iterations.py](../../../../../../../tests/integration/test_synthesis_qmod_amplification_iterations.py)
- [validate_qmod_iterations.py](../../../../../../../tools/validate_qmod_iterations.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `0c2d3702afdd1ce7cf972efb7e909f689196d6e8`.

Original SHA-256: `780b1a90d4aa5d7348a457a28095dccd4bc221d948ec478c1f9f1f8c691c905d`.

Approved SHA-256: `780b1a90d4aa5d7348a457a28095dccd4bc221d948ec478c1f9f1f8c691c905d`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
