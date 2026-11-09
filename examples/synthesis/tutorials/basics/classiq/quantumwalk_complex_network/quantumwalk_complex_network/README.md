# quantumwalk complex network

Category: tutorials / basics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/basic_tutorials/quantumwalk_complex_network/quantumwalk_complex_network.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/basic_tutorials/quantumwalk_complex_network/quantumwalk_complex_network.qmod'
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

Recorded resources: 4 physical qubits, 1718 operations, depth 1527.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_network_walk.py](../../../../../../../tests/integration/test_synthesis_qmod_network_walk.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `bf33fbac91258d17a4b7ce8f3fccf36c5ef449bf`.

Original SHA-256: `189900a2706e62fd41a7ce7043985ea7d9022a56edfe26f333173db6bf6bc03e`.

Approved SHA-256: `189900a2706e62fd41a7ce7043985ea7d9022a56edfe26f333173db6bf6bc03e`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
