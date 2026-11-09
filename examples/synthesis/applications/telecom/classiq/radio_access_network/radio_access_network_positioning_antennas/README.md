# radio access network positioning antennas

Category: applications / telecom.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/telecom/radio_access_network/radio_access_network_positioning_antennas.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/telecom/radio_access_network/radio_access_network_positioning_antennas.qmod'
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

Recorded resources: 16 physical qubits, 1176 operations, depth 270.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_cscope.py](../../../../../../../tests/integration/test_synthesis_qmod_cscope.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `78b568faaa020655af03f5804054c4e6f693c082`.

Original SHA-256: `5214a49242595d1fce3e1a86a13a1e76263a480494eab232e09d942f75b4a85c`.

Approved SHA-256: `5214a49242595d1fce3e1a86a13a1e76263a480494eab232e09d942f75b4a85c`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
