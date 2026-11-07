# oblivious amplitude amplification

Category: algorithms / amplitude_estimation.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/amplitude_amplification_and_estimation/oblivious_amplitude_amplification/oblivious_amplitude_amplification.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/amplitude_amplification_and_estimation/oblivious_amplitude_amplification/oblivious_amplitude_amplification.qmod'
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

Recorded resources: 4 physical qubits, 136 operations, depth 122.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_oaa.py](../../../../../../../tests/integration/test_synthesis_qmod_oaa.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `0b2a5e4d4282c2b7bc6413478f00b902752dc427`.

Original SHA-256: `2eb114b6709e3b13fea4099471c413b77b2e009d2d2235e4da35e6582d8827f8`.

Approved SHA-256: `2eb114b6709e3b13fea4099471c413b77b2e009d2d2235e4da35e6582d8827f8`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
