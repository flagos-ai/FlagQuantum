# solving qlsp with aqc

Category: algorithms / quantum_linear_solvers.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/quantum_linear_solvers/adiabatic_linear_solvers/solving_qlsp_with_aqc.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/quantum_linear_solvers/adiabatic_linear_solvers/solving_qlsp_with_aqc.qmod'
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

Recorded resources: 3 physical qubits, 4542 operations, depth 4053.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_aqc_qlsp.py](../../../../../../../tests/integration/test_synthesis_qmod_aqc_qlsp.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `63e536947861bae5e9aed3fa0ab7d503710054c8`.

Original SHA-256: `b29fb53678bca0aa7bcf2c0baf392cac8b9c427ccf2a73be00bce36e86b200a1`.

Approved SHA-256: `b29fb53678bca0aa7bcf2c0baf392cac8b9c427ccf2a73be00bce36e86b200a1`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
