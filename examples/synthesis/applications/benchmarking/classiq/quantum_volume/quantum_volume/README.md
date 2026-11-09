# quantum volume

Category: applications / benchmarking.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/benchmarking/quantum_volume/quantum_volume.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/benchmarking/quantum_volume/quantum_volume.qmod'
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

Recorded resources: 3 physical qubits, 93 operations, depth 61.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_operation_helpers.py](../../../../../../../tests/integration/test_synthesis_qmod_operation_helpers.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `e82274ef0ac4cff694e6c51746c94c59844b32ee`.

Original SHA-256: `23bc75bf6f7d753210c05c81859ce301c3c8ca56d78caad9c2a3b988130f8814`.

Approved SHA-256: `23bc75bf6f7d753210c05c81859ce301c3c8ca56d78caad9c2a3b988130f8814`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
