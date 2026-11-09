# tensor hypercontraction

Category: applications / chemistry.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/chemistry/tensor_hypercontraction/tensor_hypercontraction.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/chemistry/tensor_hypercontraction/tensor_hypercontraction.qmod'
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

Recorded resources: 83 physical qubits, 43 operations, depth 34.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_tensor_zero_walk.py](../../../../../../../tests/integration/test_synthesis_qmod_tensor_zero_walk.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `ec90173c3669c622d38d09f17bcb67dc8b9cd735`.

Original SHA-256: `e10515cc546fd5898db4c369a273604f318506cb193b2a0128d2f951805c4b19`.

Approved SHA-256: `e10515cc546fd5898db4c369a273604f318506cb193b2a0128d2f951805c4b19`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
