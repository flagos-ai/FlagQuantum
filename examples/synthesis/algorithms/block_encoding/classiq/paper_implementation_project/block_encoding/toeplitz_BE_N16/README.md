# toeplitz BE N16

Category: algorithms / block_encoding.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/community/paper_implementation_project/block_encoding/toeplitz_BE_N16.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'community/paper_implementation_project/block_encoding/toeplitz_BE_N16.qmod'
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

Recorded resources: 11 physical qubits, 151 operations, depth 128.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_toeplitz.py](../../../../../../../../tests/integration/test_synthesis_qmod_toeplitz.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `431b605942611984114640d00298a8d5821691fc`.

Original SHA-256: `5b9038970ff95355496fdde1c6cdd5cbc1130249d45fff49b5ac90631ab94796`.

Approved SHA-256: `5b9038970ff95355496fdde1c6cdd5cbc1130249d45fff49b5ac90631ab94796`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
