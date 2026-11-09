# laplacian BE Nx4 Ny4

Category: algorithms / block_encoding.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/community/paper_implementation_project/block_encoding/laplacian_BE_Nx4_Ny4.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'community/paper_implementation_project/block_encoding/laplacian_BE_Nx4_Ny4.qmod'
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

Recorded resources: 12 physical qubits, 139 operations, depth 113.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_laplacian_views.py](../../../../../../../../tests/integration/test_synthesis_qmod_laplacian_views.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `ef4fe785f0261938bdcd2fda32565a578abbde68`.

Original SHA-256: `84a29820a360508a798b9c2fcb13f5696522b7501869aec3831604f778daa737`.

Approved SHA-256: `84a29820a360508a798b9c2fcb13f5696522b7501869aec3831604f778daa737`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
