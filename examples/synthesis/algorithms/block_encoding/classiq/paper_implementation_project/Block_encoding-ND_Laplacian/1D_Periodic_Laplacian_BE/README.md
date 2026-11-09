# 1D Periodic Laplacian BE

Category: algorithms / block_encoding.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/community/paper_implementation_project/Block_encoding-ND_Laplacian/1D_Periodic_Laplacian_BE.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'community/paper_implementation_project/Block_encoding-ND_Laplacian/1D_Periodic_Laplacian_BE.qmod'
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

Recorded resources: 10 physical qubits, 27 operations, depth 25.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_laplacian.py](../../../../../../../../tests/integration/test_synthesis_qmod_laplacian.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `828f035741e109e98b27338cf6654106e7715da1`.

Original SHA-256: `d693b2da6bac932c51549f2222901f9eb366ff43216ad56468b93d4630fbad41`.

Approved SHA-256: `d693b2da6bac932c51549f2222901f9eb366ff43216ad56468b93d4630fbad41`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
