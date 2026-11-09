# checkerboard BE N8

Category: algorithms / block_encoding.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/community/paper_implementation_project/block_encoding/checkerboard_BE_N8.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'community/paper_implementation_project/block_encoding/checkerboard_BE_N8.qmod'
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

Recorded resources: 7 physical qubits, 22 operations, depth 11.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_block_encoding.py](../../../../../../../../tests/integration/test_synthesis_qmod_block_encoding.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `0152906ff6404e7bdff03f42a0e49f8b03c86cdc`.

Original SHA-256: `91874079574963311cf224000906697430d4dd53bd9cfbfbe8346aac0a5f26dd`.

Approved SHA-256: `91874079574963311cf224000906697430d4dd53bd9cfbfbe8346aac0a5f26dd`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
