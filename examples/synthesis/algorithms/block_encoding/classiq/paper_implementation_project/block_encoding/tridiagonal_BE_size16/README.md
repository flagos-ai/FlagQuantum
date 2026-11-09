# tridiagonal BE size16

Category: algorithms / block_encoding.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/community/paper_implementation_project/block_encoding/tridiagonal_BE_size16.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'community/paper_implementation_project/block_encoding/tridiagonal_BE_size16.qmod' --revision approved
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

Original result: `frontend_refused`. Approved result: `execution_smoke_passed`.

Recorded resources: 10 physical qubits, 17250 operations, depth 13899.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Approved numerical revision

Normalize the three equal nonzero selector amplitudes.

[Approved source](comparisons/accepted.qmod)

The original remains byte-identical and is expected to be refused.
Run the command without `--revision approved` to inspect that refusal.

Recorded original diagnostic:

```text
qmod.amplitudes: ../synth-proposal/classiq-qmod-audit/qmod/community/paper_implementation_project/block_encoding/tridiagonal_BE_size16.qmod:2:30: inplace preparation requires a normalized finite real vector and a nonnegative finite bound
```

## Existing validation sources

- [test_synthesis_qmod_block_encoding.py](../../../../../../../../tests/integration/test_synthesis_qmod_block_encoding.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `b9020adb43b64e9b67b86300158c5447e21c6693`.

Original SHA-256: `0ba8f8e3d2d2bfdb356c8b2b9b53fb57101170b73b6c2a0717a57c1f62603a71`.

Approved SHA-256: `d05c8686edab7492fd781244d856edcd1730e2bee705b2e54de957a322ca4802`.

[Classiq MIT license](../../../../../../CLASSIQ_LICENSE.txt)
