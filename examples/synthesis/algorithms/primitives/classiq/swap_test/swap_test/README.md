# swap test

Category: algorithms / primitives.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/quantum_primitives/swap_test/swap_test.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/quantum_primitives/swap_test/swap_test.qmod' --revision approved
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

Recorded resources: 7 physical qubits, 37 operations, depth 16.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Approved numerical revision

Normalize both input states while preserving their relative amplitudes.

[Approved source](comparisons/accepted.qmod)

The original remains byte-identical and is expected to be refused.
Run the command without `--revision approved` to inspect that refusal.

Recorded original diagnostic:

```text
construction.register.amplitudes: ../synth-proposal/classiq-qmod-audit/qmod/algorithms/quantum_primitives/swap_test/swap_test.qmod:15:2: preparation requires a normalized finite real vector matching the target and a nonnegative finite bound
```

## Existing validation sources

- [test_synthesis_qmod_swap_test.py](../../../../../../../tests/integration/test_synthesis_qmod_swap_test.py)
- [validate_qmod_swap_test.py](../../../../../../../tools/validate_qmod_swap_test.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `c686d373a0c2d12be1521b0e884e9413c48299af`.

Original SHA-256: `a5ac91c85f0868e14164385710af056a2c0e84b73b4eb1ccd5bc449d421b99f2`.

Approved SHA-256: `86c639106f27dddecc0135ff36287c3fd1c1b96e98ec84cc7a1d5533e11fcc00`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
