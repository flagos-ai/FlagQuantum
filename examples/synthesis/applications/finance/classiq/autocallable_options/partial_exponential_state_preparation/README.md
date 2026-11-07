# partial exponential state preparation

Category: applications / finance.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/finance/autocallable_options/partial_exponential_state_preparation.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/finance/autocallable_options/partial_exponential_state_preparation.qmod' --revision approved
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

Recorded resources: 8 physical qubits, 2265 operations, depth 1921.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Approved numerical revision

Use the marked overlap of the existing exponential preparation.

[Approved source](comparisons/accepted.qmod)

The original remains byte-identical and is expected to be refused.
Run the command without `--revision approved` to inspect that refusal.

Recorded original diagnostic:

```text
qmod.amplification.amplitude: ../synth-proposal/classiq-qmod-audit/qmod/applications/finance/autocallable_options/partial_exponential_state_preparation.qmod:7:33: supplied amplitude 0.99662542476596105 disagrees with proven amplitude 0.99795457844373736; original parameters are not corrected
```

## Existing validation sources

- [test_synthesis_qmod_exact_amplification.py](../../../../../../../tests/integration/test_synthesis_qmod_exact_amplification.py)
- [validate_qmod_exact_amplification.py](../../../../../../../tools/validate_qmod_exact_amplification.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `fc5fd712a1853369318db7d14f3caac029de5209`.

Original SHA-256: `da0b25a150ed9bf74852d131b21f7324a28eac354b9e581b846a7a749d361cc2`.

Approved SHA-256: `28850eafc1fb5267ddfd87e5ba7e93bb10e9977da73141c6829fa6269a709e6e`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
