# whitebox fuzzing

Category: applications / cybersecurity.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/cybersecurity/whitebox_fuzzing/whitebox_fuzzing.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/cybersecurity/whitebox_fuzzing/whitebox_fuzzing.qmod'
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

Recorded resources: 18 physical qubits, 1186 operations, depth 840.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_decisions.py](../../../../../../../tests/integration/test_synthesis_qmod_decisions.py)
- [validate_qmod_decision_predicates.py](../../../../../../../tools/validate_qmod_decision_predicates.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `e3ce4a5d400dedd1de83cbdaf2b3ff1c0aa96662`.

Original SHA-256: `837a0d8b9863f29a6aaf1af4ec7670af42149187abaca1cde5e06545209cf98d`.

Approved SHA-256: `837a0d8b9863f29a6aaf1af4ec7670af42149187abaca1cde5e06545209cf98d`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
