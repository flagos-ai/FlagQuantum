# task scheduling problem

Category: applications / logistics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/logistics/task_scheduling_problem/task_scheduling_problem.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/logistics/task_scheduling_problem/task_scheduling_problem.qmod'
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

Recorded resources: 24 physical qubits, 2004 operations, depth 458.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_cscope.py](../../../../../../../tests/integration/test_synthesis_qmod_cscope.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `b605c29e54863d145fed2ecf7b66af2759b4952e`.

Original SHA-256: `e2f05ede7be9bb9520aff145db7cf2f67124d3b6d141f1e1418e23cb12a34eb2`.

Approved SHA-256: `e2f05ede7be9bb9520aff145db7cf2f67124d3b6d141f1e1418e23cb12a34eb2`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
