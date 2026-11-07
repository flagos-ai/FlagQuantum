# link monitoring

Category: applications / cybersecurity.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/cybersecurity/link_monitoring/link_monitoring.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/cybersecurity/link_monitoring/link_monitoring.qmod'
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

Recorded resources: 12 physical qubits, 24 operations, depth 2.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_cscope.py](../../../../../../../tests/integration/test_synthesis_qmod_cscope.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `765ba91742a18f541dcac74da4b9d24fdc01453d`.

Original SHA-256: `387eea491283597dc063d0d759828acf2968776658a04c831f3f59ce076eef42`.

Approved SHA-256: `387eea491283597dc063d0d759828acf2968776658a04c831f3f59ce076eef42`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
