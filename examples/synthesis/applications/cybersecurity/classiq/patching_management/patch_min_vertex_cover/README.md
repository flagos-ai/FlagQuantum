# patch min vertex cover

Category: applications / cybersecurity.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/cybersecurity/patching_management/patch_min_vertex_cover.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/cybersecurity/patching_management/patch_min_vertex_cover.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "params": [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 8 physical qubits, 103 operations, depth 31.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_records.py](../../../../../../../tests/integration/test_synthesis_qmod_records.py)
- [validate_qmod_records.py](../../../../../../../tools/validate_qmod_records.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `068de18b7b1a31e64458a39112399d2dac87ad97`.

Original SHA-256: `a2b851ead5727fe7c121c50ea7e91966e3c6d65368b3a98633d52b1a3c62b735`.

Approved SHA-256: `a2b851ead5727fe7c121c50ea7e91966e3c6d65368b3a98633d52b1a3c62b735`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
