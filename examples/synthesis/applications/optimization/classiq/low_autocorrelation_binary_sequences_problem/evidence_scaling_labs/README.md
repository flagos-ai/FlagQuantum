# evidence scaling labs

Category: applications / optimization.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/optimization/low_autocorrelation_binary_sequences_problem/evidence_scaling_labs.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/optimization/low_autocorrelation_binary_sequences_problem/evidence_scaling_labs.qmod'
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

Recorded resources: 13 physical qubits, 1922 operations, depth 1484.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_cscope.py](../../../../../../../tests/integration/test_synthesis_qmod_cscope.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `2d9bf8493ad546b495cf53a4541e0d4232b5f9c8`.

Original SHA-256: `ef09e4c4cbdef9b75d6731725869d66c6281d423eca46069c8bab8325a3d5a70`.

Approved SHA-256: `ef09e4c4cbdef9b75d6731725869d66c6281d423eca46069c8bab8325a3d5a70`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
