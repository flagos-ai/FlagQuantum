# randomized benchmarking

Category: applications / benchmarking.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/benchmarking/randomized_benchmarking/randomized_benchmarking.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/benchmarking/randomized_benchmarking/randomized_benchmarking.qmod'
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

Recorded resources: 1 physical qubits, 0 operations, depth 0.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_gate_inverse.py](../../../../../../../tests/integration/test_synthesis_qmod_gate_inverse.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `4b3b3a46668181ac59f525e19421f94bdb57fddb`.

Original SHA-256: `0736cf0c45627639f7fa7a300702f166fa8cf459baf09708fffdbe5e75c486f5`.

Approved SHA-256: `0736cf0c45627639f7fa7a300702f166fa8cf459baf09708fffdbe5e75c486f5`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
