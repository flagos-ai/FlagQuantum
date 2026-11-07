# linear pauli rotations

Category: tutorials / library.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/linear_pauli_rotations/linear_pauli_rotations.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/linear_pauli_rotations/linear_pauli_rotations.qmod'
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

Recorded resources: 9 physical qubits, 21 operations, depth 9.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_linear_pauli.py](../../../../../../../../../tests/integration/test_synthesis_qmod_linear_pauli.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `62dab91a568855021a78930b2bb50011121bdf90`.

Original SHA-256: `0ca012b95dedab4226fe65e97ae62cfcb890633a78db4882d0bfddac29f584bb`.

Approved SHA-256: `0ca012b95dedab4226fe65e97ae62cfcb890633a78db4882d0bfddac29f584bb`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
