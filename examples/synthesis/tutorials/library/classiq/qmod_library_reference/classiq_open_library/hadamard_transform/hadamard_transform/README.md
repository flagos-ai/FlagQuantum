# hadamard transform

Category: tutorials / library.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/hadamard_transform/hadamard_transform.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/hadamard_transform/hadamard_transform.qmod'
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

Recorded resources: 3 physical qubits, 3 operations, depth 1.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_array_hadamard.py](../../../../../../../../../tests/integration/test_synthesis_qmod_array_hadamard.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `e8d222ff1e61f792e1ae875a839c05ebfec47e46`.

Original SHA-256: `22e4cc216165feb902b6319eaf0ca9cbb8f90e42b23e4a27f1e0d440676e8ade`.

Approved SHA-256: `22e4cc216165feb902b6319eaf0ca9cbb8f90e42b23e4a27f1e0d440676e8ade`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
