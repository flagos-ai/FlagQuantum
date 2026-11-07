# grover operator

Category: tutorials / library.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/functions/qmod_library_reference/classiq_open_library/grover_operator/grover_operator.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'functions/qmod_library_reference/classiq_open_library/grover_operator/grover_operator.qmod'
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

Recorded resources: 5 physical qubits, 142 operations, depth 118.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_predicate_oracle.py](../../../../../../../../../tests/integration/test_synthesis_qmod_predicate_oracle.py)
- [validate_qmod_predicate_oracle.py](../../../../../../../../../tools/validate_qmod_predicate_oracle.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `ece299a0cc8a1643fc8c721c86495411f5bea354`.

Original SHA-256: `e683a404b90804f257a73f3aa7b9e281d757098885c08c951d7ea08dfce677f6`.

Approved SHA-256: `e683a404b90804f257a73f3aa7b9e281d757098885c08c951d7ea08dfce677f6`.

[Classiq MIT license](../../../../../../../CLASSIQ_LICENSE.txt)
