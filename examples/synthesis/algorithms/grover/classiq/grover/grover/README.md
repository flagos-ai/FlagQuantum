# grover

Category: algorithms / grover.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/algorithms/search_and_optimization/grover/grover.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'algorithms/search_and_optimization/grover/grover.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "r": 2
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 7 physical qubits, 455 operations, depth 395.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_predicate_oracle.py](../../../../../../../tests/integration/test_synthesis_qmod_predicate_oracle.py)
- [validate_qmod_predicate_oracle.py](../../../../../../../tools/validate_qmod_predicate_oracle.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `a32bc67c7dfc247d92bd6b63ae538ec63f26f65a`.

Original SHA-256: `0c13b87b292de8e4abfa4bd9c9f3c5c847973a647a652419ee0eb772d87a81c6`.

Approved SHA-256: `0c13b87b292de8e4abfa4bd9c9f3c5c847973a647a652419ee0eb772d87a81c6`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
