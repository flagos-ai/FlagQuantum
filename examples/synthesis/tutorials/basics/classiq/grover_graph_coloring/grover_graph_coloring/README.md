# grover graph coloring

Category: tutorials / basics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/basic_tutorials/grover_graph_coloring/grover_graph_coloring.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/basic_tutorials/grover_graph_coloring/grover_graph_coloring.qmod'
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

Recorded resources: 16 physical qubits, 271 operations, depth 190.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_decisions.py](../../../../../../../tests/integration/test_synthesis_qmod_decisions.py)
- [validate_qmod_decision_predicates.py](../../../../../../../tools/validate_qmod_decision_predicates.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `4ca89b253622cd229e922f36b617b4420382ec50`.

Original SHA-256: `63f26e3c2bf50b9c0e5b0f3c5b7c83205118a0998e4f201969730f307023346a`.

Approved SHA-256: `63f26e3c2bf50b9c0e5b0f3c5b7c83205118a0998e4f201969730f307023346a`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
