# traveling salesman problem

Category: applications / logistics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/logistics/traveling_salesman_problem/traveling_salesman_problem.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/logistics/traveling_salesman_problem/traveling_salesman_problem.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "params": [
    0.003,
    -0.006,
    0.009000000000000001,
    -0.012,
    0.015,
    -0.018000000000000002,
    0.021,
    -0.024,
    0.027,
    -0.03,
    0.033,
    -0.036000000000000004,
    0.039,
    -0.042,
    0.045,
    -0.048
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 16 physical qubits, 929 operations, depth 157.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_phase_factors.py](../../../../../../../tests/integration/test_synthesis_qmod_phase_factors.py)
- [validate_qmod_phase_factors.py](../../../../../../../tools/validate_qmod_phase_factors.py)

Circuit-based validation source links: 2.

## Provenance

Original Git blob: `ff4c071d90018bdf75ccfc026d154534c3e5af7a`.

Original SHA-256: `e67a949ccb921de1d95f8d60600ccaf3b4acd0ab800287f94b5a723259e931b6`.

Approved SHA-256: `e67a949ccb921de1d95f8d60600ccaf3b4acd0ab800287f94b5a723259e931b6`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
