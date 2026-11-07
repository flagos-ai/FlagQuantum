# vehicle routing problem

Category: applications / logistics.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/applications/logistics/vehicle_routing_problem/vehicle_routing_problem.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'applications/logistics/vehicle_routing_problem/vehicle_routing_problem.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "params": [
    0.013,
    0.026,
    0.039,
    0.052,
    0.065,
    0.078,
    0.091,
    0.104,
    0.11699999999999999,
    0.13,
    0.143,
    0.156,
    0.16899999999999998,
    0.182,
    0.19499999999999998,
    0.208,
    0.221,
    0.23399999999999999,
    0.247,
    0.26,
    0.27299999999999996,
    0.286,
    0.299,
    0.312
  ]
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 16 physical qubits, 1249 operations, depth 176.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_vehicle_routing.py](../../../../../../../tests/integration/test_synthesis_qmod_vehicle_routing.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `145478f7bf78ed174907eb2e36417c2960b89704`.

Original SHA-256: `b88589a757a939a36a20b14ce9c062653dc595d92ced31343904f000ae1002ec`.

Approved SHA-256: `b88589a757a939a36a20b14ce9c062653dc595d92ced31343904f000ae1002ec`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
