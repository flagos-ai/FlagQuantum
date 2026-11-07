# quantum walk circle

Category: tutorials / advanced.

[Original Classiq source](https://raw.githubusercontent.com/Classiq/classiq-library/2bfd835e73b6882fcc515a725098b04766c238e7/tutorials/advanced_tutorials/discrete_quantum_walk/quantum_walk_circle.qmod) at `2bfd835e73b6882fcc515a725098b04766c238e7`.

## Implementations

| Form | Availability |
| --- | --- |
| Qmod original | [source.qmod](comparisons/source.qmod) |
| Native Python qfunc | Not yet provided for this exact original |
| Circuit comparison | See the validation source links below |

## Run

Run from the repository root with FlagQuantum installed.

```sh
python -m examples.synthesis.run_qmod 'tutorials/advanced_tutorials/discrete_quantum_walk/quantum_walk_circle.qmod'
```

The shared runner binds the recorded classical inputs and selects the
recorded exact CPU execution profile. It prints measured resources and
execution validation, using the same audit implementation as corpus acceptance.

Classical bindings used by this entry:

```json
{
  "t": 2
}
```

## Verification

Recorded implementation commit: `7675271cb758cd700aa31962b004d1f87b479518`.

Original result: `execution_smoke_passed`. Approved result: `execution_smoke_passed`.

Recorded resources: 12 physical qubits, 158 operations, depth 152.

Batch execution checks successful exact execution and basic result validity.
Linked tests and validators document their own input bindings, full-state
references, global-phase and workspace checks; inspect their individual scopes.

## Existing validation sources

- [test_synthesis_qmod_circle_walk.py](../../../../../../../tests/integration/test_synthesis_qmod_circle_walk.py)

Circuit-based validation source links: 1.

## Provenance

Original Git blob: `753125e0da1c819cd57a2e43030f53498b421880`.

Original SHA-256: `b54d15e1ab729d3f68b5642a4af3880ba58f40f3e22f1c3460e5d4b58bdc9b3e`.

Approved SHA-256: `b54d15e1ab729d3f68b5642a4af3880ba58f40f3e22f1c3460e5d4b58bdc9b3e`.

[Classiq MIT license](../../../../../CLASSIQ_LICENSE.txt)
