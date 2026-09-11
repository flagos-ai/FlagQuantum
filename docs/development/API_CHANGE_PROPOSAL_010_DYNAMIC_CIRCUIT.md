# API Change Proposal 010: Dynamic Circuit Construction Contract

## Status

**Frozen by API owner — candidate stable contract approved and frozen.**

- Candidate stable namespace: `flagquantum.dynamic`.
- Candidate stable name: `DynamicCircuit` only.
- Root API changes: none.
- Machine contract: `contracts/dynamic-circuit-v1-candidate.json`.
- `run_dynamic`, providers, dialects, and native results remain experimental.
- Approval: `approve 008-010`, explicitly issued by the API owner on 2026-09-01.
- The aggregate API candidate inventory bound by review-packet hashes is unchanged.

## Decision

Stabilize dynamic program expression without prematurely stabilizing execution.
Users build programs through a separate namespace:

```python
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2)
circuit.h(0)
circuit.measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0, equals=1)
```

`measure`, `reset`, single-bit or multi-bit conjunction conditions, and CircuitIR v1
encoding enter the candidate contract. Static gates reuse the frozen `Circuit`
operator schema.

## Why DynamicExecutionResult Is Not Stable

Local trajectory execution currently returns `final_states`, while Qiskit Aer
returns an empty tensor. `provider_metadata` and `statistics` also contain
implementation-specific fields. Freezing them now would make backend differences
permanent user contracts.

The stable result remains `fq.ExecutionResult`. Experimental executors temporarily
return `fq.experimental.dynamic.DynamicExecutionResult`, explicitly projected
through `to_execution_result()`. Before stabilizing `run_dynamic`, define the
measurement, runtime, provenance, and mid-circuit data semantics of returning
`ExecutionResult` directly.

## Changes in This Round

1. Align `DynamicCircuit.state` with its parent: `state(*, refresh=False)`.
2. Invalid wires, classical bits, and conditions consistently raise `ValidationError`.
3. `state()` with dynamic instructions consistently raises `CapabilityError`.
4. Adding dynamic instructions fully clears compilation and kernel caches.
5. Move `DynamicCircuit` from experimental feature exports into its separate
   candidate stable namespace.

## Acceptance Criteria

- [x] `flagquantum.dynamic.__all__ == ("DynamicCircuit",)`.
- [x] Machine contracts protect constructor and dynamic-method signatures.
- [x] CircuitIR round-trip preserves measurement, reset, and condition metadata.
- [x] Errors use the stable hierarchy.
- [x] Stable root gains no names.
- [x] Execution, deployment, and adapter implementations are not mislabeled stable.
- [x] API owner approved the DynamicCircuit construction contract freeze.
