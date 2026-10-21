# API Change Implementation 076: Dynamic Program IR lowering

## Status

**Approved for a stacked implementation pull request by the repository owner on
2026-10-21.** This change is stacked on the typed contract introduced by
[`API_CHANGE_PROPOSAL_076_DYNAMIC_PROGRAM_IR_CONTRACT.md`](API_CHANGE_PROPOSAL_076_DYNAMIC_PROGRAM_IR_CONTRACT.md).
Its pull request targets that contract branch and must not merge before its base.

## User outcome

The user-facing dynamic workflow remains the same:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(3)
circuit.h(0)
circuit.measure(0, classical_bit=0)
circuit.reset(0)
circuit.conditional("x", 2, classical_bit=0, equals=1)

result = fq.experimental.dynamic.run_dynamic(circuit, shots=1024, seed=7)
print(result.counts)
```

This implementation changes an internal compiler boundary, not that public
execution surface. A validated `DynamicCircuit.to_ir()` may now cross the
Circuit-to-Program boundary without losing its measurement, reset, or condition
semantics.

## Exact mapping

- every input qubit begins as one linear Program IR `qubit` value;
- measurement consumes the live qubit, defines its successor and a `bool`, and
  associates the public classical-bit slot with that boolean definition;
- reset consumes the live qubit and defines its successor;
- conjunction and disjunction metadata become canonical DNF over the live boolean
  definitions associated with the referenced classical bits;
- a later measurement into the same classical-bit slot replaces the value read by
  subsequent conditions; and
- restoring a record succeeds only when lowering its attested canonical payload
  reproduces the complete Program IR function graph exactly.

The last rule makes the payload evidence rather than an unchecked source: editing
an opcode, constant, value edge, measurement result, reset, or condition reference
causes an exact round-trip mismatch.

## Compatibility

- no root export or public API snapshot changes;
- no `CircuitIR` field, serialization shape, or `IR_VERSION` change;
- no change to `DynamicCircuit`, `run_dynamic`, or result types;
- static Program IR lowering remains supported; and
- malformed or unsupported dynamic metadata remains fail-closed with typed
  diagnostics.

## Exit evidence

The stacked change is ready for review when conformance proves exact lowering and
restoration for measurement, reset, conjunction, disjunction, and classical-bit
overwrite; a live but incorrect condition reference is rejected as a graph/payload
mismatch; malformed metadata and unsupported dynamic markers are refused; the
user example runs; and the repository's contract, formatting, typing,
architecture, documentation, API-snapshot, and dynamic regression gates pass.
