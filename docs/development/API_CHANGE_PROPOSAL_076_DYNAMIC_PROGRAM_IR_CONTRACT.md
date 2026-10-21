# API Change Proposal 076: Dynamic Program IR contract

## Status

**Approved by the repository owner on 2026-10-21 through the explicit directive
`approve API-CHANGE-076-DYNAMIC-PROGRAM-IR-CONTRACT`.**

The numeric identifier is the approval identifier supplied by the owner. The
repository also contains an independently named proposal 076 for the statevector
gradient contract; the full proposal name is therefore authoritative wherever an
unqualified number would be ambiguous.

This proposal implements the typed internal contract authorized by
[`IR_007_PHASE_4_HYBRID_DYNAMIC_SCOPE.md`](../architecture/decisions/IR_007_PHASE_4_HYBRID_DYNAMIC_SCOPE.md)
and
[`API_CHANGE_PROPOSAL_072_MULTI_LEVEL_IR_PHASE_4_ACTIVATION.md`](API_CHANGE_PROPOSAL_072_MULTI_LEVEL_IR_PHASE_4_ACTIVATION.md).
It does not change a public API, the public `CircuitIR` schema, or its version.

## User-visible behavior

User code remains unchanged. Dynamic construction and execution continue through
the candidate and experimental surfaces:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2)
circuit.h(0)
circuit.measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0, equals=1)

result = fq.experimental.dynamic.run_dynamic(circuit, shots=1024, seed=7)
print(result.counts)
```

The repository keeps the runnable form of this workflow in
`examples/local/dynamic_validation.py`. Public `CircuitIR` lowering of dynamic
metadata remains fail-closed in this change; enabling that lowering requires a
separate reviewed implementation.

## Contract admitted by this change

The internal Program IR gains three typed concepts:

1. `measure` consumes one `qubit` value and defines the successor `qubit` plus
   one `bool` value.
2. `reset` consumes one `qubit` value and defines its successor `qubit`.
3. A condition is canonical disjunctive normal form over live `bool` `ValueRef`
   definitions. Conditions may decorate gates, but never constants, measurement,
   or reset.

The verifier rejects a non-boolean comparison, a non-boolean or undefined
condition input, a non-canonical clause set, and an illegal condition target with
distinct typed diagnostics. The checked-in contract and CI gate read the same
signatures from the Program IR model so prose and implementation cannot drift.

## Compatibility boundary

- `IR_VERSION` remains `1.0`.
- `CircuitIR`, `Instruction`, and the Stable Core export snapshot do not change.
- The new types remain internal and are not exported from `flagquantum` or
  `flagquantum.core.ir`.
- Static circuit lowering and exact round trips retain their current behavior.
- Dynamic and conditional public metadata continue to be refused until the next
  implementation change can lower and restore the admitted def-use exactly.

## Non-goals

This change does not add loops, general classical computation, timing, pulses,
provider-native instructions, execution dispatch, or a second pass manager. It
does not promote dynamic execution from experimental status.

## Exit evidence

The change is complete when:

- the internal contract gate validates the typed signatures and every new
  diagnostic is both declared and raised;
- conformance constructs a measurement/reset/conditional value graph and proves
  the verifier accepts it;
- conformance proves each invalid dynamic def-use category is rejected;
- existing refusal tests prove public dynamic lowering is still inactive; and
- formatting, typing, architecture, API-snapshot, documentation, and focused
  unit gates pass.

The successor change may implement public dynamic-program lowering only after
these conditions pass.
