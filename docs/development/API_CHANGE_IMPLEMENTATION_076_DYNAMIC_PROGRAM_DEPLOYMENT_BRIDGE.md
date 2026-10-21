# API Change Implementation 076: Dynamic Program deployment bridge

## Status

**Directed for implementation by the repository owner on 2026-10-21 after the
merge of the exact Dynamic Program IR lowering.** This is the next bounded change
after
[`API_CHANGE_IMPLEMENTATION_076_DYNAMIC_PROGRAM_IR_LOWERING.md`](API_CHANGE_IMPLEMENTATION_076_DYNAMIC_PROGRAM_IR_LOWERING.md).

## User outcome

The deployment workflow remains unchanged:

```python
import flagquantum as fq
from flagquantum.deployment import CloudBackendProfile
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2)
circuit.h(0)
circuit.measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)

backend = CloudBackendProfile(
    provider="example",
    name="dynamic-qpu",
    n_qubits=2,
    supports_openqasm=True,
    supports_dynamic_circuits=True,
    max_classical_bits=2,
)
report = fq.experimental.dynamic.assess_dynamic_backend(circuit, backend)
assert report.compatible
assert report.program_ir_verified
```

The difference is internal and fail-closed: backend assessment now requires the
public `CircuitIR` to lower to Program IR and restore exactly before a deployment
package may reach a provider.

## Boundary

The bridge applies to remote dynamic deployment, where an unsupported operation
must never become provider text by accident. Local trajectory execution retains
its current broader explicit-matrix support and is not narrowed by this change.

For deployment assessment:

1. lower `DynamicCircuit.to_ir()` to the internal Program IR;
2. require a typed record with no lowering diagnostics;
3. restore the record and require an exact graph/payload round trip;
4. report typed lowering or round-trip codes as compatibility blockers; and
5. record `program_ir_verified=true` in compatible assessment and package
   metadata.

This makes Program IR a real consumer gate in the deployment path rather than a
detached conversion utility. Existing capacity, dialect, basis-gate, routing, and
provider preflight checks remain in force.

## Compatibility

- no root export, stable signature, `CircuitIR` schema, or `IR_VERSION` change;
- no change to dynamic construction or local execution behavior;
- no provider request is sent when Program IR verification fails;
- the new compatibility fact is additive on the experimental deployment report;
  and
- existing supported OpenQASM 3 and Braket IQM paths remain accepted.

## Exit evidence

The bridge is ready when supported dynamic deployment reports and packages carry
the verification fact; an explicit-matrix operation outside Program IR receives
an actionable lowering blocker and cannot reach a provider; before- and
after-routing assessment both cross the bridge; existing dynamic deployment,
provider, routing, public API, architecture, documentation, formatting, typing,
and contract gates pass; and the user example remains executable.
