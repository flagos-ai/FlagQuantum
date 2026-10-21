# API Change 075: Dynamic deployment preflight

## Status

Implemented as the second pull request in the dynamic-circuit stack. Its base is
`feat/dynamic-circuit-v1`, the branch for API Change 074. This change does not
promote a public API and does not claim hardware execution support.

## User workflow

Users can inspect compatibility without submitting a task:

```python
import flagquantum as fq
from flagquantum.deployment import CloudBackendProfile
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2)
circuit.measure(0, classical_bit=0)
circuit.conditional("h", 1, classical_bit=0)

backend = CloudBackendProfile(
    provider="example",
    name="limited-dynamic-qpu",
    n_qubits=2,
    basis_gates=("x",),
    supports_dynamic_circuits=True,
    max_classical_bits=2,
)
report = fq.experimental.dynamic.assess_dynamic_backend(circuit, backend)

assert not report.compatible
assert report.blockers == ("conditional_gate_not_in_backend_basis:h",)
```

The executable form is
`python -m examples.local.dynamic_deployment_preflight`.

## Boundary strengthened

- Backend capacity is read from `n_qubits` without entering the deprecated
  `n_wires` compatibility property.
- A conditioned operation is rejected when a non-empty backend basis does not
  contain its opcode on the standard OpenQASM 3 path. Vendor dialects continue
  to validate against their own lowering, where a user operation may map to a
  differently named native operation.
- Duplicate blockers are removed while preserving discovery order.
- Deployment packaging raises `flagquantum.errors.CapabilityError` with the
  stage and exact blockers before provider submission.
- The provider is not called after a failed preflight.

An empty backend basis still means that the backend has not supplied a basis
constraint; this change does not reinterpret missing provider data as a complete
gate declaration. Whether a backend supports a gate only statically or also as a
conditioned operation remains unexpressible in `CloudBackendProfile` and is not
claimed here.

## Compatibility

No name, signature, serialized schema, root export, or candidate contract is
changed. The exception for an incompatible dynamic deployment changes from a
bare `RuntimeError` to the stable backend-neutral `CapabilityError` category.
The deployment helpers remain outside the protected root API; the experimental
preflight result and its existing signature are unchanged.

## Verification

The focused tests cover a successful preflight, capacity and classical-register
refusals, a conditioned-gate basis refusal, and proof that an incompatible
package never reaches `provider.run`. Repository formatting, lint, type, API
contract, documentation, and language gates are run before the branch is pushed.
