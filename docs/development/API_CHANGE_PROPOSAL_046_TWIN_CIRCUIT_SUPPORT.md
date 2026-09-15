# API Change Proposal 046: Topology-qualified Twin circuit support

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** The user
authorized the next Twin slice after candidate-suite convergence: make the
evidence boundary state which physical interactions and circuit depth it
actually supports.

## Problem

`TwinEvidenceEnvelope` correctly binds a frozen snapshot, ordered physical
mapping, operation names, instruction-count limit, exact circuit identities and
statistical bounds. Its structural check intentionally has no physical topology
or circuit-depth input. Consequently, an application cannot distinguish a
tested connected path from an untested interaction between two mapped qubits.

Changing the frozen evidence-envelope v1 fields would break serialized evidence.
Inferring topology from calibration is also unsafe: device availability is not
the same fact as prospective validation coverage.

## Decision

- Keep `TwinEvidenceEnvelope` and its v1 schema unchanged.
- Add immutable `TwinCircuitSupport`, which contains one evidence envelope, an
  explicit set of directed physical couplers and a maximum circuit depth.
- Add `supports(...)` and `unsupported_reasons(...)` for offline inspection.
- Add `evidence_report(twin, circuit)`, which preserves the existing evidence
  result only inside both boundaries and otherwise returns `out_of_scope`
  without an error bound.
- Add canonical `dump_circuit_support(...)` and `load_circuit_support(...)`.
- Treat multi-qubit operations with arity greater than two as outside this first
  support contract.
- Perform no provider access, task submission, topology discovery, evidence
  creation, model update, routing decision or model promotion.

## Public API

```python
import flagquantum as fq

twin = fq.twin.load_twin("qpu-twin.json")
evidence = fq.twin.load_evidence("twin-evidence.json")

support = fq.twin.TwinCircuitSupport(
    evidence=evidence,
    directed_couplers=((20, 27), (27, 20), (27, 34), (34, 27)),
    maximum_circuit_depth=8,
)

# Twin reports predict the full computational-basis (Z-basis) distribution.
# TwinExperiment.prepare() binds the corresponding all-qubit hardware readout.
circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
report = support.evidence_report(twin, circuit)

print(report.status)
print(report.tv_error_bound)
print(report.confidence_level)
```

The Twin and evidence files are produced by the complete executable workflow in
`examples/remote/quafu_twin_evidence.py`. Twin predictions and reports concern
the full computational-basis (Z-basis) output distribution. The prospective
hardware workflow binds the matching all-qubit measurement when
`TwinExperiment.prepare()` emits validation OpenQASM. This support check is
entirely offline and performs no live measurement or provider call.

## Interpretation

The new object narrows evidence; it never broadens it. A device coupler is not
supported merely because it exists in provider calibration. Both physical
direction and circuit depth must be declared from a prospectively validated
workload. The result remains an error bound for classical measurement
distributions, not quantum-state fidelity or arbitrary-circuit accuracy.

## Compatibility

The change is additive. Existing Twin classes, signatures, literal values and
serialized schemas retain their meanings. The existing evidence artifact loads
unchanged and can be wrapped by the new support boundary.

## Acceptance

- A circuit on declared directed couplers and within the depth limit preserves
  its existing evidence report.
- An undeclared direction, non-edge interaction, excessive depth, operation
  outside the evidence structure or operation of arity greater than two fails
  closed without an error bound.
- Invalid, duplicate or out-of-mapping couplers are rejected.
- Persistence is deterministic, private, idempotent for identical content and
  refuses destructive replacement.
- Loading and assessment perform no provider operation.
- The API contract, guide, release notes, scenario tests and executable example
  describe the same behavior.
