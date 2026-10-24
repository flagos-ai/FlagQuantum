# API Change Implementation 086: Dynamic remote requirement/snapshot bridge

## Status

**Approved for implementation on 2026-10-23. Stable Runtime routing and provider
submission remain unauthorized.**

The API owner authorized this stage with:

```text
approve API-CHANGE-086-DYNAMIC-REMOTE-REQUIREMENT-SNAPSHOT-BRIDGE
```

## User outcome

The current credential-free user workflow remains unchanged:

```python
import flagquantum as fq
from flagquantum.deployment import CloudBackendProfile
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).measure(0, classical_bit=0)
circuit.conditional("h", 1, classical_bit=0)

backend = CloudBackendProfile(
    provider="example",
    name="limited-dynamic-qpu",
    n_qubits=2,
    basis_gates=("x",),
    supports_dynamic_circuits=True,
)
report = fq.experimental.dynamic.assess_dynamic_backend(circuit, backend)

assert not report.compatible
assert report.blockers == ("conditional_gate_not_in_backend_basis:h",)
```

The new bridge is intentionally internal. It gives the later stable preflight a
canonical, serializable answer without asking users to construct requirements,
snapshots, fingerprints, or evidence records.

## Implementation

`flagquantum.runtime.dynamic.remote_capabilities` now:

1. derives the nine-field dynamic requirement vector from validated, ordered
   Circuit IR and requires an exact Program IR round trip;
2. stores dynamic requirements in the registered
   `org.flagquantum.dynamic-remote-v1` extension of Core `RequirementSet`;
3. projects an explicitly supplied `DynamicFeatureSet` and device-scoped
   evidence into Core `TargetCapabilitySnapshot`;
4. leaves target identity, qubit capacity, freshness, canonical identity, and
   evidence references under the existing Core contract;
5. matches the registered extension through a pure, non-submitting matcher; and
6. names missing, stale, unknown, unmeasured, unsupported, not-exposed, identity,
   capacity, and value mismatches separately.

The coarse `CloudBackendProfile.supports_dynamic_circuits` boolean is only a
consistency hint. `True` without the device's detailed feature vector produces
missing-fact blockers and cannot admit execution.

## Compatibility and boundary

- no root, stable, experimental, or Runtime package export is added;
- no existing signature or serialized schema changes;
- no provider SDK, credential, discovery, network, compiler, package, or task
  boundary is entered;
- RequirementSet and TargetCapabilitySnapshot retain their existing v1 schema;
- the extension codec is internal and owned by Runtime dynamic execution; its
  removal condition is a reviewed Core schema containing the same nine fields
  plus lossless migration tests, with Target Capabilities v2 as the target
  contract version; and
- existing local and static remote behavior is unchanged.

## Exit evidence

The bridge exits when deterministic round trips preserve requirement and
snapshot identity; detailed declared facts admit an offline match; the coarse
boolean alone cannot; every non-accepted fact state has a distinct blocker;
stale and wrong-target snapshots fail; capacity and feature mismatches remain
separate; unavailable mid-circuit data is never invented; and the bridge remains
absent from public exports.
