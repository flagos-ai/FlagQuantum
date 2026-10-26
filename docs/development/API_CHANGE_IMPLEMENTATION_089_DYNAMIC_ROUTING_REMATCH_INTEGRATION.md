# API Change Implementation 089: Offline routing integration

## Status

**Batch C implemented on 2026-10-24; provider integration remains inactive.**

This batch connects validated dynamic routing evidence to the pure Runtime
post-routing rematch. The bridge is internal and offline. It neither changes
the stable execution path nor authorizes package sealing, provider SDK import,
credentials, discovery, network access, task creation, submission, or a live
QPU support claim.

## User-facing behavior

The user continues to see one fail-closed execution entry:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError

circuit = DynamicCircuit(3).measure(0, classical_bit=0)
circuit.conditional("cx", (0, 2), classical_bit=0)

try:
    fq.run(circuit, target="quafu:Baihua", shots=8)
except CapabilityError as error:
    assert "dynamic_remote_preflight_failed" in str(error)
else:
    raise AssertionError("unsupported remote dynamic execution was admitted")
```

Expected result: `CapabilityError` is raised before compiler, credential,
provider, network, package, task, result, or RNG activity. Batch C does not make
the example succeed; it strengthens the offline checks required before a future
provider path may do so.

## Implemented bridge

For an already-routed `DynamicCircuit`, the internal bridge now:

1. validates the routing plan with the existing deployment routing-evidence
   authority;
2. records deterministic routed-artifact and routing-evidence SHA-256
   identities;
3. derives additive requirements for every routed operation, physical qubit,
   and directed or undirected coupling used by the artifact;
4. resolves operation and physical-capacity facts from the exact
   `TargetCapabilitySnapshot` through the existing Core matcher;
5. refuses caller attempts to override Core-backed facts;
6. requires explicit connectivity facts because Target Capabilities v1 has no
   connectivity field; and
7. invokes the Batch B rematch with the original requirement identity, target,
   snapshot, and provider-legality decision.

Missing, unknown, unmeasured, unsupported, and not-exposed facts remain
distinct. A coarse dynamic-support boolean or qubit count cannot stand in for
native-gate or connectivity evidence.

## Compatibility and ownership

- no root, deployment, dynamic, or experimental export is added;
- no public signature or serialized schema changes;
- Runtime remains provider-neutral;
- Deployment coordinates existing routing evidence and Runtime contracts;
- the stable `fq.run` path is unchanged; and
- no provider or positive remote target is activated.

## Next stage

A separately reviewable provider vertical-slice proposal must define how one
provider supplies evidence-bearing connectivity and legality facts before this
bridge can participate in package sealing or submission.
