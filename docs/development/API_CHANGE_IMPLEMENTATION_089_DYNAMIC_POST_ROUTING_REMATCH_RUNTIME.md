# API Change Implementation 089: Pure Runtime rematch

## Status

**Batch B implemented on 2026-10-24; routing integration remains inactive.**

This batch replaces the API Change 089 test-only reference evaluator with a
pure Runtime decision. It is internal and credential-free. It does not enable
a remote dynamic target, build a deployment package, import a provider SDK, or
perform network or task activity.

## User-facing behavior

The stable user workflow is unchanged and remains fail-closed when no exact,
evidence-bearing device snapshot is available:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError

circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)

try:
    fq.run(circuit, target="quafu:Baihua", shots=8)
except CapabilityError as error:
    assert "dynamic_remote_preflight_failed" in str(error)
else:
    raise AssertionError("unsupported remote dynamic execution was admitted")
```

Expected result: `CapabilityError` is raised before compilation, credential
lookup, provider import, network access, package sealing, task creation, result
construction, or random-number consumption.

## Implemented boundary

The internal rematch now:

1. preserves and verifies the original `RequirementSet` identity;
2. derives source semantic preservation by comparing the source and routed
   dynamic requirement extensions;
3. requires the complete `TargetIdentity` and exact snapshot identity used at
   preflight;
4. reruns the existing dynamic capability matcher against that snapshot;
5. checks freshness again at the post-routing decision time;
6. adds route-induced fact states without replacing source requirements; and
7. keeps provider-specific legality as a separate, mandatory decision.

Blockers follow the order frozen by the candidate contract. Missing, unknown,
unmeasured, unsupported, and not-exposed route facts remain distinguishable.
The decision cannot refresh a snapshot, substitute a target, or fall back to a
local simulator.

## Compatibility

- no root or `flagquantum.runtime.dynamic` export is added;
- no public signature or serialized schema changes;
- no compiler or deployment dependency is added to the Runtime rematch;
- local dynamic and static remote behavior is unchanged; and
- no positive QPU-support claim is made.

## Verification

The conformance vectors now execute the production policy rather than a
test-only fake. Focused tests also cover an unchanged legal route, source
semantic loss, stale and changed snapshots, complete target-identity mismatch,
all route-fact refusal states, and the absence of public exports.

The next batch may connect this pure decision to the existing offline dynamic
routing path. It must still stop before provider, credential, network, package,
or task boundaries and requires separate review before merge.
