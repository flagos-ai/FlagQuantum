# API Change Implementation 089: Post-routing rematch contract

## Status

**Batch A candidate contract implemented on 2026-10-24; Batch B pure Runtime
rematch is implemented in the successor change.**

The API owner authorized this bounded work with:

```text
approve API-CHANGE-089-DYNAMIC-POST-ROUTING-CAPABILITY-REMATCH
```

The machine-readable baseline is
[`dynamic-post-routing-rematch-v1-candidate.json`](../../contracts/dynamic-post-routing-rematch-v1-candidate.json).
This batch does not activate Runtime, a provider, networking, or task submission.

## User outcome

The supported user-facing behavior remains one early, credential-free refusal
for remote dynamic targets that do not supply an exact detailed snapshot:

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

The example performs no credential lookup, provider import, network request, or
task submission. Batch A adds no successful remote route and no user-facing
post-routing object.

## Frozen decisions

- The source `RequirementSet` and its identity remain immutable for an
  execution attempt.
- Effective requirements are the preserved source requirements plus any
  requirements introduced by routing.
- Source semantic loss is a blocker even if the routed artifact would otherwise
  match the device.
- Preflight and post-routing admission use the same exact target and snapshot.
- Snapshot freshness is checked again after routing; the gate cannot refresh it
  silently.
- Route-induced facts distinguish missing, unknown, unmeasured, unsupported,
  and not-exposed states.
- Blockers are emitted in contract order: source identity, semantic
  preservation, target identity, snapshot identity, freshness, route facts,
  then provider legality.
- No blocker can cause target substitution, local fallback, or a static rewrite.

## Conformance baseline

The API-contract suite executes the checked-in vectors against a private
reference fake. The vectors cover:

1. an unchanged legal route;
2. replacement of the source requirement identity; and
3. simultaneous semantic loss, target and snapshot mismatch, staleness,
   unsupported and not-exposed route facts, and unavailable provider legality.

The Batch B pure Runtime implementation passes these vectors without changing
their expected blockers. The former test-only fake has been removed; the
contract suite now calls the production policy directly.

## Compatibility and safety boundary

- no stable or experimental public export is added;
- no signature, Circuit IR, ExecutionPlan, DeploymentPackage, ExecutionResult,
  or provider schema changes;
- no positive remote target is enabled;
- no package is sealed and no provider dry run occurs;
- no provider SDK, credential, discovery, network, task, charge, result, or RNG
  boundary is crossed; and
- local `DynamicCircuit` and static remote behavior are unchanged.

## Exit evidence

Batch A exits when the candidate contract records the exact approval, identity
inputs, monotonicity rules, decision order, blocker vocabulary, external
boundary, and reusable conformance vectors; tests also prove the current stable
workflow still refuses before compilation, remote imports, and RNG mutation.

The Batch B implementation remains pure and credential-free: it reuses the
existing requirement/snapshot matcher and returns a deterministic internal
decision. Routing integration remains a later, separately reviewable batch.
