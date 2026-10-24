# API Change Proposal 089: Dynamic post-routing capability rematch

## Status

**Proposal approved on 2026-10-24; Batch A establishes the candidate contract.**

API Change 086 requires capability matching both before and after routing. API
Change 087 installed the stable, fail-closed preflight and preserved the source
requirement identity, but intentionally stopped before compiler, provider,
package, network, and task boundaries. This proposal defines the next internal
gate: proving that routing and dialect lowering did not weaken the requirements
that admitted the source program.

The API owner approved this scope with:

```text
approve API-CHANGE-089-DYNAMIC-POST-ROUTING-CAPABILITY-REMATCH
```

The approval authorizes a credential-free, non-submitting implementation of
the contract below. It does not authorize a provider SDK, credential lookup,
network request, deployment package, remote task, live-QPU claim, or new stable
public API.

## User outcome

The user keeps one execution entry. No post-routing object, capability snapshot,
fingerprint, or provider-specific option becomes part of the user API:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)

# This remains the eventual remote workflow. A positive remote provider slice
# is not enabled by this proposal.
result = fq.run(circuit, target="provider:backend", shots=128)
```

Until a separately approved provider vertical slice supplies fresh, exact
device facts and a legal routed artifact, the call continues to fail before any
external activity.

## Current runnable example

Users can verify the present fail-closed boundary without credentials or
hardware:

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

Expected result: the program raises `CapabilityError` before compilation,
credentials, provider import, network access, task creation, charging, or random
number consumption. This example describes implemented behavior; the positive
remote result shown above remains a target state.

## Problem

A source program can match a target and still become illegal after routing or
dialect lowering:

1. routing can introduce physical qubits, swaps, decomposed gates, or a mapping
   whose directed edges the target does not support;
2. lowering can replace one operation with a sequence that has stronger gate,
   timing, controller, or classical-control requirements;
3. a transform can accidentally drop a measurement, condition, result
   requirement, or other dynamic semantic while still producing syntactically
   valid output;
4. the exact device snapshot can expire while compilation is in progress; and
5. matching only a newly derived routed requirement set can hide requirements
   that disappeared because of a compiler defect.

The existing experimental package builder checks coarse backend compatibility
before and after routing. That is useful but insufficient for stable admission:
it does not prove that the API Change 086 requirement identity survived, and its
coarse profile fields are not evidence-bearing target facts.

## Decision

### 1. Preserve the source requirement identity

The requirement set derived from validated source Circuit IR is immutable for
the execution attempt. Routing and lowering may add requirements but cannot
replace, narrow, or reinterpret that source set.

The post-routing gate receives the original `RequirementSet` and its
`requirement_set_id`; it does not derive a new source truth from the routed
artifact. A different or absent source identity is a refusal, not a cache miss
or a reason to recompute silently.

### 2. Match both preserved and route-induced requirements

The routed artifact is inspected for additional requirements such as physical
qubit span, introduced operations, directed connectivity, classical-control
shape, and requested result availability. The gate then checks:

- every source requirement against the exact target snapshot;
- every route-induced requirement against the same snapshot; and
- semantic preservation between source and routed artifacts.

The effective requirement set is monotonic: it is the preserved source set plus
requirements introduced by the approved transform. A routed artifact cannot
pass by making a source measurement, conditional gate, condition value, or
result obligation disappear.

No generic equivalence attestation is added. If a transform cannot demonstrate
preservation using the existing IR and routing evidence, the gate fails closed.

### 3. Reuse one exact target and snapshot

Preflight and post-routing checks use the same canonical target identity and
device-scoped `TargetCapabilitySnapshot`. A post-routing gate must reject:

- a different provider, backend, target revision, or environment;
- a missing, unknown, unmeasured, unsupported, or not-exposed mandatory fact;
- a snapshot that became stale before the post-routing decision; and
- evidence whose scope no longer covers the routed physical program.

The rematch does not refresh discovery or acquire a replacement snapshot. A
future provider-owned refresh path requires separate authorization because it
crosses credential and network boundaries.

### 4. Keep generic and provider-specific legality separate

Runtime owns the provider-neutral requirement rematch and fail-closed decision.
Compiler or the current dynamic router owns the transformed artifact and
routing evidence. Remote owns provider dialect declarations and device-specific
restrictions. Provider SDK objects and raw payloads do not enter Runtime or
Core.

The later provider vertical slice must combine both results before sealing a
package:

```text
source DynamicCircuit
  -> source requirements + exact target snapshot
  -> pre-routing match
  -> route and lower without external activity
  -> semantic-preservation check
  -> source requirements + route-induced requirements
  -> post-routing match against the same exact snapshot
  -> provider-specific legality check
  -> package sealing (not authorized here)
```

No later stage can erase an earlier blocker.

### 5. Fail before irreversible or provider-owned work

Any post-routing refusal occurs before:

1. deployment-package sealing or identity publication;
2. provider SDK import or construction;
3. credential lookup, discovery, or network access;
4. provider dry run;
5. task creation, submission, polling, cancellation, or charging;
6. result construction; and
7. random-number consumption.

There is no automatic fallback to a simulator, static rewrite, another backend,
or local execution.

### 6. Add no public API or serialized schema

This proposal adds no root export, public class, stable parameter, target alias,
exception family, result field, provider option, or serialized public schema.
It reuses the API Change 086 `RequirementSet` and
`TargetCapabilitySnapshot` authorities and the API Change 087 stable refusal.

An implementation may add private Runtime helpers and deterministic diagnostics.
It must not add another public capability model or require users to understand
internal identities.

## Deterministic refusal contract

The implementation proposal must freeze precise blocker codes for at least:

- missing or changed source requirement identity;
- source semantic lost after routing;
- route-induced capability missing, unsupported, unknown, unmeasured, or not
  exposed;
- target or snapshot identity mismatch;
- snapshot expiry before post-routing admission;
- routed capacity or physical-scope overflow;
- directed-connectivity or introduced-operation mismatch; and
- provider-specific legality unavailable.

Blocker ordering is deterministic. Diagnostics may expose source requirement,
snapshot, routed-artifact, and routing-evidence identities for inspection, but
these remain internal and are not added to stable exceptions or results by this
proposal.

## Proposed implementation sequence

1. **Failing-first contract tests.** Use a credential-free fake router and
   exact in-memory snapshot. Prove that a removed condition, added unsupported
   gate, wrong target, stale snapshot, or oversized physical mapping is refused
   before all external boundaries.
2. **Pure Runtime rematch.** Reuse the existing requirement/snapshot matcher,
   preserve the source requirement identity, derive only route-induced facts,
   and return one deterministic internal decision.
3. **Routing integration.** Connect the pure gate to the existing dynamic
   routing/package path without activating stable remote execution. Prove that
   unchanged legal routing passes offline and that coarse backend booleans
   cannot admit it.
4. **Provider vertical slice proposal.** Separately define provider-owned
   snapshot acquisition, dialect legality, sealed package, exactly-once mocked
   submission, task identity, normalized result, and bounded live evidence.

Each implementation PR must include a runnable credential-free user example.
No subsequent PR is authorized to submit a remote task or make a hardware
support claim merely because this proposal exists.

## Verification plan

- source and routed requirement identities are deterministic;
- source requirements remain mandatory even if absent from the routed artifact;
- route-induced requirements are additive and fully rematched;
- the same exact target and snapshot are used at both gates;
- snapshot freshness is checked again after routing;
- all refusals occur before provider, credential, network, package, task, result,
  and RNG activity;
- stable local `DynamicCircuit` execution is unchanged;
- existing static remote routes are unchanged;
- root exports and serialized schemas are unchanged; and
- the shortest user example remains executable without credentials.

## Acceptance

- [x] The API owner approves this proposal and no-submit boundary.
- [x] A failing-first contract freezes post-routing blocker semantics.
- [ ] The original requirement identity survives routing and lowering.
- [ ] Route-induced requirements are additive rather than replacements.
- [ ] Preflight and rematch use one exact target and snapshot.
- [ ] Stale, wrong-target, semantically weakened, and physically illegal routed
      artifacts fail closed.
- [ ] No provider, credential, network, package, task, result, or RNG boundary is
      crossed on refusal.
- [ ] Local dynamic and static remote behavior remains unchanged.
- [ ] No public API or serialized schema changes.
- [ ] A provider vertical slice remains separately authorized.

## Non-goals

- Enabling any positive remote DynamicCircuit target.
- Discovering, refreshing, or trusting provider capabilities over a network.
- Defining provider pricing, queues, credentials, tasks, polling, or results.
- Claiming Quafu, Azure, Braket, IQM, IBM, Quantinuum, or another QPU supports
  FlagQuantum dynamic execution.
- Adding automatic backend selection or simulator fallback.
- Adding loops, nested control flow, conditional measurement, conditional reset,
  dynamic gradients, noise, or distributed dynamic execution.
- Merging `DynamicCircuit` into `Circuit`.
