# API Change Proposal 063: Unify backend capability registration with extension execution

## Status

**Proposed. No Stable Core change requested.**

This proposal connects two facilities that already exist in the repository and
currently do not reference each other. It adds no new plugin concept, no new
entry-point group, and no new manifest format.

## Problem

FlagQuantum has two independent backend facilities.

**Capability registration** lives in `flagquantum/runtime/backend_registry.py`.
It defines `BackendCapabilities` (name, tensor backend, devices, dtypes,
autograd and mode support, preferred device, accelerators), keeps them in a
`ContextVar`-scoped immutable mapping, and exposes `register_backend`,
`list_backends`, `get_backend_capabilities`, `set_active_backend`,
`resolve_device`, and `resolve_dtype`. It is consumed by
`runtime/execution.py`, `runtime/configuration.py`, and
`runtime/execution_plan_contract.py`.

**Extension lifecycle** lives in `flagquantum/ecosystem/extensions/sdk.py`. It
defines `SDK_API_VERSION`, the `flagquantum.extensions` entry-point group,
`ExtensionManifest`, `CapabilityRequest`/`CapabilityResponse`,
`ExtensionRegistry.negotiate`, version and duplicate-registration rejection, and
typed extension protocols including `ExecutionBackendExtension`, whose contract is
`execute(program, parameters)`.

`register_backend` accepts a capabilities record and nothing else. It has no
execution function to register, so a backend registered there can describe itself
but cannot run anything. Conversely, nothing under `flagquantum/runtime/` imports
the extension SDK, so an `ExecutionBackendExtension` that has been discovered,
version-checked, and negotiated still has no route into execution. The two halves
never meet.

The consequences are concrete:

- `fq.run` cannot dispatch to a third-party backend, because the only registration
  path reachable from Runtime carries no executable.
- Adding a backend today requires editing Runtime rather than installing a
  package, which defeats the boundary that engineering decision principle 10 asks
  to prove by replacement.
- `BackendCapabilities` carries no distribution-semantics or scalability fields,
  so a backend cannot declare the evidence vocabulary that non-negotiable rules 1
  and 5 require before any distributed claim.
- The `execute(program, parameters)` signature is untyped against the actual
  artifact and result contracts. `MULTI_LEVEL_IR_ARCHITECTURE.md` § 7.1 and § 7.2
  define `ExecutableArtifact` and `RuntimeAdapter`; a backend extension should
  consume the former under the latter, not an unconstrained `Any`.

This is a wiring and admission gap, not a missing plugin architecture.

## Decision

### 1. One admission path from extension to execution

An `ExecutionBackendExtension` that is discovered, version-compatible, and
negotiated successfully contributes a runtime-visible capability record through
the existing `register_backend` mechanism. Runtime consults that record exactly as
it consults the built-in record today.

No second registry is created. `ExtensionRegistry` remains the discovery,
version, and negotiation authority; `backend_registry` remains the execution-time
capability authority; the admission step is the single crossing between them.

### 2. Capability records carry the evidence vocabulary

The backend capability record gains the fields required to describe execution
semantics honestly, so that a backend cannot present itself as more than it is:

| Field | Purpose |
| --- | --- |
| `distribution_semantics` | one of the classifications in `AGENTS.md` § Required Development Workflow |
| `scalability_claim_allowed` | false unless one logical workload is partitioned across ranks |
| `supports_gradients` | whether backward is available through this backend |
| `artifact_formats` | accepted `ExecutableArtifact` formats |
| `blockers` | declared, deterministic reasons a capability is unavailable |

A record that declares sharded distribution without the corresponding evidence
requirement must not be usable to promote a scalability claim.

### 3. Fail closed at planning time

A requested backend that is registered but not executable, or executable but
lacking a required capability, is rejected at planning time with a typed
capability error naming the blocker. There is no execution-time fallback, no
substitution of a different backend, and no silent CPU path.

The repository already has an explicit opt-in for the ecosystem adapters:
`ExecutionOptions.allow_backend_fallback`, read through `runtime/policy.py` and
observed by the Qiskit, Cirq, and PennyLane adapters and by `runtime/module.py`.
This proposal does not remove it and does not extend it. It states two limits.
First, the flag governs adapter-level fallback and stays opt-in, defaulting to
disabled, so the default path already fails closed. Second, a fallback selected
under that flag must be recorded in the plan, result, and evidence rather than
performed silently, per non-negotiable rule 9. Any admission behaviour introduced
here is subordinate to the flag rather than an additional, independent fallback
mechanism.

### 4. Keep the local fast path unchanged

The built-in PyTorch record remains the default and must not acquire an extension
lookup, a negotiation step, or a version check on the ordinary path. Extension
admission happens at registration or explicit refresh, never per execution.

### 5. Prove the boundary by replacement

Acceptance requires a second implementation of an existing capability, registered
through the extension SDK and selected by configuration, with no change to the
consuming call site. Per engineering decision principle 10, a boundary is not
complete because an interface exists.

## Public API

No Stable Core export is added, removed, or changed. `docs/public_api_v1.json`
and `contracts/public-api-v0.2-baseline.json` are unaffected.

The externally visible behaviour change is narrow and is the point of the
proposal:

```python
import flagquantum as fq

# A third-party backend installed as a `flagquantum.extensions` entry point is
# selected by name, exactly as a built-in backend is. Its own package documents
# any installation prerequisite.
result = fq.run(circuit, options=fq.ExecutionOptions(backend="third_party_sv"))

# A backend that is registered but cannot serve the request fails here, before
# execution, with an actionable blocker.
```

Requesting a registered-but-inexecutable backend currently fails late or
ambiguously; after this change it fails at planning time with a typed capability
error. That is a failure-behaviour improvement, not an API addition.

## Compatibility

`register_backend`, `list_backends`, `get_backend_capabilities`, and
`set_active_backend` keep their names, parameters, and return types. New
capability fields are additive with defaults that preserve current behaviour, so
existing callers that construct a record or read a field are unaffected.

Rollback is removal of the admission step: the built-in path returns to being the
only executable record, and pushed-back third-party records become
non-executable again. No serialized public schema is involved, so rollback needs
no migration and no deprecation cycle.

## Acceptance

- [ ] An `ExecutionBackendExtension` registered through the `flagquantum.extensions`
      entry-point group becomes selectable by name and executes through the
      ordinary runtime path.
- [ ] A version-incompatible extension is rejected before activation with
      `ExtensionCompatibilityError`; a duplicate registration fails closed.
- [ ] A registered backend lacking a required capability fails at planning time
      with a typed capability error naming the blocker.
- [ ] No silent backend substitution or fallback occurs on any path; a test fails
      if a fallback is introduced.
- [ ] The built-in PyTorch path gains no per-execution extension lookup or
      negotiation, verified against the recorded local baseline.
- [ ] A replacement test swaps at least one existing backend implementation for an
      extension-provided one without modifying the consuming call site.
- [ ] A backend record that declares sharded distribution cannot be used to
      promote a scalability claim without the evidence required for that
      classification.
- [ ] `python tools/ci_tier.py pr-runtime` and the `distributed_cpu` tier pass.
- [ ] `docs/public_api_v1.json` and `IR_VERSION` are unchanged.
- [ ] Repository owner authorizes the wiring change.

## Non-goals

- A stable C ABI for out-of-process or native performance backends. That is a
  separate proposal with its own ABI-stability, packaging, and language-binding
  analysis; it is deliberately not bundled here so that the Python path can land
  first and be measured.
- Any credential, quota, queue, or network lifecycle in a backend. Compiler and
  execution backends own no network lifecycle, per
  `MULTI_LEVEL_IR_ARCHITECTURE.md` § 19.11.
- A new entry-point group, manifest format, or negotiation protocol.
- Any Stable Core change, capability maturity promotion, or parity claim.
