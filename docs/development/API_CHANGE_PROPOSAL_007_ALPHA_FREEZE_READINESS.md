# API Change Proposal 007: Extension Protocol and Alpha Freeze Readiness

## Status

**Approved, not frozen — migration into Ecosystem approved; previous freeze revoked.**

- Target: first public alpha.
- Affected interface: `flagquantum.ecosystem.extensions`.
- Root name changes: none.
- Machine candidate: `contracts/extension-protocol-v1-candidate.json`.
- Authorization: the API owner explicitly authorized this implementation on
  2026-09-01. This did not formally freeze Proposals 001-007 or the entire public API.
- Freeze record: on 2026-09-01, the API owner explicitly approved the exact
  Proposal 007 contract bound by the review packet. Concrete plugin implementations
  and the overall first public alpha freeze remained unapproved.
- Migration record: on 2026-09-07, before public release, the API owner explicitly
  revoked the old namespace freeze and authorized direct migration into
  `flagquantum.ecosystem.extensions`, without a compatibility layer.
- Amendment: API Change Proposal 019 added explicit installed-extension discovery
  and whole-circuit compiler contracts on 2026-09-09. Proposal 019 is now the source
  for the machine candidate; this document retains the original decision history.

## Decisions

### 1. Stabilize Extension Boundaries, Not Third-Party Implementations

`flagquantum.ecosystem.extensions` is a public namespace awaiting freeze, providing
manifests, capability negotiation, lifecycle, task-local registration, exception
isolation, and conformance entry points. It adds no root `flagquantum` names.
Concrete backends, providers, and compiler passes remain experimental by default;
certification requires separate compatibility, numerical, security, and maintainer review.

Extension authors need only import:

```python
from flagquantum.ecosystem.extensions import (
    CapabilityRequest,
    CapabilityResponse,
    ExtensionConfig,
    ExtensionManifest,
)
```

Reference backends/providers live in `examples/extensions/reference_extensions.py`.
Contract tests inspect its AST to reject FlagQuantum internal/runtime/core imports
and execute backend conformance.

### 2. Retain `noise_model`

`noise_model` already appears in `fq.plan`, `fq.run`, Circuit convenience methods,
ExecutionPlan extension serialization, docs, and tests. The name accurately
identifies a model, not noise strength or one noise event. Renaming it to `noise`
before release would add migration work and ambiguity without sufficient benefit.
The candidate API retains `noise_model`.

### 3. Keep ExecutionPlan and DeploymentPackage Separate

- `ExecutionPlan` owns local planning identity, capability constraints,
  caching/recovery, and exact execution.
- `DeploymentPackage` owns provider targets, portable submission assets, routing
  evidence, and submission lifecycles.
- Plans expose no `submit`, provider credentials, or job identities.
- Deployment packages are not replacement local-planner results.

These are consecutive but distinct lifecycles. Merging them would compromise both
the minimal local contract and remote deployment's security boundary.

## Candidate Extension Protocol

Protocol version: `1.0`. Extensions must:

1. Declare identity, kind, API version, and capabilities through `ExtensionManifest`.
2. Negotiate capabilities before activation.
3. Manage lifecycles explicitly through `start`/`close`.
4. Install through immutable task-local registries without modifying root APIs or
   global operator tables.
5. Keep credentials out of serializable `ExtensionConfig`.
6. Wrap extension errors at the boundary, retaining original `__cause__`.
   Compatibility errors map to `CapabilityError`; lifecycle errors to `ExecutionError`.
7. Verify backend/provider behavior through public conformance entry points.

## Executable Acceptance Criteria

- [x] A separate machine-readable extension protocol candidate exists.
- [x] `flagquantum.ecosystem.extensions.__all__` exactly matches the candidate.
- [x] Extension names stay out of the stable root namespace.
- [x] Third-party backend examples import only the public extension namespace.
- [x] Capability, dtype/device, gradient, exception, and cleanup conformance pass.
- [x] The `noise_model` naming decision is recorded as a candidate stable decision.
- [x] Tests/docs protect ExecutionPlan/DeploymentPackage ownership.
- [ ] API owner reapproves the migrated extension contract freeze.
- [ ] API owner separately approves the overall first public alpha freeze.

## Compatibility Strategy

After refreezing, SDK `1.x` permits compatible additions only. Breaking protocol
methods, manifest fields, or lifecycles requires a new SDK API major, explicit
diagnostics, and a supported-version window. Extension package versions evolve
separately from SDK protocol versions and cannot replace protocol negotiation.
