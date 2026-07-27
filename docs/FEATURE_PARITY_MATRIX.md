# FlagQuantum Capability Matrix

This page is an index, not a second capability database.

| Capability class | Authoritative source | Meaning |
| --- | --- | --- |
| Stable Python API | [`public_api_v1.json`](public_api_v1.json) | Importable and covered by the stable API contract test |
| Operators and backend lowerings | [`operator_manifest.json`](operator_manifest.json) | Registered executable lowering; see the [generated table](generated/OPERATOR_CAPABILITIES.md) |
| Runtime evidence fields | [`runtime_contracts.schema.json`](runtime_contracts.schema.json) | Typed/versioned vocabulary, not proof a run occurred |
| Measured benchmark evidence | [`../benchmarks/results/`](../benchmarks/results/) | Audited artifacts with provenance; plans and fixtures are excluded |
| Supported Python/dependencies | [`../pyproject.toml`](../pyproject.toml) | Installable support range and dependency groups |

## Current claim boundary

Local execution, registered lowering support, distributed development probes,
and release-grade scalability are separate claims. A `yes` in the generated
operator table only proves a lowering is registered. Distributed or accelerator
claims additionally require runtime-generated evidence accepted by benchmark
audit and release policy.

Planned capabilities belong in roadmap documents labeled **future intent**.
They do not appear as supported rows here until executable manifests and tests
exist.
