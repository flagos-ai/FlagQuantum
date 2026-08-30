# FlagQuantum Documentation

FlagQuantum documentation is organized by purpose so that users, contributors,
and maintainers can find the right level of detail quickly.

## Start here

- [API reference](reference/API.md) — public Python interfaces and examples
- [Runtime architecture](architecture/RUNTIME_ARCHITECTURE.md) — execution
  model, planning, and distributed semantics
- [Runtime configuration](reference/RUNTIME_CONFIGURATION.md) — backend and
  environment controls
- [Known limitations](reference/KNOWN_LIMITATIONS.md) — current support
  boundaries
- [Testing manual](development/TESTING.md) — verification tiers and commands
- [Product roadmap](roadmap/FLAGOS_ALIGNED_RELEASE_TRAIN.md) — planned
  user-facing capabilities

## Documentation map

| Area | Purpose |
| --- | --- |
| [Concepts](concepts/README.md) | Product-wide principles and scalability semantics |
| [Architecture](architecture/README.md) | Runtime, parallelism, state, and subsystem design |
| [Reference](reference/README.md) | Public APIs, contracts, policies, and support matrices |
| [Guides](guides/README.md) | Operational runbooks and performance case studies |
| [Development](development/README.md) | Repository, testing, dependency, and release workflows |
| [Roadmap](roadmap/README.md) | Vision, maturity, ecosystem, and delivery plans |
| [Generated](generated/) | Machine-generated capability and stable-API inventories |

Machine-readable capability contracts live in the repository-level
[`contracts/`](../contracts/README.md) directory. Repository-wide architecture,
capability-maturity, and dependency policies remain at the repository root.
These files are inputs to repository tooling rather than manually curated
guides.

For runnable workflows, continue to the [examples index](../examples/README.md).
For benchmark methodology and evidence, use the
[benchmark index](../benchmarks/README.md).
Repository retention and evidence-placement rules are defined in
[repository governance](development/REPOSITORY_GOVERNANCE.md).
