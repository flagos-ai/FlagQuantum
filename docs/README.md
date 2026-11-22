# FlagQuantum Documentation

FlagQuantum documentation is organized by purpose so that users, contributors,
and maintainers can find the right level of detail quickly.

## Start here

- [Publication validation scope](reference/PUBLICATION_VALIDATION.md) — tested
  workflows, evidence boundaries, and research goals

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
| [API change proposals](api-changes/README.md) | Approved and proposed Stable Core API changes |
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

## HTML documentation

The source-checkout Sphinx entry is `docs/conf.py`, with `docs/index.rst` as
the landing page. Install the optional development renderers in an isolated
environment, then build from the repository root.

```console
python -m pip install 'sphinx==8.1.3' 'myst-nb==1.3.0' 'pydata-sphinx-theme==0.16.1'
python -m sphinx -b html docs docs/_build/html
```

Notebook outputs are rendered as stored. Rendering keeps notebook execution off.
Local project branding reuses `assets/logo_flagquantum.png`. Source-audit
environment details and site acceptance are recorded in
[the documentation site audit](development/DOCUMENTATION_SITE_FILE_AUDIT.md).

The complete component notebook catalog can be published before the same build.
Publication preserves the authored notebook bytes and uses the explicit section
order maintained by the existing development tools.

```console
python -m tools.development.notebook_publication
python -m sphinx -b html docs docs/_build/html
```

The generated library entry is `docs/_build/html/components/index.html`, with
120 distinct authored notebooks. Generated copies under `docs/components` follow
the checkout's existing publication and retention policy. The root homepage's
full catalog integration and the API generator's setter-signature refusal are
tracked separately in the development audits.
