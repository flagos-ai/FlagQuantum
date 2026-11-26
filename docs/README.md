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
python -m tools.development.notebook_publication
python -m sphinx -b html docs docs/_build/html
```

Notebook outputs are rendered as stored. Rendering keeps notebook execution off.
Local project branding reuses `assets/logo_flagquantum.png`. Source-audit
environment details and site acceptance are recorded in
[the documentation site audit](development/DOCUMENTATION_SITE_FILE_AUDIT.md).

Publish the complete component notebook catalog before the HTML build so that
the homepage's component-library route resolves every registered notebook.
Publication preserves the authored notebook bytes and uses the explicit section
order maintained by the existing development tools.

The homepage links through `docs/_build/html/reference/components.html` to the
generated library entry at `docs/_build/html/components/index.html`, with
120 distinct authored notebooks. Generated copies under `docs/components` follow
the checkout's existing publication and retention policy. The root homepage's
catalog integration is documented in
[the homepage audit](development/DOCUMENTATION_HOMEPAGE_FILE_AUDIT.md).
The API generator's setter-signature refusal remains a separate open boundary.

For a local documentation preview that rebuilds after edits, install the
optional pinned watcher and use the same authored source tree.

```console
python -m pip install 'sphinx-autobuild==2024.10.3' 'starlette==0.50.0' \
  'uvicorn==0.40.0' 'watchfiles==1.1.1' 'websockets==15.0.1' \
  'anyio==4.12.1' 'h11==0.16.0'
python -m sphinx_autobuild --host 127.0.0.1 --port 8000 docs docs/_build/html
```

The complete nine-entry upstream documentation input, native owners, installed
tool constraints and actual local rebuild evidence are recorded in
[the documentation dependency audit](development/DOCUMENTATION_DEPENDENCY_INPUTS_AUDIT.md).

The complete three-stage contributor and publication workflow is described in
[the documentation build guide](development/DOCUMENTATION_BUILD_WORKFLOW.md).
The same optional tools are also available through `make -C docs html` and
`make -C docs live`. Both executable names must be on PATH for the Makefile's
original preflight. See [the Makefile audit](development/DOCUMENTATION_MAKEFILE_AUDIT.md)
for target dispatch, actual build/clean and retained publication boundaries.
