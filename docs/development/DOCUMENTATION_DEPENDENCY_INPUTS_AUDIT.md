# Complete optional documentation dependency input

## Source and native ownership

Q-0095 covers the entire nine-entry docs.txt declaration at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its frozen Git blob is
`fa1c3a966986532f9d1e81715e45468b53960c55`. The acceptance test parses every
entry in order and verifies each actual installed distribution against the
original constraint. The complete source input is retained in the reference
environment. Native code continues to use its existing developer-tool owners.

| Source dependency | Existing native responsibility and actual evidence |
| --- | --- |
| ipywidgets | Native notebook widgets. Real slider state creation, mutation and cleanup execute in the acceptance test. Browser widget interaction retains its separate UI scope. |
| nbconvert | Existing notebook_execution export owner. The actual Markdown exporter renders stored native notebook content and output. Kernel execution and batch exports reuse their accepted owner tests. |
| nbformat | Existing notebook reading, publication and rendering owners. Actual notebook construction, stored output and complete 120-notebook publication are exercised. |
| sphinx | Native docs/conf.py. Actual source/native strict fixtures and complete native site builds remain recorded in the site audit. |
| sphinx-autobuild | Optional external local build/watch process using that same configuration and Sphinx. A real loopback HTTP server returns the initial page and changed content after an actual Markdown edit and rebuild. |
| pydata-sphinx-theme | Existing native site theme. Sidebar, local images, styles and stored notebook rendering use the accepted actual renderer fixtures. |
| myst-nb | Existing native Markdown/notebook renderer. Mathematical markup, definition lists, anchors, stored outputs and execution-off are exercised in the site fixtures. |
| tensorflow-docs ~=2023.5.24 | Existing reference_templates/reference_template_pages owners. The installed 2023.5.24.56664 SDK satisfies the constraint and retains the complete original/native template comparison evidence. |
| griffe | Existing reference_package and reference page owners. The installed 1.15.0 loader, signature and docstring paths retain their accepted source/native comparisons. |

The root pyproject retains native build/runtime/development authority. These
optional tooling installations stay in the external development environment.
Their standard Python module and subprocess boundaries are already used by the
native commands. Missing optional tools produce the existing visible failures.
The optional live preview can be replaced by another external process using the
same authored source and generated HTML. Removing it leaves the existing Sphinx
build command as the documentation owner.

## Actual build and rebuild workflow

The live acceptance copies the real native configuration, templates, styles and
logo into an isolated documentation tree. It launches the actual frozen-version
sphinx-autobuild command, bound explicitly to 127.0.0.1 on a free local port.
The first HTTP response contains the authored title. After the source file is
edited, the HTTP response contains the rebuilt title and new content.
No notebook code is executed by the renderer. Each observation phase is bounded
at 45 seconds. The child has its own process session on POSIX. Finalization
terminates the process group, waits for it, and has a bounded force-kill path.
The standard-library HTTP client bypasses external proxies for this local test.

The initial sandbox denies loopback binding. The actual live comparison runs
with the explicitly authorized network execution permission. The test's listen
address remains local. The isolated nbconvert target also needs JUPYTER_PATH
pointing to its own share/jupyter directory, as in the existing notebook tests.
A normal environment installation discovers these bundled templates itself.

The native user command is recorded in docs/README.md. It runs through the
existing configuration and authored files. Generated paths follow the current
publication policy. The live preview adds no framework API or execution layer.

## Dependency intake and retained scope

The live preview adds seven explicitly pinned distributions to the isolated
reference environment. All seven match the original docs lock.

| Distribution | Version |
| --- | --- |
| sphinx-autobuild | 2024.10.3 |
| starlette | 0.50.0 |
| uvicorn | 0.40.0 |
| watchfiles | 1.1.1 |
| websockets | 15.0.1 |
| anyio | 4.12.1 |
| h11 | 0.16.0 |

All active base requirement edges of these seven distributions are checked
against the actual invoking environment. Installed package metadata and bundled
license-file hashes are retained in the external receipt. The installed notices
remain with the optional tools. The existing site renderer separately records
its 73-distribution closure, constraint-extra normalization and Darwin-only
appnope addition. The widget and conversion distributions satisfy the original
unconstrained direct inputs, with their existing installed versions recorded.

Q-0104 remains open for the complete scientific docs.env.txt lock. Its 194 pins
include runtime and interoperability packages beyond the documentation tools.
The nine-entry declaration acceptance and the measured renderer subset retain
their own exact scopes. The original static API generator's setter refusal,
full-site content warnings and pending protected extensions remain in the main
ledger. Quantum primitive counts, measurements and workspace do not apply to
this dependency declaration. Existing algorithm execution and separate measured
primitive/model evidence remains with the accepted domain owners.


## Batch verification

The concentrated regression passes 129 cases with 899 retained parser and
package-walk warnings. It includes the complete format, imported-source typing,
protobuf build, local live rebuild, notebook kernel/export, Sphinx site,
template and static reference owners. The integration regression passes all 29
cases. Strict typing covers all three changed test files. Black, Ruff,
Integration ownership, architecture, IR, protected API, local documentation
links and whitespace checks pass. The full scientific lock and preceding
whole-site content-warning boundaries retain their recorded scopes.
