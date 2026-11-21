# Complete notebook catalog source closure

## Source scope and native ownership

Q-0080 covers the complete notebook_specs.py at Qualtran v0.7.0
bcb17bb25360ed68ce3bc084645db3e929799389. The unchanged module declares 11 ordered
sections, 110 NotebookSpecV2 instances and 237 BloqDocSpec references. All module
imports, title strings, directory overrides, path_stem overrides and shared
references execute in the original baseline. Its Git blob and each referenced
original notebook's blob are verified against the immutable inventory.

The native catalog lives in tools/development/notebook_catalog.py. It explicitly
registers 110 existing native notebooks in the original section and notebook
order. Existing authored notebooks own the narrative, examples, graphical
signatures, resource queries and append workflows. Tools owns discovery of their
publication paths. Runtime generation of component documentation remains with
the existing NotebookSpec and renderer. Registering an authored asset does not
create a second component type or run its code during import.

The frozen source combined catalog registration, live Python imports, notebook
path construction and implicit metadata lookup. Native separates the publication
catalog from numerical/example imports. notebook_sections resolves the explicit
catalog under the current checkout's examples/notebooks directory, or an explicit
build directory. It preserves missing paths so a build can identify gaps. It does
not scan arbitrary notebooks into the publication list.

## One-to-one mapping and validation

The fixture tests/fixtures/notebook_catalog_source_mapping.json records each
original title, module, resolved source notebook, every original document class
and its native notebook destination. This fixture is attribution and audit data,
not a runtime dependency on Qualtran. Production registration uses native asset
names. All 110 native paths are unique. Every original notebook already has an
accepted parent-file disposition in the ledger.

An actual import of the unchanged source instantiates all 110 specifications.
The comparison checks every field, full path, 237 ordered document references
and all 11 groups against the explicit fixture. Every original reference notebook
is read and Git-blob verified. This covers the source's otherwise easy-to-miss
custom stems and output-directory overrides. Native path construction is tested
with both current-checkout discovery and explicit relative/absolute roots.

All native notebooks are parsed as nbformat4 JSON. Each has a level-one narrative
title and code cells. Every code cell is parsed as Python, and every native
from-import module and symbol resolves in the locked environment. Direct
Qualtran and Kickmix imports are rejected. Existing execution and scientific
acceptance evidence is reused from the 110 accepted notebook parents rather
than claiming that AST/import validation executes all notebooks again. The
Apply-to-Lth notebook starts with a code cell before its title, a valid authored
ordering that remains unchanged. Existing non-raw LaTeX escape warnings are
visible and retained.

## Rendering boundary and remaining work

Q-0036's top-level render-all, contents generation and missing-document report
remain open as a separate orchestration source file. Its 11 concept notebooks
and one root notebook also require explicit mapping before that source is
accepted. Q-0024's clean-tree shell check remains separate. This catalog does not
rewrite authored notebooks using inferred metadata or substitute a discovery
scan for the explicit 237-reference source list.

This is classical documentation metadata. Complete complex amplitudes,
measurement branches, work-bit cleanup and separate measured primitive/model
resource accounts are inherited from the accepted notebook owners. Publication
path registration emits no quantum gates. The broader Algorithms dependency and
copyright preceding-execution failures remain explicit, and the full default
smoke/unit suite remains unverified after the bounded 1,200-second run.

## Verification

Four catalog tests pass in the locked environment. The concentrated catalog,
reference generation, notebook execution, discovery and coverage owner suite
passes 292 tests with zero skips and 917 visible upstream/parser warnings.
Strict mypy covers the two new Python modules, and Black, Ruff, Integration
scope, architecture, multi-level IR, protected API and diff checks pass.
The JSON mapping asset separately passes Integration scope. No existing native
notebook is rewritten by this catalog batch.
