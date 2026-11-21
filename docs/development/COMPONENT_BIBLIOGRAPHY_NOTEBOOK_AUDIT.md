# Component bibliography notebook source closure

Q-0037 covers the complete dev_tools/bibliography.ipynb notebook at immutable
Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. Its original Git
blob is verified against the inventory. The native authored notebook is
examples/notebooks/component_bibliography.ipynb. It reuses the accepted
component_discovery and docstrings modules. No bibliography parser, component
registry or domain superclass is added.

Open the notebook in the locked development environment and run its cells in
order. It discovers public classes owned by the native algorithms package,
extracts their existing References sections, displays references grouped by
class, reports unparsed references and groups backreferences by URL. Empty
reference lists remain silent. The displayed backreference groups retain the
source count ordering and sorted names within each group. The source's final
title for repeated URLs remains authoritative, including its unresolved title
disagreement TODO.

Every source code cell is preserved in order with explicit namespace and native
class-discovery substitutions. The native domain predicate accepts its owned
public classes without requiring a shared Bloq base. The source-specific
_class_name_in_pkg_ method maps to the existing Python module and qualified
class name. This is a native class metadata report. It does not add a discovery
model for procedural operations or claim an identical native/upstream class
population. The notebook is a development aid and remains outside the explicit
110-topic scientific publication catalog.

The entire unchanged source notebook code executes with only the discovery and
reference-parser imports replaced at their accepted boundaries. The native code
executes over the same class population. Reference dictionaries, titles,
backreference membership, Markdown display strings and parsing-error text match.
Fixtures cover shared URLs, unique URLs, unparsed references and an empty class.
The source metadata and code-cell ordering are checked, and native output cells
are committed empty.

The real native notebook also executes in a Jupyter kernel with actual Git-backed
class discovery and existing parser imports. It displays the KXorInstance
reference and its shared paper backreference. This execution is a metadata
workflow. It does not run scientific examples or make gate-count claims. Existing
quantum amplitude, measurement, workspace and measured-primitive/model evidence
remain with their accepted owners. Broad default-suite dependency and prior
execution interaction failures remain open.

The concentrated regression passed 314 tests with 916 visible warnings and zero
skips. Strict mypy passed the changed test file. Black, Ruff, Integration scope,
architecture, IR, API and whitespace checks passed. All native notebook Python
cells compile, and actual kernel execution passed. The notebook retains the
source's interactive coding style; the strict-check result applies to its test
harness and is not a claim of whole-notebook static type certification.
