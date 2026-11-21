# Development helpers

This package owns buffered generated-file output, development command logging,
checkout discovery and comparison revisions. File, process, repository and
coverage helpers use only Python's standard library. Docstring rendering uses
optional Sphinx and attrs development dependencies.
Framework, Compiler, Runtime and Simulation packages must not import it.

## Layout and dependency ownership

Developer helpers run from the repository checkout. Import them with
`from tools.development import process`, or invoke documented commands with
`python -m tools.development.<command>`. This internal namespace needs no separate
developer-package installation. The framework wheel and algorithm workflows have
their own package installation contract.

The complete developer README from Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389` (Q-0035) maps to these native owners.

| Original responsibility | Native location and use |
| --- | --- |
| Developer scripts, modules and configuration | This directory holds internal source maintenance, documentation and analysis tools. |
| `conf/` tool configuration | Root `pyproject.toml` and `.pre-commit-config.yaml` own native policy. Sibling `coverage.ini`, `retained_source_lint.ini` and `retained_source_mypy.ini` own their documented source-audit scopes. |
| Runtime and additional developer dependency declarations | Root `pyproject.toml` owns production requirements and optional development extras. Optional source-audit subsets and frozen versions are recorded in each owner audit. |
| Pinned environment specifications | The external `ftqc-reference-env` workspace holds unchanged source locks and isolated audit targets. Auditing a subset leaves whole-lock acceptance with its separate ledger parent. |
| Reference-generation templates | `templates/` contains native documentation assets consumed by `reference_templates.py` and `reference_template_pages.py`. |
| CLI scripts and reusable developer library installation | Commands and reusable helpers share the source-checkout `tools.development` namespace. Examples below show direct imports and module invocation. Framework-scale analysis is optional for algorithm users. |

The original separate editable developer distribution is represented by the
existing checkout namespace. This is an explicit installation-layout adaptation.
All dependency, template, command and library responsibilities remain documented
at their native owners. The source README's dependency declarations describe the
original project, while the root native manifest governs this repository.

## First contributor change

Scripts import `WriteIfDifferent` from `tools.development.files`, `run` and
`output_of` from `tools.development.process`, or `get_git_root` from
`tools.development.repository`. Supply a timeout for commands that may stall.

```python
from pathlib import Path
from tools.development.files import WriteIfDifferent

with WriteIfDifferent(Path("report.txt")) as output:
    output.write("Current report\n")
```

Run `python -m pytest tests/unit/test_development_tools.py` after a change.
Tests compare the complete frozen reference functions and source failure edges.
Source comparison requires the locked G2 reference environment. The file writer
preserves the source behavior of writing partial content when a context raises.

`PreparedEnvironment` in `environment.py` records actual/base revisions and the
checkout directory. `incremental_coverage.py` provides the source-equivalent
optional annotated-coverage report. Run the report inside that checkout, and
pass a concrete destination directory. It preserves the frozen ignore profile
and indentation heuristics. The existing repository coverage policy remains its
own authority. See `docs/development/DEVELOPMENT_COVERAGE_FILE_AUDIT.md` for the
source limits and the explicit 15-second Git-query timeouts.

`docstrings.py` renders Google-style class documentation and structured references.
Its reference environment uses Sphinx 7.4.7 and attrs 26.1.0. Install these only
for documentation tooling. Run `tests/unit/test_development_docstrings.py` in the
locked G2 environment to compare both original tests and all 340 frozen classes.
See `docs/development/DEVELOPMENT_DOCSTRINGS_FILE_AUDIT.md` for dependency review,
replacement rules and the preserved source parsing boundaries.

`component_discovery.py` discovers Python classes and existing Testing example
records. Supply a package prefix for an explicit source root and a domain class
predicate for class discovery. For a first example, call `module_examples` with
`Path("component_example_catalog.py")` and `package="examples.algorithms"`.
Discovery leaves the example factory unexecuted. The default root is the tracked
native algorithms package. Both native and original test filename conventions
are recognized. Run `tests/unit/test_development_discovery.py` to verify the
complete source finder and native catalog paths.

`library_call_graph.py` consumes lazy example records and delegates a shared
weighted graph traversal to Compiler. Supply the domain's declaration callback,
terminal predicate and existing resource generalizers, or use
`resource_library_call_graph` for the built-in native resource policy. The result is native
adjacency, preserving isolated roots and shared nodes. Run
`tests/unit/test_library_call_graph.py` for the native profile and complete frozen
393-example graph comparison. See `LIBRARY_CALL_GRAPH_FILE_AUDIT.md` for the
concrete-root policy, fresh allocator behavior and model/physical distinction.

A small model-only library query can reuse the existing resource example.

```python
from examples.algorithms.resource_call_graphs import declared_example
from flagquantum.testing.component_examples import ComponentExample
from tools.development.library_call_graph import library_call_graph

root, declarations = declared_example()
example = ComponentExample(lambda: root, "declared_program", type(root))
graph = library_call_graph(
    [example], lambda node, symbols: declarations.get(node, {})
)
assert len(graph[root]) == 3
```

`component_report.py` builds, styles and summarizes the five-check component
report. Supply the domain's existing checker and class predicate. It reuses
Testing example metadata and status records, with no decomposition or numerical
logic in the tool. Pass a reusable list of examples for both report traversals.
The original one-shot iterable exhaustion and empty-input `KeyError` are recorded
in `COMPONENT_REPORT_FILE_AUDIT.md`. Explicit examples override the original two
slow-Hubbard-example exclusions.

For a runnable register-program report, use the existing three-X fixture.

```python
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.testing.component_examples import ComponentExample
from flagquantum.testing.example_checks import RegisterExample, check_example
from tests.unit.test_reference_testing_file import lower_fixture, parallel_example
from tools.development.component_report import get_report, summarize_results

fixture = RegisterExample(parallel_example, lower_fixture, lambda: {"x": 3})
example = ComponentExample(parallel_example, "parallel", ProgramRecord)
report = get_report(
    [ProgramRecord],
    [example],
    check=lambda example, kind: check_example(fixture, kind),
    is_component=lambda cls: cls is ProgramRecord,
)
assert summarize_results(report)["pass"].tolist() == [1] * 5
```

The fixture import is confined to this developer example. The reporting tool
itself has no test-package dependency. Run `tests/unit/test_component_report.py`
and `tests/unit/test_reference_testing_file.py` after changing this workflow.

`tensor_report.py` times the existing Compiler expansion callback, Simulation
node construction and pinned contraction-width planner. Supply `flatten` and
`build_nodes` explicitly and run `report_on_tensors` as a task with
`ProcessDeadlineExecutor` from `process_deadlines.py`. Tasks receive `cxn` and
send one complete result. Use the executor as a context manager. As required
by multiprocessing spawn, process-launching scripts must guard their main entry.

```python
from tools.development.process_deadlines import ProcessDeadlineExecutor


def report_value(cxn, value):
    cxn.send(value)


if __name__ == "__main__":
    with ProcessDeadlineExecutor(timeout=10, max_workers=1) as executor:
        executor.submit(report_value, {"value": "ready"})
        arguments, result = executor.next_result()
        assert result == "ready"
```

The complete Compiler-to-Simulation report example is
`tests/unit/test_tensor_report.py::test_report_uses_existing_compiler_and_simulation_in_worker`.
Run that file with the existing tensor-node, dense-component and width-policy
tests. See `TENSOR_REPORT_FILE_AUDIT.md` for the original blocking defects and
native lifecycle corrections. Width estimates and observed timing fields remain
separate from physical gate counts and algorithmic resource estimates.

`notebook_autogen.py` renders existing ComponentDocumentation records into a
NotebookSpec. Supply the domain's graphical-signature and call-graph Python
code callbacks. The tool merges cells by their stored cq.autogen ID, keeps
existing notebook cell IDs and authored cells, and delegates equal-content
writes to WriteIfDifferent. Use factories that assign the named example before
returning it. The source parser omits terminal return expressions.

For the shortest complete path, run
`tests/unit/test_notebook_tools.py::test_runnable_native_cells_use_append_and_existing_drawer`.
It generates and executes a WordSwap notebook using ordinary circuit.append(op),
the existing circuit Drawer and the same component's Compiler call graph.
`notebook_cleaning.clean_notebooks(Path("checkout"), do_clean=False)` reports
tracked notebook cleanup without writing. Set do_clean=True to apply it. Output
and execution counts are removed while generated-cell identifiers survive.

Run `tests/unit/test_notebook_tools.py` together with the existing docstring,
metadata/discovery and development-file tests after a change. See
`NOTEBOOK_TOOLS_FILE_AUDIT.md` for source parser limits and explicit fixes to the
original diff-status and temporary-file cleanup defects.

`notebook_execution.py` runs Git-tracked notebooks and exports executed notebook
and Markdown outputs beneath docs. Start with
`tests/unit/test_notebook_execution.py::test_real_spawn_batch_exports_and_aggregates_failures`.
Call execute_and_export_notebooks with explicit source roots for a focused
checkout and guard its caller with `if __name__ == "__main__"` for spawn.
Testing owns actual Jupyter execution. The tool owns stale-output selection,
ordered Markdown links, per-file atomic exports and batch failure summaries.
Rendering completes before destination writes. Each destination replacement
is atomic, and cross-file replacement can leave one updated destination. Markdown image resource files retain the original
exporter's separate-resource limitation. See NOTEBOOK_EXECUTION_FILE_AUDIT.md.

`reference_pages.py` owns internal module/class documentation records.
`reference_links.py` owns Markdown reference links and alias descriptions, while
`reference_context.py` supplies the current mappings to each writer. For a
minimal edit path, run
`tests/unit/test_reference_page_links.py::test_native_page_links_reuse_existing_equal_content_writer`.
It writes a native circuit documentation reference through WriteIfDifferent and
checks that an unchanged second render preserves the timestamp. Optional Griffe
and Markdown anchor dependencies follow the frozen documentation environment.
See REFERENCE_PAGE_LINKS_FILE_AUDIT.md for the cross-document source cache defect
and the remaining renderer/package-walker scope.

`reference_signatures.py` formats static Python call signatures from Griffe
objects, preserving source decorator and overload behavior.
`reference_docstrings.py` renders canonical Griffe Google-style sections and
reuses the accepted reference parser. Start with
`tests/unit/test_reference_signatures.py::test_native_signature_uses_existing_reference_link_writer`
for the full signature-to-linked-file path. Run the signature and docstring
files together with test_reference_page_links.py after edits. The full frozen
340-class docstring corpus is compared in test_reference_docstrings.py. See
REFERENCE_RENDERING_FILE_AUDIT.md for preserved source quirks and explicit native
object-name mappings.

`reference_class_pages.py`, `reference_module_pages.py` and `reference_toc.py`
render owned class/module pages and a callback-selected table of contents.
All three consume the accepted page/context, signature, docstring and link
owners. Start with
`tests/unit/test_reference_page_renderers.py::test_full_class_page_and_alias_members`
to render an append-method page and verify unchanged-file timestamps. The same
test file covers category grouping, actual inherited/alias members and TOC
failure cleanup. Package discovery and policy belong to reference_package.py.
See REFERENCE_PAGE_RENDERERS_FILE_AUDIT.md for full source closure and output
failure semantics.

`reference_package.py` owns Griffe traversal, alias qualification, page grouping
and link/context construction. `reference_docs.make_reference_docs` connects
those records to the accepted page renderers with native domain captions. Start
with `tests/unit/test_reference_package.py::test_complete_generation_outputs_and_new_destination_fix`
for a runnable package-to-files example and new destination creation. The same
file compares the complete frozen 41-page package, its 42 generated outputs and
concurrent ordinary/native parser behavior. The source's unordered submember
regions are compared by named content while every other byte remains exact.

The current whole-FlagQuantum walk succeeds. Full rendering retains the source
setter-decorator refusal at Circuit.n_qubits and can leave partial outputs.
See REFERENCE_PACKAGE_FILE_AUDIT.md for this explicit capability boundary,
module-cycle handling and per-visitor property normalization.

Reference tests import optional parsers at execution time. Ordinary test collection
with `-m "not ftqc_reference"` works without Griffe. Explicit reference selection
requires the locked documentation dependencies and fails if they are missing.

`copyright_checker.py` provides an opt-in Pylint plugin for retained Google
Apache-2.0 source notices. After installing the frozen optional Pylint environment,
run `python -m pylint --load-plugins=tools.development.copyright_checker --disable=all --enable=wrong-or-nonexistent-copyright-notice path/to/imported_source.py`.
A valid original notice produces status 0. An invalid or truncated notice produces
status 8. File-level Pylint opt-out follows the host's existing suppression rules.
Run `tests/unit/test_copyright_checker.py` to compare the original suite, exact raw
positions and actual plugin loading. See COPYRIGHT_CHECKER_FILE_AUDIT.md for the
source truncation defect, preserved year behavior and optional dependency review.

`reference_templates.py` owns the template-driven SDK generation entry.
`reference_template_pages.py` owns its runtime page plugins and per-page
formatting, and `reference_output.py` owns suffix repair, metadata cleanup and
Sphinx contents. The six original Jinja assets live in `templates/`. With the
locked optional TensorFlow-docs environment and this checkout as the current
working directory, generate the native loaded namespace profile with

```python
from pathlib import Path
from tools.development.reference_templates import build_reference_docs

build_reference_docs(Path.cwd())
assert Path("docs/reference/flagquantum/Circuit.md").is_file()
assert Path("docs/reference/autotoc.rst.inc").is_file()
```

Run `tests/unit/test_reference_templates.py` for all four real page categories,
complete frozen-package output comparisons, suffix/metadata/TOC handling and
concurrent ordinary/native formatting. Source alias routing and captions become
native paths and identity. REFERENCE_TEMPLATES_FILE_AUDIT.md records the measured
loaded profile, preserved formatting limits and isolated dependency procedure.

The accepted documentation workflows have ordinary module command entry points.
Run `python -m tools.development.notebook_cleaning` to check tracked package
notebooks, or add `--apply` to clear output and metadata. Its exit status remains
the number of dirty notebooks, including after applying cleanup. Run
`python -m tools.development.notebook_execution --n-workers 1 --output-md` to
execute tracked package/example notebooks and export Markdown with the existing
notebook output. Use `--no-output-nbs` to disable notebook export and
`--no-only-out-of-date` to force execution.

`python -m tools.development.reference_templates` runs the loaded-namespace
reference workflow. `python -m tools.development.reference_docs` invokes the
static Griffe workflow, whose inherited-setter refusal remains explicit. Both
write to the current checkout's reference directory and propagate owner errors.
Run `tests/unit/test_documentation_commands.py` with the locked documentation
environment for exact original parser comparisons and actual subprocess usage.
See `DOCUMENTATION_COMMANDS_FILE_AUDIT.md` for full source dispositions and reused
owner evidence.

`python -m tools.development.incremental_coverage HEAD~1` checks touched Python
lines against a comparison revision using the existing coverage annotations.
Missing annotations conservatively count relevant changed lines as uncovered.
The command exits zero only when the owner's uncovered-line count is zero.
Run `tests/unit/test_incremental_coverage_command.py` and
`tests/unit/test_development_coverage.py` for the complete source entry and real
Git-checkout workflow. See `INCREMENTAL_COVERAGE_COMMAND_FILE_AUDIT.md` for source
argument and diagnostic semantics.

`notebook_catalog.notebook_sections()` returns the explicit 110-notebook library
publication catalog in its 11 source section orders. It resolves authored native
assets under the current checkout's `examples/notebooks`, or a supplied build
root. Missing paths stay visible to the caller. The catalog does not import or
execute numerical examples. Each original notebook title, path override and
component-document reference has an explicit audited native destination.
Run `tests/unit/test_notebook_catalog.py` to check path ownership and assets, or
use the locked reference environment for the original 110-spec/237-document
comparison and native module-symbol resolution. See `NOTEBOOK_CATALOG_FILE_AUDIT.md`
for rendering boundaries and the separately open top-level orchestration.

`python -m tools.development.notebook_publication` publishes the explicit native
library into `docs/components`, retaining authored notebook text and generating
its ordered Sphinx index. The 110 topic registrations and 12 concept/root
references resolve to 120 distinct outputs. Unregistered native notebook assets
are reported and remain outside the explicit catalog.

For generated metadata, `publish_notebooks(..., render=...)` invokes a domain
callback on registered topics before publication. That callback can call the
existing `notebook_autogen.render_notebook` with its owned NotebookSpec and
rendering functions. The default publishes the already authored native assets.
Run `tests/unit/test_notebook_publication.py` for the complete source-order/TOC
comparison, actual full-native publication and renderer integration. See
`NOTEBOOK_PUBLICATION_FILE_AUDIT.md` for the explicit metadata-to-asset adaptation.

`python -m tools.development.notebook_publication_check` runs that publication
command and requires an empty Git status for the entire checkout. Generator
failure bypasses the status query and retains its exit code. New generated
outputs and unrelated working changes fail the check until the checkout owner
has supplied the intended baseline. The check preserves the current ignore
policy. Run `tests/unit/test_notebook_publication_check.py` for the actual
unchanged source shell comparison and native full-library workflow. See
`NOTEBOOK_PUBLICATION_CHECK_FILE_AUDIT.md` for process bounds and literal-path
handling.

`python -m tools.development.documentation_cleanup docs/components` removes
generated notebooks, Markdown and checkpoints from the explicit owned directory.
It accepts `docs/reference/flagquantum` for generated package reference pages or
an external generated-only staging directory. Other in-checkout paths are
refused, including the authored `docs` root. Unrelated RST, HTML and images
remain present. The native staged reference TOC spelling is recognized alongside
the original spelling. Run `tests/unit/test_documentation_cleanup.py` for the
complete unchanged source comparison, real 120-notebook publication cleanup and
CLI ownership refusals. See `DOCUMENTATION_CLEANUP_FILE_AUDIT.md` for the
adaptation of the original docs-branch assumptions.

`examples/notebooks/component_bibliography.ipynb` composes existing class discovery
and reference parsing into a development report. Run its cells to display owned
algorithm-class references, parsing failures and URL backreferences. Python
module and qualified class names label native backreferences. The development
notebook remains outside the scientific publication catalog. Run
`tests/unit/test_component_bibliography_notebook.py` for complete unchanged-source
cell comparisons and actual native Jupyter execution. See
`COMPONENT_BIBLIOGRAPHY_NOTEBOOK_AUDIT.md` for the explicit discovery adaptation.

`examples/notebooks/component_report.ipynb` runs the existing five checks on a
real two-bit WordSwap Program and labels results by native package owner. It
prints the six measured CX primitives separately from the six declared CX
primitives. Domain views, styled check results and full summaries reuse the
report owner. Run `tests/unit/test_component_report_notebook.py` for actual
Jupyter execution, exact complex-state append behavior, deliberate count failure
and unchanged-source view comparisons. See `COMPONENT_REPORT_NOTEBOOK_AUDIT.md`
for the preserved source Other/factoring query counterexamples and explicit owner
labels.

`python -m tools.development.pytest_command package -n0 -q` runs native tests.
Use `development` for the development-tools and complete native unit-test
directories. Both retain the source's initial automatic xdist flag and forward
caller flags, including the final -n0 override. `changed HEAD -q` selects existing
filename-associated tests against an explicit revision. Default revisions retain
the source upstream/main, origin/main, main order. Native centralized unit-test
names supplement the original same-directory suffix rule. This selection does
not infer dependencies. Run `tests/unit/test_development_pytest_commands.py` for
all three unchanged shell sources, real Git selection and actual native module
commands. See `DEVELOPMENT_PYTEST_COMMANDS_AUDIT.md` for scope mappings, direct
revision semantics and process bounds.

`python -m tools.development.coverage_command package HEAD -n0 -q` runs native
package tests and then checks annotated touched-line coverage against the merge
base. Add `--no-analyze` to retain coverage collection while omitting analysis
and annotation cleanup. `changed HEAD` selects existing associated tests and
their changed Python directories. The changed form retains the source's ignored
extra arguments. Analysis cleans untracked `*.py,cover` artifacts using the
original Git pattern, including pre-existing ignored annotations. Run
`tests/unit/test_development_coverage_commands.py` for complete unchanged-shell
comparisons and real native pytest-cov/annotation/analyzer workflows. See
`DEVELOPMENT_COVERAGE_COMMANDS_AUDIT.md` for failure order, the original ignored
annotation-conversion status, unrelated-history behavior and native path mappings.

`typecheck_command.py` runs the package or complete developer-helper typing scope
under the existing native strict policy. Use
`python -m tools.development.typecheck_command development` or replace
`development` with `package`. Extra mypy arguments pass through in order.
The developer scope supports the `tools` namespace with explicit package bases.
The explicit `retained-source path/to/source.py` mode uses the complete imported
profile and optional locked Duet/NumPy plugins. Supply explicit targets and the
isolated audit environment. See `DEVELOPMENT_TYPECHECK_COMMANDS_AUDIT.md` for
wrapper comparison and `DEVELOPMENT_MYPY_PROFILE_AUDIT.md` for full configuration,
plugin behavior, dependency setup and the separate native strict policy.

`format_command.py` preserves the complete changed/all Python file selection and
check/apply orchestration of the frozen formatter. Run
`python -m tools.development.format_command HEAD`, adding `--apply` to modify
selected files or using `--all` for tracked files. Optional Flynt 0.78 owns
f-string conversion. Existing Ruff import sorting and Black own native source
formatting. See `DEVELOPMENT_FORMAT_COMMANDS_AUDIT.md` for dependency setup,
configuration differences and retained source edge cases.

`lint_command.py` runs the native package or changed-file lint scopes using the
existing Ruff policy. Use `python -m tools.development.lint_command changed HEAD`
for a first comparison. Changed mode retains first-revision-only arguments.
The explicit `retained-source` mode accepts paths and Pylint flags under the
complete imported-source profile, including the existing optional copyright
checker and documentation-parameter checks. It requires the accepted optional
Pylint environment. See `DEVELOPMENT_LINT_COMMANDS_AUDIT.md` for ownership,
complete profile mapping, source defects and diagnostic comparisons.

`check_command.py` composes accepted lint, strict package typing, incremental
formatting and coverage owners. Run
`python -m tools.development.check_command HEAD --only-changed-files` for changed
lint and coverage scopes, or omit the flag for package scopes. Use
`--apply-format-changes` to pass `--apply` to the formatter. Ordinary failure in
any phase makes the final command fail after all four phases run. See
`DEVELOPMENT_CHECK_COMMANDS_AUDIT.md` for the original false-success and argument
forwarding defects and the explicit native corrections.
