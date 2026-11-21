# Development helpers

This package owns buffered generated-file output, development command logging
and checkout-root discovery. File, process, repository and coverage helpers use
only Python's standard library. Docstring rendering uses optional Sphinx and attrs
development dependencies.
Framework, Compiler, Runtime and Simulation packages must not import it.

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
