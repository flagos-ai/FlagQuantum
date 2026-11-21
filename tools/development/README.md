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
