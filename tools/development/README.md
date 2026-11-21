# Development helpers

This package owns buffered generated-file output, development command logging
and checkout-root discovery. It depends only on Python's standard library.
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
