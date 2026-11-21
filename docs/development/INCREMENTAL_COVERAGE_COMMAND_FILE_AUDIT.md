# Incremental coverage command source closure

## Complete source and ownership

Q-0044 covers the complete check-incremental-coverage-annotations.py script at
Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. Its Git blob is verified
against the immutable inventory. Original Google Apache-2.0 source notices are
retained. The source entry is now the existing Tools incremental_coverage
module's __main__ path. PreparedEnvironment owns revision context, process owns
terminal highlighting and bounded Git commands, and the existing coverage
functions own line selection and annotations.

The normal command uses a comparison revision and the current checkout.

```bash
python -m tools.development.incremental_coverage HEAD~1
```

The command does not run tests or produce coverage annotations. It consumes the
owner's existing annotations, or conservatively counts relevant changed lines
as uncovered when an annotation is absent. Repository coverage policy remains
independently authoritative.

## Complete source comparison and actual execution

The unchanged script executes its __main__ entry with only its imported owners
replaced. Missing arguments produce the exact original colored diagnostic and
status one before any environment or coverage call. The first argument becomes
the comparison revision, actual_commit_id is None, and destination_directory is
the current working directory. Extra arguments are ignored, matching the source.
The existing origin/master example text is retained as part of exact diagnostic
comparison. Native documentation uses HEAD~1 for the current repository.

Zero uncovered lines succeed, while positive or negative truthy counts produce
status one. Errors from the coverage owner propagate. Tests compare exact owner
arguments, stdout and exit semantics. Native main also exercises ordinary
sys.argv input. Importing the module does not execute it.

A real temporary Git checkout runs the module as a subprocess. An unchanged
working tree succeeds, a changed unannotated Python statement fails, an explicit
uncovered annotation fails, and a covered annotation succeeds. Missing arguments
and an invalid revision fail through their respective original boundaries.
Git fixture author and committer use FlagQuantum's configured identity. Every
subprocess has a timeout.

## Reused evidence and remaining boundaries

The complete environment and coverage source comparisons remain covered by
DEVELOPMENT_COVERAGE_FILE_AUDIT.md and test_development_coverage.py. Original
ignore patterns, indentation limitations, line numbers, annotation behavior,
actual versus local revisions and deleted-file handling retain their evidence.
This is classical developer tooling. Existing quantum amplitudes, measurement,
workspace and separate primitive/model resource acceptance are reused and no
quantum gate count applies.

The broader default smoke/unit run timed out at its 1,200-second limit after
48 percent, with existing Algorithms NumPy and copyright collection-order
failures observed. This command batch does not certify the full default suite.
Pending Proposals 116, 117 and 118 and notebook catalog Q-0080 remain open.

## Verification

The command and its complete direct owner regression pass 288 tests with zero
skips and 903 visible upstream warnings. Strict mypy covers the two changed
Python files. Black, Ruff, Integration scope, architecture, multi-level IR,
protected API and diff checks pass. All subprocess probes remain bounded.
A full 98,035-item collection followed by selecting only the copyright original
suite passes that test, with 51 optional collection skips and 98,034
deselections. Thus collection alone does not reproduce the earlier full-run
failure. Its preceding-execution interaction remains unresolved, with evidence
in g2-copyright-full-collection-probe-20261121.log.
