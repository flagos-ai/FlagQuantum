# Incremental coverage wrapper source closure

## Complete sources and native ownership

This batch covers the complete source parents Q-0031
check/pytest-and-incremental-coverage, Q-0033
check/pytest-changed-files-and-incremental-coverage and Q-0047
dev_tools/conf/.coveragerc at immutable Qualtran v0.7.0
bcb17bb25360ed68ce3bc084645db3e929799389. Every original Git blob is verified.
The native command owner is tools/development/coverage_command.py and the native
profile is tools/development/coverage.ini. The profile's entire parsed content
matches the original run section, with include = ./* and no other settings.
Comments describe the native checkout and retain source attribution.

Existing pytest_command owns package test invocation, changed_tests owns commit
defaults and filename-associated test selection, PreparedEnvironment owns changed
path queries, process owns child execution and incremental_coverage owns touched
line analysis. The new owner composes these capabilities. Existing repository
coverage thresholds and CI selection retain their current owners.

```bash
python -m tools.development.coverage_command package HEAD -n0 -q
python -m tools.development.coverage_command package HEAD --no-analyze -n0 -q
python -m tools.development.coverage_command changed HEAD
```

## Arguments, baselines and failure order

Both modes consume a first nonempty non-option argument as the comparison commit.
Otherwise upstream/main, origin/main and main are tried in that order. Invalid
commits and missing defaults fail before invoking tests. The selected commit is
compared at its merge base with HEAD, matching these two source wrappers and
remaining distinct from ordinary changed-file pytest's direct comparison.

Package mode forwards all remaining arguments to pytest except --no-analyze.
Repeated --no-analyze options are removed and disable annotation conversion,
analysis and annotation cleanup. Package pytest still records coverage. With
analysis enabled, pytest runs first, then coverage annotate, then the accepted
touched-line analyzer, then Git annotation cleanup. The original ignores the
annotation-conversion exit status. This behavior is retained explicitly; an
annotation failure alone does not make the command fail if pytest and the
analyzer succeed. Pytest failure still proceeds through analysis and cleanup.
The final result is zero only when pytest and the analyzer both return zero.

Changed mode retains the source's first-revision-only argument handling. Other
arguments, including --no-analyze, are ignored as in the source. It uses existing
filename-associated tests, including the previously accepted native centralized
test layout adaptation. Coverage targets are sorted unique existing parent
directories of changed Python paths, excluding _pb2.py paths from coverage
targets. The source still selects a corresponding generated-file test if one
exists. No associated tests returns zero before pytest, analysis or cleanup.
Changed pytest itself emits annotations. It is followed by the same analyzer
and cleanup. Invalid revisions fail in either mode.

The original no-common-ancestor edge is preserved. An empty merge-base result
causes changed mode's failed diff pipeline to yield no tests and return zero.
Package mode retains that empty revision for its analyzer; --no-analyze can
therefore succeed on unrelated histories. Tests execute this behavior in the
unchanged original script and retain the result as a source limitation. It is
not used as evidence that unrelated histories were meaningfully analyzed.

## Cleanup, process bounds and layout adaptations

Cleanup uses the exact source Git pattern, git clean --force --quiet -x --
*.py,cover. It removes untracked annotation files, including ignored annotations,
and preserves tracked annotation files. As in the source, this includes existing
untracked annotations matching the pattern. Coverage data and unrelated files
remain present. A nonzero cleanup status is not part of the original final
pytest/analyzer aggregation. Native analyzer timeouts propagate, with cleanup in
a finally block so generated annotations are still removed. Test subprocesses
retain the accepted 1200-second limit. Annotation and analysis commands use
120-second limits, and Git commands use 15-second limits.

Native module commands use the current Python interpreter. Package tests use
the already accepted tests/ mapping. The coverage config path maps to the native
profile, and the analysis script path maps to the existing module entry point.
Native informational errors omit source ANSI color. The unchanged source code
is never edited for comparison.

## Evidence and unresolved scope

Actual original shells run against real Git trees with boundary executables
that record full child argument sequences and controlled exits. Native calls
record the corresponding child sequence on the same trees. The comparisons
cover all combinations of successful/failed pytest, annotation conversion and
analysis, flag forwarding, repeated --no-analyze, default revisions, changed-only
ignored extras, empty selection, protobuf coverage exclusions, divergent
merge-base comparison, invalid revisions and unrelated-history behavior.
Tracked and untracked annotation cleanup are exercised with real Git.

The native module also runs real pytest-cov, coverage annotation and the actual
touched-line analyzer in temporary checkouts. A covered changed line succeeds,
an unexecuted changed function fails, both scopes clean generated annotations,
and --no-analyze retains coverage data without making a touched-line claim.
The test harness disables unrelated environment plugins and loads the required
pytest-cov and xdist plugin modules explicitly.

This is classical development orchestration. Existing quantum amplitude,
measurement, workspace and measured-primitive/model evidence is reused. This
batch does not re-execute the complete upstream library or resolve the recorded
full default-suite timeout, Algorithms NumPy ownership or preceding-execution
copyright interaction. Q-0023 and Q-0025 through Q-0029 remain separate source
parents. Pending Proposals 116, 117 and 118 remain separate contract work.

The concentrated regression passed 366 tests with 914 visible warnings and zero
skips. Strict mypy passed both changed Python files. Black, Ruff, Integration
scope, architecture, IR, API and whitespace checks passed. The runtime profile
comparison records only a source inside the checkout and excludes an executed
external source for both unchanged original and native configurations. The first
real-CLI harness run loaded the pytest-cov package instead of its plugin module;
loading pytest_cov.plugin explicitly corrected that harness error. No production
behavior or reference source was changed to address it. The final suite includes
all 26 coverage-command tests and existing real Jupyter execution checks.
