# Complete incremental formatter source audit

## Source and ownership

The complete frozen `check/format-incremental` Q-0025 at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389` maps to
`tools/development/format_command.py`. Existing process and repository helpers
own bounded child execution and checkout discovery. The repository owner
now also hosts the accepted commit validation, ordered defaults and merge-base
comparison helpers. Existing changed-test and coverage imports remain available
through explicit re-exports. The formatter has no coverage-command dependency. The complete
original shell file runs unchanged in temporary Git checkouts.

```sh
python -m tools.development.format_command HEAD
python -m tools.development.format_command HEAD --apply
python -m tools.development.format_command --all
```

Integration owns this development command. It supports the existing
`circuit.append(op)` workflow through source maintenance. Framework execution
packages have no dependency on the formatter. The existing repository Ruff and
Black configuration remain the native policy authorities.

## Selection and execution closure

The command retains optional revision, `--apply` and `--all` argument behavior,
including repeated flags, empty revision handling and invalid extra arguments.
Default revisions are `upstream/main`, `origin/main`, then `main`. Changed-file
selection compares to the common ancestor and uses the original modified,
added and renamed diff filter. It excludes nested `protos/*.py`. All-file
selection uses tracked Python paths and excludes `*_pb2.py`. These two generated
file exclusions intentionally differ in the source and are retained. Untracked
files are excluded. Spaces in tracked filenames pass as single arguments.

All-file selection can include a tracked file removed from the working tree.
The command forwards that path and retains formatter errors, as the source does.
An unrelated comparison history produces an empty merge base and an invalid Git
diff followed by a zero-file success in the original script. The native command
retains that source defect. Such success does not establish a useful comparison.

The three formatting phases always run in order after ordinary nonzero statuses.
Any nonzero formatting status produces a final status of one. Black version
query failure is ignored in the source and excluded from that aggregation.
Flynt `-f` means fail-on-change. The source retains it in apply mode, so a successful f-string rewrite can still yield final status one. This contradicts the shell usage comment promising zero after applying. Native execution preserves the actual behavior and a clean recheck succeeds.

An empty selection succeeds without launching a formatter. Git/version queries
are bounded at 15 seconds and each formatting child at 1200 seconds. Native
subprocess exceptions propagate. Normal failure codes retain source aggregation.

## Tool and configuration dispositions

| Source phase or dependency | Native disposition |
| --- | --- |
| Flynt `--quiet -f` and optional `--dry-run` | Same Flynt invocation through the current Python interpreter. The fail-on-change flag retains source exit behavior. |
| isort and optional `--check --diff` | Existing Ruff import sorting with `check --select I --diff`, or `--fix` when applying. No second native import sorter is installed. |
| Black `--color` and optional `--check --diff` | Same Black invocation through the current interpreter using existing native configuration. |
| `deps/format.txt` Q-0096 | Complete file inspected and blob-verified. Flynt is an optional development dependency. Native sorting uses existing Ruff. Dependency manifest parent remains separate. |
| `envs/format.env.txt` Q-0105 | Complete frozen lock inspected. Acceptance environment uses its Flynt 0.78, astor 0.8.1, isort 5.10.1 and Black 24.8.0. Runtime dependencies in this lock are outside this formatter batch. |
| Root `pyproject.toml` Q-0137 | Complete source file inspected and blob-verified. Native formatting continues using the existing root configuration. Other pytest and packaging settings remain separately owned. |

Source Black uses width 100, single-quote preservation and disabled magic trailing
commas. Native Black uses its existing width 88 policy. Source isort also has its
own ordering and alias rules. These policies can produce different source text.
Actual formatting tests retain a concrete source/native textual difference and
verify equal executed string values. General isort/Ruff or Black-profile output
identity is not an acceptance claim. Original shell orchestration, selection and
failure aggregation are compared independently from these policy adaptations.

## Optional dependency review

Flynt fills the original f-string conversion responsibility, which the existing
native configured Ruff rules do not provide. Flynt 0.78 and astor 0.8.1 are
installed only in the isolated development acceptance environment. For this
optional command, make Flynt available to the invoking Python interpreter,
for example with `uv pip install --python <python> flynt==0.78 astor==0.8.1`.
Missing Flynt is a visible child failure and contributes to the final failure.
The command then continues its remaining formatting phases, matching the source
failure sequence. There is no dependency substitution on error.

Flynt, Black and isort use MIT licenses. Astor uses BSD-3-Clause. The installed
license files were inspected and their hashes and versions recorded in
`g2-development-format-dependencies-20261121.json` in the reference environment.
The isolated index installation was explicitly version-pinned. Only the original
comparison uses isort. Integration owns the optional Flynt subprocess boundary,
which can be replaced after the same conversion and failure evidence passes.
Source conversion operates on local files and belongs exclusively to developer
tooling. The license/provenance review is limited to this boundary.

## Verification and remaining boundaries

Tests verify source and direct configuration blobs, execute all eight formatter
status combinations in both check/apply modes, retain full original command
order, flags and path selection, and exercise actual divergent and unrelated Git
histories. Real native CLI execution checks no-write mode, applied f-string and
import conversion, a clean recheck, and an original/native semantic round trip.
Strict typing, formatting and architecture checks cover the changed Python files.

Existing quantum amplitude, measurement, workspace and measured-primitive/model
evidence is reused for this classical tooling batch. Q-0049 typing profile,
Algorithms NumPy ownership, preceding-execution copyright interaction and the
broad default-suite timeout remain separate open work.

Final concentrated regression passed 409 tests with 914 visible warnings and zero
skips. Strict mypy passed all five changed Python files. Black, Ruff, Integration
scope, architecture, IR, API and diff checks passed. The shared-query ownership
change and final version-query failure case were included in this final run.


## Complete formatter dependency input

Q-0096 is now accepted as the complete three-entry formatter declaration.
The frozen input contains Flynt ~=0.60, Black ~=24.8.0 and isort ~=5.10.1 in
that order. The actual source-comparison tools are Flynt 0.78, Black 24.8.0 and
isort 5.10.1. All three satisfy their original constraints, retain installed
license files and execute through the original formatter command evidence.
Astor remains available as Flynt's conversion dependency. Complete input and
installed direct-dependency checks run alongside the existing actual format
and semantic round-trip tests.

Native ownership remains the existing format_command process boundary. Flynt
retains conversion, while native Black and Ruff import sorting follow the
existing root pyproject development policy. This explicit configuration and
sorter adaptation is already measured in the formatter tests. The optional
source-comparison environment supplies the original isort and Black versions.
The production and native development dependency authority remains the root
project metadata. Full source format.env.txt includes the scientific runtime
and retains its independent Q-0105 acceptance scope.
