# Complete lint command and retained-source profile audit

## Frozen source closure and native ownership

The complete source parents are Q-0028 `check/pylint`, Q-0029
`check/pylint-changed-files` and Q-0048 `dev_tools/conf/.pylintrc` at Qualtran
v0.7.0 `bcb17bb25360ed68ce3bc084645db3e929799389`. All original Git blobs are
verified. The unchanged original shell files execute in temporary Git checkouts.
The native owners are `tools/development/lint_command.py` and
`tools/development/retained_source_lint.ini`.

Existing repository helpers own root discovery, commit validation, ordered default
branches and merge-base comparison. Existing process helpers own child execution.
The accepted copyright checker owns the retained imported-source notice protocol.
Existing root Ruff configuration and pre-commit workflow own native project lint
policy. No framework execution package imports this development tooling.

```sh
python -m tools.development.lint_command package
python -m tools.development.lint_command changed HEAD
python -m tools.development.lint_command retained-source --jobs=1 path/to/source.py
```

The changed command on a clean tracked checkout prints zero lintable files and
returns zero. Package mode reports the current native Ruff findings and exact
Ruff status. Retained-source mode audits explicitly supplied paths using the
complete imported-source Pylint profile. These tools support source maintenance
for the existing `circuit.append(op)` workflow.

## Complete wrapper semantics

Package mode forwards arguments before the two native target paths. Source
`qualtran/` and `dev_tools/` targets map to `flagquantum/` and
`tools/development/`. Changed mode consumes only its first nonempty non-option
argument as the revision. Remaining arguments are ignored, preserving the source
behavior. A leading option chooses default revision resolution. Extra `--fix`
therefore does not modify changed files in this mode.

Revision defaults remain `upstream/main`, `origin/main`, then `main`. Comparison
uses their common ancestor with HEAD. The exact changed selection remains
`git diff --diff-filter=d --name-only <revision> -- '*.py'`. It includes Python
files outside the package and generated Python names, excludes removed and
untracked files, and retains renamed paths and spaces in filenames. Backend
ignore policies apply after that selection.

Zero selected files returns zero without a lint child. Invalid revisions return
one without a lint child. Unrelated histories produce the source's empty
merge-base comparison and invalid diff, followed by zero selected files and
status zero. This original defect is retained and tested separately from useful
comparison evidence. Existing direct comparison/default-resolution evidence is
reused from the accepted revision helper and pytest/coverage command batches.

Every lint invocation returns its child status exactly, including ordinary
failure bitmasks. The native module/interpreter replaces the source PATH command.
Native execution inherits the environment. The original hardcoded development
PYTHONPATH becomes a fully qualified native plugin path in the retained profile.
Lint children have a 1200-second bound and Git queries have a 15-second bound.

## Project policy and retained-source policy

Native package/changed modes invoke existing Ruff with `--force-exclude` and the
checkout's `pyproject.toml`. Ruff owns the existing project rule set and automatic
parallelism. Its arguments and diagnostics follow Ruff. The original Pylint
`--jobs=0` and configuration flags are represented by the native backend's
execution and configuration policy. Controlled shell comparisons verify forwarded
argument order, targets, status propagation and selection independently of the
backend diagnostic differences.

The complete frozen Pylint profile is separately executable for explicit retained
source audits. Its only parsed setting change is the custom checker module name,
from `pylint_copyright_checker` to `tools.development.copyright_checker`. A parsed
configuration equality check validates every section, key and value after that
single namespace mapping.

| Source setting group | Executable retained-source disposition |
| --- | --- |
| `load-plugins` | Original docstyle/docparams plugins plus the accepted native copyright checker. |
| `max-line-length`, `ignore-long-lines` | Original width 100 and exact URL/directive expression. |
| `ignore-patterns` | Original generated protobuf filename expression. |
| `output-format`, `score`, `reports` | Original colorized output with score and reports disabled. CLI overrides remain supported. |
| `py-version` | Original 3.9 target for retained-source audits. Native project target remains owned by the existing native configuration. |
| Entire `disable` list | Every source entry retained, including attrs/Cirq, documentation, import and protected-member dispositions. |
| `enable` | Original imported-source copyright diagnostic enabled. |
| `extension-pkg-whitelist` | Original NumPy entry retained. |

This opt-in profile validates files under its original notice and lint policy.
Each file's license and owner remain determined by its provenance. Its inclusion
establishes executable source-policy coverage while the normal project lint modes
continue using the existing native policy. Generic project lint diagnostic parity
between Ruff and Pylint is outside this policy mapping.

## Direct dependency reuse and correction

The accepted isolated lint environment supplies frozen Pylint 3.3.9, Astroid
3.3.11 and their recorded dependencies. Their existing provenance and license
review is in `COPYRIGHT_CHECKER_FILE_AUDIT.md`. The Pylint host supplies docstyle,
docparams, argument parsing, suppression, parallel execution and JSON diagnostics.
This batch adds no dependency manifest changes. Explicit retained-source mode
requires that optional development environment, and missing Pylint is a visible
child failure. Integration owns this subprocess/profile boundary. Its removal or
replacement requires the same complete configuration and diagnostic comparisons.

The native notice checker already corrects the source's truncated-header
acceptance defect. Complete notice diagnostics match the unchanged source. An
incomplete header is explicitly tested as source status zero and native status
eight, with the first missing-line diagnostic. The original source is unchanged.

## Verification

The complete shell wrappers run across success and five failure statuses.
Tests compare flags, target order, working directory, source environment,
changed-mode ignored arguments, Git selection and actual divergent/unrelated
histories. Actual native Ruff CLI executions verify passing and undefined-name
failures for both package and changed scopes, with unchanged file bytes.

Actual retained-source CLI execution compares full JSON diagnostics against the
original full profile for valid notices, wrong holders, undefined names and
missing parameter documentation. The known truncated-notice correction is
asserted independently. Frozen configuration equality covers the entire profile.

Existing quantum amplitude, measurement, workspace and separate measured
primitive/model evidence is reused for this classical tooling batch. Q-0049
mypy profile, Algorithms NumPy ownership, broad default-suite timeout and the
preceding-execution copyright interaction remain open. Pending Proposals
116, 117 and 118 remain outside the approved implementation scope.

Final concentrated regression passed 438 tests with 912 visible warnings and zero
skips. Strict mypy passed both changed Python files. Black, Ruff, Integration
scope, architecture, IR, API and diff checks passed. The actual current checkout's
`changed HEAD` invocation also returned zero with zero tracked Python changes.
