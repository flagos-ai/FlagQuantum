# Complete pytest developer command source group

## Source parents and native ownership

This batch closes complete Qualtran v0.7.0 source parents Q-0030 check/pytest,
Q-0032 check/pytest-changed-files and Q-0034 check/pytest-dev-tools at immutable
bcb17bb25360ed68ce3bc084645db3e929799389. Every original Git blob is verified.
The source wrappers execute unchanged in temporary Git checkouts with a boundary
pytest executable that records the complete arguments and returns controlled
exit statuses. Actual native command execution separately runs real pytest.

tools/development/pytest_command.py owns command scope and dispatch. Existing
process.run owns child invocation, repository.get_git_root owns root discovery,
and PreparedEnvironment owns the changed-file Git query. The focused
changed_tests.py module owns revision selection and filename-associated test
selection. It is reused directly by the command; there is no new environment
record, process abstraction or test dependency graph.

```bash
python -m tools.development.pytest_command package -n0 -q
python -m tools.development.pytest_command development -n0 -q
python -m tools.development.pytest_command changed HEAD -q
```

Package and developer commands retain the source's initial -n=auto, all caller
arguments in order and exact pytest exit status. Callers can supply -n0 as in
the source. The source package test root maps to native tests/. The source
developer-tool root maps to tools/development/ and tests/unit/, since native
tests live separately from implementation. The developer command deliberately
includes that complete unit directory, rather than guessing which tests belong
to a tooling source. Commands run from the checkout root using the current
Python interpreter. Child commands have an explicit 1200-second bound; process
launch and timeout failures propagate visibly. Existing repository CI and
coverage policy remain their authoritative owners.

## Revision and selection fidelity

Changed-file mode consumes an explicit first non-option argument as a revision
and forwards remaining arguments to pytest in order. Otherwise it tries
upstream/main, origin/main and main in that exact order. Git cat-file must report
a commit. A missing default, invalid revision or blob reference returns one
without invoking pytest. Original color escapes are omitted in native error
messages. Git queries retain the accepted 15-second bound.

The changed mode intentionally compares the supplied revision directly against
the working tree. The original coverage wrappers have a different merge-base
policy and remain separate open source parents. A real diverged-branch test
proves that source and native changed selection include the comparison branch's
different implementation and do not substitute the merge base.

For every changed .py path, the original same-directory name_test.py rule is
retained, including deduplication of already suffixed tests. Native test_name.py
paths are also recognized. For flagquantum, tools and examples sources, an
existing tests/unit/test_name.py counterpart is included. Native selection is
still filename-based and does not claim dependency completeness or infer
renamed test implementations. Only existing files are selected. Results are
sorted and deduplicated. Untracked source paths retain the original Git diff
exclusion; staged files and rename targets are checked, and deleted tests are
excluded. No selected tests succeeds without launching pytest, matching source.

## Verification scope and remaining work

The original package/developer wrappers compare argument order and exit statuses
0, 1, 2 and 5. Changed mode compares original actual shell selection on real Git
changes, direct divergent revisions, forwarded flags, no matching tests, missing
defaults and invalid revisions. Native tests additionally cover centralized
layout, staged files, renames, deleted tests, actual module command invocation
and process failures. A real temporary checkout runs passing package/developer
scopes and a failing changed-file test through the normal module entry point.

The full source library's tests are not re-executed by these wrapper tests. This
is classical test orchestration. Existing accepted quantum amplitude,
measurement, workspace and measured-primitive/model evidence remains with its
owners. No new gate-count or full-library green claim is made. Full default-suite
NumPy dependency and prior-execution copyright debt remains open. Q-0023,
Q-0025 through Q-0029, Q-0031 and Q-0033 remain independent source parents. The
coverage-wrapper behaviors cannot be inferred from these three command closures.

The concentrated regression passed 340 tests with 914 visible warnings and zero
skips. Strict mypy passed all three changed Python files. Black, Ruff,
Integration scope, architecture, IR, API and whitespace checks passed. The final
sixteen command tests include actual original shell execution and actual native
pytest subprocesses. The developer wrappers are exercised on bounded temporary
trees; this is not a claim that a fresh full-repository pytest run is green.
