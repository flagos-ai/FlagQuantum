# Complete typing command source audit

## Ownership and invocation

`tools/development/typecheck_command.py` owns the complete `check/mypy` and
`check/mypy-dev-tools` command orchestration from frozen Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Existing process and repository helpers
own execution and checkout discovery. The existing `pyproject.toml` and
`.pre-commit-config.yaml` remain the typing policy authorities.

```sh
python -m tools.development.typecheck_command package
python -m tools.development.typecheck_command development
```

Both commands forward additional mypy arguments and exact child status, resolve
the checkout root, and retain the original red terminal wrapper. Calls are bounded
at 1200 seconds. Terminal reset also occurs on native subprocess exceptions.
The package target maps `qualtran/` to `flagquantum/`. The developer target maps
`dev_tools/` to `tools/development/`. The native interpreter invokes mypy as a
module. These are development commands supporting existing `circuit.append(op)`
workflows through static checking, with no quantum execution changes.

## Native typing policy mapping

The original config path maps to the existing `pyproject.toml`. The package
command uses the pre-commit package gate's `--strict`, Python 3.12 and missing
import policy. Strictness also applies to the complete developer helper target.
The developer command uses `--explicit-package-bases` to support the native
`tools` namespace. The source `--no-namespace-packages` assumes an ordinary
`dev_tools` package and would assign different module identities here.
User arguments follow policy defaults, retaining source argument override order.
The wrappers do not certify that the entire target currently passes strict mypy.

## Complete source configuration disposition

The frozen `dev_tools/conf/mypy.ini` is now delivered in full as
`tools/development/retained_source_mypy.ini`, through explicit `retained-source`
audit mode with explicit targets and optional locked plugins. Q-0049 is closed
by `DEVELOPMENT_MYPY_PROFILE_AUDIT.md`, which records each setting, complete
parsed equality, actual source/native diagnostics and policy-control witnesses.

Package and development modes keep the existing strict project policy. Their
diagnostics differ from the imported-source profile. The optional audit profile
owns source reproduction, including disabled override checks, silent imports,
SymPy traversal and Protobuf stub suppression. The audit does not certify
full-library native strict typing or resolve production NumPy ownership.

## Verification

Tests execute the complete unchanged source shell wrappers against a recorded
mypy child, compare argument order and terminal bytes across success and three
failure codes, and verify frozen blobs for both wrappers and the profile.
Actual native CLI runs validate package and namespace developer targets, imported
sibling module identity, passing code, strict return-value errors and exit codes.
Timeout cleanup and invalid command inputs are exercised separately.

This classical tooling batch reuses previously accepted quantum amplitude,
measurement, workspace and resource evidence. It adds no primitive gate counts or
model estimates. Full default regression timeout, Algorithms NumPy ownership and
preceding-execution copyright interaction remain open in the main ledger.

The concentrated regression passed 382 tests with 914 visible warnings and zero
skips. Strict mypy passed both changed Python files. Black, Ruff, Integration
scope, architecture, IR contract, API baseline and diff checks passed. The first
native developer CLI fixture accidentally shadowed the repository tool package.
The fixture now includes the actual command and its process/repository owners,
matching an executable checkout. Production and frozen reference were unchanged
by that harness correction.
