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

The frozen `dev_tools/conf/mypy.ini` was inspected in full and blob-verified.
Its migration parent Q-0049 remains open. The following dispositions explain the
current commands and identify the remaining dependency-policy work.

| Source setting | Native disposition |
| --- | --- |
| `show_error_codes=true` | Current mypy default, demonstrated by actual `[return-value]` diagnostics. |
| `duet.typing` plugin | Foreign async framework plugin. Native developer orchestration uses subprocess and existing helpers. No plugin added. Applicability across remaining migrated sources remains open. |
| `numpy.typing.mypy_plugin` | No plugin added. Algorithms NumPy dependency ownership remains an explicit unresolved cross-policy issue. |
| `allow_redefinition=true` | Native strict policy retains the existing default. Source-wide relaxation requires a concrete native requirement. |
| `check_untyped_defs=true` | Included by native `--strict`. |
| `disable_error_code=override` | Native override checks remain enabled under existing strict policy. Source subclass suppression is not copied. |
| `__main__` silent imports and missing stubs | Existing native missing-import policy retained. No per-module silent import suppression added. |
| Google and enumerated third-party silent imports and missing stubs | Existing global missing-import policy retained. Source per-module import traversal suppression is not copied. |
| SymPy `follow_untyped_imports=true` | Source-specific traversal policy remains unported pending native dependency scope review. |
| Protobuf skip imports including stubs | Source suppression remains unported. Native generated-schema typing scope must be assessed before changing approved policy. |

The source and native strict policy deliberately have different diagnostics.
Whole profile equivalence is not claimed. Closing the two wrapper sources does
not close the source configuration parent or the full-library strict typing gap.

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
