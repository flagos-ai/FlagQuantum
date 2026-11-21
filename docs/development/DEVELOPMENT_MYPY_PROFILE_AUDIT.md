# Complete imported-source typing profile audit

## Source closure and ownership

Q-0049 is the entire `dev_tools/conf/mypy.ini` from Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The source and its direct requirement
and environment files Q-0097/Q-0106 are restored unchanged and Git-blob verified.
Every parsed section, key and value equals the native profile. Provenance comments
use the native imported-source terminology.

`tools/development/retained_source_mypy.ini` owns this explicit source audit
policy. The existing `typecheck_command.py` owns execution through accepted
process/root helpers. It forwards explicit targets and mypy arguments, exact
status and the red/reset terminal wrapper. Child calls are bounded at 1200
seconds. There is no implicit target. Optional flag overrides follow the
configuration argument, preserving normal mypy CLI precedence.

```sh
python -m tools.development.typecheck_command retained-source path/to/source.py
```

Run this with the isolated audit dependencies described below on `PYTHONPATH`.
A valid annotated file exits zero. Invalid types return the original diagnostic
and status. A missing plugin exits two and reports its import failure. Package
and developer modes continue to use the existing root configuration, Python 3.12
strict flags and native namespace policy. Framework production code never imports
this helper or profile. This developer workflow supports the existing
`circuit.append(op)` execution through source auditing.

## Complete profile comparison

| Setting | Executed witness |
| --- | --- |
| `show_error_codes=true` | Exact source/native argument, assignment and import error codes. |
| `duet.typing` | `sync` retains an async function's integer argument and awaited string result. Removing the plugin accepts an invalid argument and changes the signature to arbitrary arguments. |
| `numpy.typing.mypy_plugin` | `intp` resolves to the host pointer width. Removing the plugin leaves the 32/64-bit union. |
| `allow_redefinition=true` | A list variable can become a tuple. Disabling this produces the original assignment diagnostic. |
| `check_untyped_defs=true` | A bad annotated assignment in an unannotated function is diagnosed. Disabling this accepts it. |
| `disable_error_code=override` | An incompatible subclass parameter override is accepted. Removing suppression produces the original override diagnostic. |
| `mypy-__main__` | Both silent traversal and missing-import suppression match the complete parsed source configuration. |
| Google/third-party module lists | Every original module pattern is retained. Missing Google and Matplotlib submodules are accepted, while an unlisted missing module is diagnosed. |
| SymPy `follow_untyped_imports=true` | `Symbol` resolves to the real symbolic class. Disabling traversal produces an untyped-import diagnostic and `Any`. |
| Protobuf skip including stubs | `Message` resolves to `Any` and permits an unknown member. Normal stub traversal exposes the typed class and missing-member diagnostic. |

Seven actual mypy CLI scenarios compare complete diagnostics, stderr and status
between the unchanged original config and native audit mode. Controlled single
setting/plugin mutations demonstrate the settings' observable effect. Config-only
`__main__` verification is explicit. Exhaustive module-pattern execution and
whole migrated-library typing are outside this acceptance scope.

## Isolated dependency boundary

The original direct file declares `mypy~=1.14`, `mypy-protobuf` and `sympy`. The
frozen environment pins mypy 1.19.1. The isolated target installs the required
plugin and representative import closure using these exact frozen versions.

```sh
uv pip install --python /path/to/python3.12 --target /path/to/mypy-profile-deps --no-deps \
  mypy==1.19.1 librt==0.7.7 mypy-extensions==1.1.0 numpy==2.2.6 \
  duet==0.2.9 sympy==1.14.0 mpmath==1.3.0 protobuf==6.33.2 \
  types-protobuf==6.32.1.20251210 mypy-protobuf==4.0.0 pathspec==1.0.1 \
  typing-extensions==4.15.0
PYTHONPATH=/path/to/mypy-profile-deps:$PWD python -m tools.development.typecheck_command retained-source path/to/source.py
```

For this batch the target is `ftqc-reference-env/mypy-profile-deps` beside the
checkout. Tests require the exact twelve-distribution set without duplicate
metadata and match every version to the unchanged lock. This is the source-audit
subset, not acceptance of the complete large environment parent Q-0106 or direct
requirements parent Q-0097. No production dependency manifest changes are needed.

Installed metadata and license-file hashes are recorded in
`g2-development-mypy-profile-dependencies-20261121.json` in the reference workspace.
Mypy, librt and mypy-extensions carry MIT notices. Duet, mypy-protobuf and
Protobuf stubs carry Apache-2.0 notices. SymPy, mpmath, Protobuf and NumPy carry BSD
notices. NumPy's binary notice also lists bundled components and the compiler
runtime exception and LGPL notices. Pathspec carries MPL-2.0 and typing-extensions
PSF-2.0. Dependencies remain installed external tools with original notices.
No third-party plugin implementation is copied into production. Their replacement
boundary is the optional mypy subprocess/profile, and removal affects only source
auditing. Native strict checks continue with their existing environment.

An initial targeted dependency reinstall left duplicate metadata. That task-owned
target was deleted and rebuilt in a single exact `--no-deps` installation before
acceptance. Exploratory fixtures were renamed to avoid shadowing the `duet` and
`numpy` modules. Final diagnostics use the rebuilt environment and corrected
filenames. Frozen sources and plugin code remain unchanged.

## Verification and remaining boundaries

The concentrated regression includes the complete check composer, lint, format,
typing, coverage, notebook and documentation tool groups. Strict typing,
formatting, Integration scope, architecture, IR and API checks run separately.
This classical configuration has no primitive gates, resource model, measurements
or workspace. Existing quantum validation remains with its accepted modules.

The broad default suite timeout, Algorithms NumPy ownership and preceding-execution
copyright interaction remain unresolved in the main ledger. Optional source
NumPy typing evidence does not resolve production numerical ownership. Pending
Proposals 116/117/118 remain outside this batch.

The final concentrated run passed 528 cases with 915 visible warnings and zero
skips in 219.93 seconds. Strict mypy passed all three changed Python files. Black,
Ruff, Integration scope, architecture, IR, API and diff checks passed. Source
profile/dependency receipts and full logs are retained beside the frozen inputs.
