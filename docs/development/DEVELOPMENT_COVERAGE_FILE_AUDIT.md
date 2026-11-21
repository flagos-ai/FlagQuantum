# Development commit comparison and incremental coverage

The complete reference files are unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Tests verify each frozen Git blob
against the inventory.

| Ledger | Source file | Native owner |
| --- | --- | --- |
| Q-0083 | `dev_tools/qualtran_dev_tools/prepared_env.py` | `tools/development/environment.py` |
| Q-0061 | `dev_tools/qualtran_dev_tools/incremental_coverage.py` | `tools/development/incremental_coverage.py` |
| Q-0062 | `dev_tools/qualtran_dev_tools/incremental_coverage_test.py` | Unmodified original tests with native provider substitution |

## Ownership and interfaces

Integration owns these internal stdlib tools. `PreparedEnvironment` owns commit
comparison context, delegates subprocess execution to the accepted process owner,
and returns changed tracked paths. Equal revisions compare against the previous
commit. A missing actual revision compares local uncommitted tracked content.
A missing destination lets subprocess inherit its caller's directory.

Incremental coverage owns unified-diff ranges, deletion-neighbor marking,
annotation parsing, ignored line/block/file rules and colored reports. It uses
the same environment and process helpers. It introduces no CI gate, policy
contract, runtime dependency or public framework export.

## Complete file acceptance

Tests execute the complete original ignored-line suite against the native owner.
Native/source comparisons additionally exercise every function, all file filters,
manual and annotated line decisions, diff insertions/deletions/overlap, malformed
hunks, the source terminal TYPE_CHECKING failure, actual committed and uncommitted
Git changes, deletion and untracked behavior, annotated and absent annotation
reports, clean reports and invalid revisions. Output comparison is exact,
including terminal colors, line numbers and reasons.

Native Git diff calls have explicit 15-second timeouts. These timeouts are the
same deliberate execution-policy change as the accepted root-discovery helper.
All other comparison and reporting behavior follows the frozen implementation.

## Preserved boundaries

The original ignored file patterns are retained, including its source-layout
and historical generated-client exclusions. They are a reference diagnostic
profile and do not replace FlagQuantum's existing coverage policy. This optional
internal analyzer therefore includes native tools paths unless an original
pattern matches. No repository CI policy is changed.

The parser is intentionally source-equivalent and indentation-based. It is not
an AST parser. A terminal TYPE_CHECKING header without a following line raises
IndexError. Multiline strings, line continuations and hash characters inside
strings retain the source heuristic limitations. The line classifier still
regards `x = "#"` as executable. Git paths with embedded newlines retain the
source splitlines limitation. Git diff executes in the caller's directory, so
full reports must run inside the analyzed checkout. Report environments with
changed Python files require a concrete destination path, preserving the source
TypeError when it is absent.

Quantum amplitudes, measurement branches, clean work qubits, measured primitive
gates and resource estimates are inapplicable to these classical tools.

The final batch passed 111 concentrated cases, strict mypy on all three new
Python files, Black, Ruff, Integration scope, architecture, internal IR, public
API baseline and whitespace checks. The missing-destination path has an explicit
runtime type guard preserving the original TypeError, with native/source tests.
