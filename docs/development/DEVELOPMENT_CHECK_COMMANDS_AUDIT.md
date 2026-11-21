# Complete development check composer audit

## Source closure and ownership

This batch covers the complete frozen `check/all` Q-0023 at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its original Git blob is verified and
the entire unchanged shell composer executes in temporary Git checkouts.
`tools/development/check_command.py` owns native orchestration.

The composer delegates directly to the accepted lint, strict typing, incremental
formatting and coverage command owners. Each owner retains its own configuration,
process bounds, path selection, generated-file cleanup and optional dependencies.
The repository helper owns explicit commit validation and checkout discovery.
Each accepted owner supplies its existing phase implementation.

```sh
python -m tools.development.check_command HEAD --only-changed-files
python -m tools.development.check_command HEAD
python -m tools.development.check_command --apply-format-changes
```

Every invocation runs lint, package typing, formatting and coverage in that order.
The changed flag chooses changed-file lint and changed-file coverage. Typing still
checks the package, as in the original. Apply mode passes `--apply` to formatting.
Arguments retain optional revision, repeated flags and invalid extra-argument
handling. Invalid arguments fail before any stage starts.

These development commands support the existing `circuit.append(op)` source
workflow. Integration owns this developer orchestration.
A result of zero means every composed stage returned zero for the actual invoked
scope.

## Original defects and native corrections

| Original behavior | Native behavior and evidence |
| --- | --- |
| Shell exit status comes only from the last coverage command. Earlier lint, typing and formatting failures can be hidden. | Every phase runs after ordinary nonzero statuses and the final result is one if any phase failed. All sixteen binary status combinations are compared in both scope/apply modes. |
| Incremental lint receives no arguments, despite the usage comment promising the chosen base revision. | Explicit revision reaches changed lint, formatting and coverage. Recorded calls compare the exact original omission and native forwarding. |
| Formatter receives two positional strings, including an empty apply argument. With an explicit revision and no apply flag, the real source formatter rejects the second empty argument as too many arguments. | Empty optionals are omitted. The unchanged original formatter reproduces the hidden failure while the native formatter accepts the same requested comparison. |
| Optional empty revision is passed to the coverage wrapper as a literal empty string. | Absent revision is omitted and each existing command resolves its own accepted defaults. |

These corrections belong to native orchestration. Comparisons execute the
original frozen shell bytes verified against the inventory. Comparison tests
retain the original erroneous success and parameter forwarding as explicit
negative evidence.

Native subprocess exceptions propagate from their owning phase. Existing
per-phase timeouts remain active.

## Dependency and policy reuse

Package and changed lint use the existing native Ruff policy. The optional
retained-source Pylint profile remains in its explicit audit command and is
independently accepted. Typing uses the accepted native strict package policy.
Formatting uses retained Flynt conversion with existing Ruff/Black formatting.
Coverage uses accepted pytest selection, annotation conversion, touched-line
analysis and exact generated annotation cleanup.

The composer uses existing development dependencies, per-phase configurations
and Python call boundaries. Optional Flynt and developer
pytest/mypy dependencies follow their existing owner audits. The accepted
copyright checker correction, formatting policy differences, formatter
fail-on-change status, unrelated-history selection defect and coverage conversion
status behavior remain documented at those owners.

## Verification

Tests execute the unchanged composer across all failure combinations for full and
changed scope, with and without applying format changes. They verify all four
child calls occur, exact original argument arrays, explicit native corrections,
defaults, repeated flags and rejected inputs.

A dedicated real source-formatter case proves the empty-argument failure is
masked by the original composer. Actual native CLI runs all four accepted owners
in a small Git checkout for default comparison, explicit comparison and changed
scope. Real pytest coverage succeeds and generated annotations are removed.
A subsequent actual Ruff failure still runs the later passing coverage phase and
returns one. File bytes remain unchanged in check mode. The test environment
bounds xdist auto concurrency at one worker while retaining the normal command.

Existing quantum amplitude, measurement, workspace and separate measured
primitive/model evidence is reused for this classical orchestration batch.
Q-0049 mypy profile, Algorithms NumPy ownership, broad default-suite timeout and
preceding-execution copyright interaction remain open. Pending Proposals
116, 117 and 118 remain outside the approved implementation scope.

Final concentrated regression passed 511 tests with 912 visible warnings and zero
skips. Strict mypy passed both changed Python files. Black, Ruff, Integration
scope, architecture, IR, API and diff checks passed. This final run includes the
actual default-comparison pipeline in addition to explicit and changed scopes.
