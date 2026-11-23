# Complete CI diagnostic matcher ownership

The four complete files Q-0004 through Q-0007 come from Qualtran v0.7.0 at
bcb17bb25360ed68ce3bc084645db3e929799389. All original Git blobs are verified.

| Source file | Git blob | Native owner |
| --- | --- | --- |
| .github/problem-matchers/README.md | ac82dcb67ab3a747bd023f012d074aec62dae253 | Native matcher README, provenance and quality-job registration |
| .github/problem-matchers/black.json | 605f10c2706c5ae64fb19142c88f9a91eb9892e6 | Existing Black quality check diagnostic annotations |
| .github/problem-matchers/mypy.json | f048fce528941f8ad5a5f9d453f1d690c7204115 | Existing strict mypy quality check diagnostic annotations |
| .github/problem-matchers/pylint.json | 5624ca695c4e0a117f65054d653d822c17897b25 | Existing optional retained-source lint diagnostic annotations |

## Interface and integration

Each JSON file consumes diagnostic text and produces a path, message, severity
and any declared line, column or diagnostic code. The declarations are copied
byte-identically. Integration owns workflow configuration and registration.
No external matcher service or numerical runtime dependency is introduced.
Existing CI commands keep their exit-status responsibility. Native ordinary
lint uses Ruff's GitHub output format, while optional retained-source Pylint
keeps its existing separate owner and installation scope.

The quality job registers all three files after checkout and environment setup,
before its existing checks. The normal lint command adds output-format github.
Black and mypy commands retain their existing options. The source JSON patterns
are single-line declarations. Registration in other workflows remains explicit.
Hosted runner annotation rendering is verified by the workflow, independently
of the local declaration and diagnostic-capture acceptance.

## Complete behavior and provenance

The Black pattern captures the complete would-reformat path as group 2 and the
message as group 1, with declared error severity. The mypy pattern captures path,
decimal line number, error/warning severity and complete message. The two Pylint
patterns classify E/F messages as errors and C/R/W messages as warnings, retaining
path, line, column, message and diagnostic code. Their original matching behavior,
including greedy path capture and the lack of mypy note matching, is retained.

The original README attributes mypy and Pylint to Home Assistant 2025.1.2 under
Apache-2.0, and Black to Sumanth077's MLflow fork dated 2022-05-29 under Apache-2.0.
Native documentation preserves these attributions. The original relative
workflows link points beneath the matcher directory. Its mypy attribution URL
spells pypy.json. Those documentation mistakes are recorded without changing the
frozen source. Native documentation identifies the actual local workflow owner.

## Verification and resources

Ten focused checks execute actual Black, mypy and Pylint failure diagnostics and
Ruff GitHub output. JavaScript RegExp performs source/native capture comparison
through the available Node runtime. Actual inputs include spaces in a path,
line/column fields, error/warning groups and source message/code preservation.
Synthetic vectors separately cover all five Pylint categories and summary/note
nonmatches. Workflow declaration checks verify matcher paths, registration order
and the existing consumers. Tool failure exit statuses remain nonzero. Native
Ruff annotations retain status 1. No dependency is silently skipped.

These configuration files add zero quantum gates, measurements or workspace and
have no quantum resource model. Existing circuit.append(op) and algorithm
amplitude, clean-work and resource receipts retain their own acceptance.
Original hosted PR/nightly/scorecard workflows have separate inventory parents.
Receipts are stored beside the worktree at
ftqc-reference-env/g2-problem-matcher-*20261123.

## Batch acceptance

All four complete parents are accepted for diagnostic declaration, native
registration and provenance ownership. Ten focused checks and 153 concentrated
CI/tooling/policy/storage checks pass. Strict typing passes the touched Python
file. Black, Ruff, Integration scope, language, architecture, multilevel IR and
protected API checks pass. The dependency finding set remains the identical
720-entry baseline. The earlier scalar-storage and fenced-link fixes remain
covered in this batch. Hosted annotation display retains its separate workflow
execution scope.
