# Public API Protection

## Purpose

This policy defines how FlagQuantum protects a frozen public API from accidental
changes by maintainers, automation, and coding agents such as Codex.

The objective is not to prevent implementation work. Runtime, compiler, planner,
backend, and performance code must remain free to evolve. The objective is to
ensure that a stable user contract cannot change without an explicit API change
decision, migration path, automated evidence, and human approval.

The governing principle is:

> An agent may propose a public API change, but it must not be able to merge one
> merely by changing the implementation and regenerating its snapshots.

This document complements:

- the user-facing compatibility promise in
  [`PUBLIC_API_POLICY.md`](../reference/PUBLIC_API_POLICY.md);
- the pre-open-source convergence work in
  [`OPEN_SOURCE_API_QUALITY_PLAN.md`](../roadmap/OPEN_SOURCE_API_QUALITY_PLAN.md);
- the version and deprecation rules in [`RELEASE_POLICY.md`](RELEASE_POLICY.md).

## Protection status

The controls in this document are the required target state. A control is not
considered active merely because it is documented. It becomes active only when
its contract, checker, CI job, repository rule, and responsible owner are all in
place and have been tested with an intentional breaking-change fixture.

Until the full target state is active, `docs/public_api_v1.json` is useful change
detection but must not be treated as a complete security or compatibility
boundary.

Phase 0 also records the pre-open-source migration baseline in
`contracts/public-api-v0.2-baseline.json`. The baseline checker is
`tools/public_api_snapshot.py`; it protects the starting point while the final
Stable Core is being selected. It is intentionally not the final frozen API
contract.

## Protected contract

For each Stable Core API, protection must cover more than its import name.

### Surface

- root and stable-namespace exports;
- canonical import path;
- class and function names;
- public methods and properties;
- supported constructor forms;
- experimental and internal namespace boundaries.

### Signature

- positional and keyword-only parameter order;
- parameter names and default values;
- accepted public types;
- return types;
- overloads;
- dataclass field names, order, defaults, and frozen status;
- Enum and Literal values.

### Behavior

- configuration precedence;
- planning and execution equivalence;
- measurement replacement or composition rules;
- capability failure stage;
- fallback visibility;
- exception class and actionable error category;
- result accessor behavior;
- PyTorch autograd and `Module.forward` behavior.

### Serialization

- IR schema and version;
- ExecutionPlan schema and identity inputs;
- ExecutionResult metadata schema;
- DeploymentPackage schema;
- checkpoint schema;
- backward-reading guarantees and migration rules.

Performance, internal data structures, private helpers, and backend algorithms
are not frozen unless a separate capability or serialization contract says so.

## Four enforcement layers

No single layer is sufficient. FlagQuantum must use all four.

### 1. Agent and maintainer instructions

The repository `AGENTS.md` must tell coding agents that the stable contract is
protected and that snapshots must not be updated merely to make a check pass.

These instructions prevent many accidental changes, but they are a soft
control. They do not replace CI or repository rules.

Required agent behavior:

- inspect the stable contract before editing a protected surface;
- preserve public signatures and semantics by default;
- stop and request explicit authorization for a breaking change;
- never regenerate a protected snapshot as an automatic repair;
- report any unavoidable API impact before implementing it;
- keep new backend-specific controls out of the stable root API;
- add behavior tests for any authorized additive API.

### 2. Machine-readable contract

The final API contract should live under `contracts/`, for example:

```text
contracts/public-api-v1.toml
```

It must describe the frozen surface, signatures, types, fields, enum values,
schema versions, and selected behavioral invariants. A checker such as:

```text
tools/check_public_api_contract.py
```

must compare the installed package against that contract using Python
introspection and explicit schema readers.

Illustrative contract entries:

```toml
[functions.run]
signature = "(program_or_plan, *, options=None, measurements=None, noise_model=None) -> ExecutionResult"

[functions.plan]
signature = "(program, *, options=None, measurements=None, noise_model=None) -> ExecutionPlan"

[classes.ExecutionOptions]
frozen = true
fields = ["mode", "backend", "device", "batch_size", "precision"]

[classes.ExecutionResult]
fields = [
  "value",
  "state",
  "samples",
  "measurements",
  "plan",
  "metrics",
  "provenance",
]
```

Contract checking must fail with a focused diagnostic, for example:

```text
public API contract changed:
  fq.run: parameter 'options' was removed
  fq.run: parameter 'config' was added
```

### 3. Executable semantic contracts

Signature checks cannot detect a behavioral change that keeps the same function
shape. Stable semantics therefore require tests under a dedicated location such
as:

```text
tests/api_contract/
```

At minimum, the suite must prove:

```python
plan = fq.plan(circuit, options=options)
result = fq.run(plan)
assert result.plan.identity == plan.identity
```

and:

```python
automatic = fq.run(circuit, options=options)
explicit = fq.run(fq.plan(circuit, options=options))
assert automatic.plan.identity == explicit.plan.identity
```

It must also prove that:

- a stale plan fails closed instead of silently replanning;
- configuration precedence is deterministic;
- measurements are never implicitly overwritten;
- unsupported capabilities fail during planning when knowable there;
- backend fallback is visible in ExecutionResult;
- stable result accessors either return the requested type or raise the
  documented FlagQuantum exception;
- `Module.forward()` returns an autograd-compatible Tensor;
- `Module.execute()` returns ExecutionResult;
- serialized stable objects pass backward-reading fixtures.

### 4. Server-side merge enforcement

Local hooks are useful feedback but are not protection: Git allows them to be
bypassed with `--no-verify`. The final guarantee must be enforced by the GitHub
repository.

The default branch ruleset must:

- prohibit direct pushes and force pushes;
- require pull requests;
- require the `public-api-contract` status check;
- require CODEOWNER approval for protected files;
- dismiss stale approvals after new commits;
- require approval of the latest revision;
- prevent ordinary bots, administrators, and maintainers from bypassing the
  rule during normal development;
- protect the checker and workflow that enforce the contract.

The practical guarantee is therefore not that Codex cannot edit a file. It is
that an unauthorized change cannot enter the protected default branch.

## Preventing snapshot-update bypass

The most important threat is a change that modifies both the implementation and
the expected snapshot. A PR-local comparison alone would then pass.

The API workflow must compare the pull request against the protected contract
from the target branch, not only against the contract contained in the pull
request.

Conceptually:

```text
PR implementation
      │
      ├── inspect actual API
      │
      └── compare with protected contract from target branch
```

If a PR changes a protected contract or its enforcement code, the default CI
result must be:

```text
protected public API material changed;
API change proposal and CODEOWNER approval required
```

Updating a snapshot must never be an automatic formatter or fixer action.

For stronger isolation, the workflow can fetch the target-branch contract by
commit SHA or store the release contract as a signed release artifact. The
target-branch comparison is the minimum requirement.

## Protected files and ownership

The exact list should be reviewed when the final Stable Core is selected. The
minimum CODEOWNERS coverage is expected to include:

```text
/contracts/public-api*                 @FlagQuantum/api-maintainers
/docs/public_api*                      @FlagQuantum/api-maintainers
/docs/reference/PUBLIC_API_POLICY.md   @FlagQuantum/api-maintainers
/flagquantum/__init__.py               @FlagQuantum/api-maintainers
/flagquantum/runtime/result.py         @FlagQuantum/api-maintainers
/flagquantum/runtime/policy.py         @FlagQuantum/api-maintainers
/flagquantum/runtime/execution.py      @FlagQuantum/api-maintainers
/flagquantum/runtime/module.py         @FlagQuantum/api-maintainers
/tools/check_public_api_contract.py    @FlagQuantum/api-maintainers
/.github/workflows/api-contract.yml    @FlagQuantum/api-maintainers
/.github/CODEOWNERS                    @FlagQuantum/api-maintainers
/AGENTS.md                             @FlagQuantum/api-maintainers
```

Replace the example team with an existing GitHub team before enabling the
ruleset. Do not add an unresolvable CODEOWNER merely to make the file appear
complete.

Implementation files may need API-owner review without being immutable. The
review protects contract impact; it must not prevent compatible internal
optimization.

## API change authorization

### Compatible implementation change

No API proposal is required when all protected surface, signature, semantic,
and serialization checks remain unchanged. Normal code review applies.

### Additive stable API

An additive API requires:

1. a concrete user journey;
2. a reason it belongs in Stable Core rather than a namespace or experimental;
3. typing and documentation;
4. executable behavior contracts;
5. API-owner approval;
6. release-note entry;
7. updated machine-readable contract after approval.

Adding an export is not automatically harmless: every stable addition creates
a long-term maintenance obligation.

### Breaking API change

A breaking change requires an API change proposal under a location such as:

```text
docs/api-changes/FQ-API-XXXX.md
```

The proposal must record:

- problem and affected user journey;
- alternatives considered;
- old and proposed signatures or schemas;
- source and behavioral compatibility impact;
- migration example;
- first deprecation version;
- planned removal version;
- documentation and tooling impact;
- owner and approvals.

After the first stable public release, a breaking change normally requires:

- a supported replacement API;
- `DeprecationWarning` containing the replacement and removal version;
- at least one published minor-version migration window unless the release
  policy requires a longer one;
- compatibility tests for old and new paths;
- migration documentation and release notes;
- the semantically appropriate major or pre-1.0 minor version.

### Emergency exception

A security, correctness, or data-loss issue may justify accelerated removal.
The exception must still be documented, approved by the API owner and release
owner, covered by a regression test, and announced in release notes. Convenience
or implementation difficulty is not an emergency.

## Required CI design

The required `public-api-contract` workflow should run at least:

1. target-branch contract comparison;
2. actual export and import-path inspection;
3. signature/default/type inspection;
4. dataclass, Enum, and Literal inspection;
5. serialization fixtures;
6. semantic API contract tests;
7. documentation source-of-truth validation;
8. detection of protected contract/checker/workflow changes;
9. an explicit approval check for authorized contract changes.

The workflow itself must be required and CODEOWNER-protected. A green general
unit-test job must not substitute for this focused check.

## Codex-specific operating rule

When Codex is asked to implement, refactor, optimize, clean up, or reorganize the
repository, it must assume the Stable Core contract is unchanged unless the user
explicitly authorizes an API change.

Codex must not treat any of the following as implicit authorization:

- “clean up the API”;
- “make the repository professional”;
- “refactor the runtime”;
- “improve typing”;
- “remove legacy code”;
- “fix all tests”;
- “update snapshots”.

If an authorized implementation exposes a previously unknown contract conflict,
Codex must stop that part of the work and report:

- the exact stable API affected;
- the current and proposed behavior;
- why a compatible implementation is not practical;
- migration options;
- required contract and release changes.

It must not silently choose the breaking option.

## Rollout checklist

The protection mechanism is complete only when all items below are checked.

- [ ] Stable Core has been selected and reviewed.
- [ ] Public contract records exports, signatures, defaults, types, fields, and
      schema versions.
- [ ] Public semantic contracts exist under a dedicated test suite.
- [ ] `AGENTS.md` contains the stable API protection rule.
- [ ] API contract regeneration is never an automatic fix.
- [ ] CI compares against the protected target-branch contract.
- [ ] CI fails when implementation and snapshot are changed together.
- [ ] API checker, contract, workflow, CODEOWNERS, and agent instructions are
      owned by the API maintainer team.
- [ ] Default-branch rules require the API check and CODEOWNER approval.
- [ ] Direct pushes and force pushes to the default branch are disabled.
- [ ] Stale approvals are dismissed after new commits.
- [ ] Additive and breaking API proposal templates exist.
- [ ] Deprecation and emergency procedures are documented and tested.
- [ ] An intentional breaking-change fixture has demonstrated that the server
      blocks merge.
- [ ] An intentional snapshot-update bypass attempt has demonstrated that the
      server blocks merge.

## Definition of protection

FlagQuantum may claim that its stable API is protected only when:

1. accidental changes fail locally or in CI with a focused diagnostic;
2. behavior-preserving internal refactors continue to pass;
3. changing both code and snapshot still fails by default;
4. changing the enforcement workflow requires API-owner review;
5. no actor covered by normal repository rules can merge a breaking change
   without explicit authorization;
6. every approved breaking change carries a migration and release record.

No repository mechanism can prevent an organization owner from deliberately
disabling all controls. Within normal development authority, the combination of
target-branch contracts, semantic tests, required checks, CODEOWNERS, and branch
rules provides the enforceable boundary.
