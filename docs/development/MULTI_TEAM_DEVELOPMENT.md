# FlagQuantum Development Across Multiple Sessions

## Goal

Multiple development sessions may work from one architecture baseline, but must
not share Git working directories, staging areas, or uncommitted state.
`FlagQuantum-vNext` is the integration workspace. Each team develops only in its
own linked worktree and branch.

The root `team-ownership.toml` is the machine-readable authority for teams,
branches, workspaces, and path ownership. This document does not maintain a
second roster.

## Workspace Model

```text
codex/flagquantum-vnext-architecture   FlagQuantum-vNext             Integration
codex/vnext-team-core                 FlagQuantum-vNext-core        Core
codex/vnext-team-compiler             FlagQuantum-vNext-compiler    Compiler
codex/vnext-team-runtime              FlagQuantum-vNext-runtime     Runtime
codex/vnext-team-simulation           FlagQuantum-vNext-simulation  Simulation
codex/vnext-team-platform-providers   FlagQuantum-vNext-platform    Compute
codex/vnext-team-execution-providers  FlagQuantum-vNext-execution   Remote
codex/vnext-team-ecosystem            FlagQuantum-vNext-ecosystem   Ecosystem
codex/vnext-team-agent-services       FlagQuantum-vNext-agent       Application Services
codex/vnext-team-docs                 FlagQuantum-vNext-docs        Docs / User Experience
```

Each Codex session opens only one directory. Do not switch a team workspace to
another team's branch or copy uncommitted files from another workspace.

Docs / User Experience may define target user journeys alongside implementation,
but unimplemented behavior must be marked `Target Experience`. This team cannot
independently change public APIs, capability maturity, dependencies, CI, or release
claims. Integration separately approves those changes.

## Session Startup Checks

Before work, every session must:

1. Read root `AGENTS.md` and the nearest `AGENTS.md` for edited directories.
2. Verify that the current branch matches the team's entry in `team-ownership.toml`.
3. Verify that the worktree is clean.
4. Read the overall architecture, relevant domain designs, capability maturity,
   and Stable Core protection policy.
5. Identify input/output contracts, failure semantics, and acceptance tests.
6. Run team-scope preflight before editing.

Explicit-file preflight:

```bash
python tools/check_team_scope.py \
  --team compiler \
  --files flagquantum/compiler/pipeline.py tests/team/compiler/test_compiler_pipeline_replacement.py
```

Completed-branch check:

```bash
python tools/check_team_scope.py \
  --team compiler \
  --base codex/flagquantum-vnext-architecture
```

Use `FLAGQUANTUM_GIT` to specify a Git executable when system Git is unavailable.

## Modification Authority

### Teams May Edit Directly

- Their most-specific `owns` paths in `team-ownership.toml`.
- Related tests, examples, benchmarks, and unprotected documentation.
- Internal module implementations that do not change cross-domain contracts.

### Integration Must Coordinate

- Stable Core, public APIs, and serialized schemas.
- `contracts/`, `architecture.toml`, `team-ownership.toml`, and root `AGENTS.md`.
- Overall architecture and ADRs.
- Dependency manifests, CI, and release configuration.
- New cross-domain contracts, Compute/Remote types, or long-term compatibility layers.

Submit a contract/ADR proposal when a protected area is involved. Do not create a
private substitute type on a team branch.

## Path Ownership Resolution

The most-specific rule wins:

- `runtime/**` defaults to Runtime.
- `runtime/executors/**` belongs to Runtime and owns only plan-aware execution
  and lifecycle orchestration.
- `simulation/**` belongs to Simulation and owns numerical algorithms/kernels.
- `compute/**` belongs to Compute.
- `remote/**` and transitional `runtime/target_execution.py` belong to Remote.
- `ecosystem/**`, including extension protocols, belongs to Ecosystem.

Current code therefore has one responsible team even before directory migration.
Remove obsolete rules when code reaches its target directory.

## Contract-First Integration Order

Cross-team features integrate in two stages:

```text
Contract/ADR proposal
 -> Integration approves and merges contracts, fakes, and contract tests
 -> Team branches synchronize the baseline
 -> Teams implement independently
 -> Team-scope checks and module tests
 -> Integration merges
 -> Cross-implementation conformance and replacement tests
```

Do not change Core contracts, Compiler, Runtime, Simulation, and Provider together
in one large change.

## Synchronization and Merging

### Sole Authoritative Version

`codex/flagquantum-vnext-architecture` is the only integration branch. Team
branches are temporary development lines, not product versions for release,
acceptance, or performance claims. A unified version must record:

- Integration commit SHA.
- Core contract and serialization versions.
- Test and capability evidence.
- The corresponding internal milestone or tag.

Linked worktrees share a Git object database. Once a team commits, Integration can
merge by branch name immediately; no file copying or prior remote push is required.

### Synchronize Before Delivery

Teams must commit their changes and leave a clean worktree. If Integration gained
commits during the round, merge it into the team branch from that team's worktree:

```bash
git merge codex/flagquantum-vnext-architecture
```

Resolve synchronization conflicts only in the team's own worktree, then rerun
scope, architecture, and module tests. Do not rebase commits already delivered or
referenced by other sessions.

Before delivery:

```bash
python tools/check_team_scope.py \
  --team TEAM \
  --base codex/flagquantum-vnext-architecture

python tools/check_architecture.py
```

Use `TEAM_HANDOFF_TEMPLATE.md`. Include the shared baseline, team branch, final
commit SHA, changed files, test evidence, capability limitations, and issues for
Integration. Uncommitted files are not deliverables.

### Direct Local Merging in a Private Repository

The integration owner merges only in `FlagQuantum-vNext`. Check first:

```bash
git branch --show-current
git status --short
```

The branch must be `codex/flagquantum-vnext-architecture` and the worktree clean.
Recheck team scope in the corresponding team worktree, then merge one team at a time:

```bash
git merge --no-ff codex/vnext-team-compiler
```

Follow dependency order:

```text
Core contracts
 -> Compiler and Runtime
 -> Simulation and Platform
 -> Remote
 -> Ecosystem
 -> Application Services
```

Independent or inventory/test-only commits may use a different order, but each
merge must be verified individually. Do not stack another team before the current
merge passes.

### Gates After Every Merge

Immediately run:

1. `python tools/check_team_scope.py --validate`.
2. `python tools/check_architecture.py`.
3. Public API and serialization contract checks.
4. The merged team's module tests.
5. Tests for affected consumers.
6. Established replacement, conformance, precision, and fallback-evidence tests.

The integration owner must review capability claims, actual execution paths, and
the migration ledger. Do not pass a merge by updating snapshots, loosening
thresholds, or adding silent fallback.

### Conflicts and Failures

If ownership is unclear, two teams need the same implementation, or a protected
contract must change, stop expanding the change. Integration must first decide
the authoritative location and merge order. Do not bypass conflicts through
copied files, temporary re-exports, free-form dictionaries, or cross-layer calls.

For merge conflicts, run in the integration worktree:

```bash
git merge --abort
```

The owner of the conflicting files then synchronizes the latest integration
branch, resolves conflicts, tests, and commits. Integration must not rewrite a
team's implementation without understanding its algorithm and evidence.

If a completed merge fails verification, prefer a fix from the originating team.
If Integration must be restored immediately, use an auditable revert:

```bash
git revert -m 1 MERGE_COMMIT_SHA
```

Do not conceal failure in shared integration history through force pushes, hard
resets, or rewritten commits.

### Synchronization Between Teams

- Team branches obtain other teams' changes only through Integration.
- Direct team-to-team merges, such as Compiler merging Runtime or Runtime merging
  Simulation, are prohibited.
- Keep each commit within one team's scope where possible. Split cross-team
  features into contract and implementation commits.
- Produce a new shared baseline after each integration round, then notify teams
  to synchronize.
- Do not start the next implementation round that depends on new contracts
  before teams finish synchronization.

### Create a New Shared Baseline

After all round tests pass, record the integration SHA and create an annotated
internal tag, for example:

```bash
git tag -a vnext-phase0-integrated -m "FlagQuantum vNext phase 0 integrated baseline"
```

The tag must point to an integration commit that passed that round's required
gates. Subsequent tasks, performance results, and team synchronization reference
that commit/tag, not irreproducible labels such as "latest code."

### Move from Local Private Development to Remote Private Collaboration

Early workflow:

```text
Team local commit -> direct integration-worktree merge -> full verification -> private remote push
```

As contributors and partner organizations grow:

```text
Team branch pushed to private remote
 -> Internal PR
 -> Path-owner approval
 -> Required CI
 -> Merge queue retests against latest integration
 -> Integration branch
```

Protect the remote integration branch against direct and force pushes. Configure
`CODEOWNERS` once actual GitHub/GitLab accounts or team identifiers are available.
Private repositories restrict access; they do not change these versioning and
merge rules.

## Definition of Done

A team task is ready for integration only when all conditions hold:

1. Changed paths belong to that team.
2. Input, output, failure, and capability boundaries are explicit.
3. No new cross-layer imports, vendor-object leakage, or duplicate authoritative types.
4. Implementation and contract fake pass the same tests, or missing replacement
   verification is explained.
5. Precision, fallback, and performance claims have corresponding evidence.
6. Relevant tests, architecture, and team-scope checks pass.
7. The handoff names one primary domain. Ordinary implementation changes across
   multiple domain internals explain why they cannot be split and identify the
   approved contract boundary. Repeated changes across four or more domains are
   architecture defects and cannot be delivered directly.
8. New contract types demonstrate why existing authoritative types cannot express
   the verified requirement.
9. At least one readable user or Provider scenario test accompanies contract and
   safety tests, or the change explains why no visible scenario changes.
10. A migrated domain has a README describing ownership, prohibitions, allowed
    dependencies, public entry points, and a golden path understandable, runnable,
    editable, and testable in approximately ten minutes.
11. Public APIs do not expose internal fingerprints, provenance records, legality
    attestations, capability snapshot IDs, scheduler objects, or evidence internals.
12. A subtraction review removes scaffolding without current uses, pass-through
    wrappers, duplicate validation/representations, comments restating code, and
    tests enumerating implementation details. New managers, registries, factories,
    protocols, helper layers, or intermediate objects justify a distinct current
    responsibility or second concrete use.
13. The worktree contains only this task's changes.
