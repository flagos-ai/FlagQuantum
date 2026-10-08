# Integration workflow

This document declares how a change actually reaches `main` in this repository. It
exists because three binding documents describe a different workflow, and a
contributor or agent who follows them will produce work this repository cannot merge.

Status: **declared from measured evidence, with an open decision.** The facts below are
measurements against this checkout and the GitHub API on 2026-10-01. The decision —
which of the three documents to correct — is not taken here.

## The workflow in use

1. A change is developed on a short-lived branch cut from `main`.
2. The branch is pushed and a pull request is opened **with `main` as the base**.
3. CI runs on the pull request. The required checks are listed in
   `.github/required-checks.json`, and since 2026-10-08 all eight are enforced on `main`:
   each must complete successfully, on a head branch up to date with `main`, before the
   merge is accepted — for the maintainer as well, because `enforce_admins` is on. The
   merge push itself is read by the same eight checks, which is what makes arriving at
   `main` a reading rather than an event nobody looks at.
4. The pull request is merged. A squash merge produces exactly one commit whose subject
   ends in `(#NNN)`; at the 2026-10-08 measurement 73 of the 274 commits on `main`'s
   first-parent line since 2026-10-01 are merge commits instead, which the addendum below
   the measurements records.
5. The head branch is abandoned. It is not deleted automatically and it is not
   reintegrated.

History was linear by this convention rather than by rule: `required_linear_history` is
disabled on `main`, and at the 2026-10-01 measurement `main` carried 214 commits, 213 of
them ending in a pull-request number, and **zero merge commits**. The 2026-10-08
measurement below no longer finds that shape.

## Evidence

### The observed flow

| Measurement | Value |
| --- | --- |
| Commits on `main` | 214 |
| Commits whose subject ends in `(#NNN)` | 213 |
| Merge commits on `main` | **0** |
| Merged pull requests | 271 |
| Closed without merging | 2 |
| Open at measurement | 7 |
| Merged pull requests based on `main` | **265 of 271** |

Head-branch prefixes across the 271 merged pull requests:

| Prefix | Count |
| --- | --- |
| `feat/` | 165 |
| `fix/` | 39 |
| `test/` | 20 |
| `refactor/` | 18 |
| `docs/` | 9 |
| `perf/` | 6 |
| `ci/` | 5 |
| `dev/` | 4 |
| five one-off prefixes (`chore/`, `fq-010`, and others) | 5 |
| **Total** | **271** |

The prefixes are per-change descriptions, not per-team branches. `refactor/` appears 18
times and `dev/` 4 times; neither is a team identity, and no prefix reproduces the 11
team names in `team-ownership.toml`.

### Addendum, 2026-10-08: `main` now carries merge commits

The table above is the 2026-10-01 measurement, and the shape it describes did not hold.
`git log --first-parent`, on a full clone (`git fetch --unshallow`), reports the line of
commits `main` actually advanced through — the commits a merge landed, not the branch
history it pulled in. Boundaries are ISO commit dates, because `git log --since` reads its
boundary in the committer's timezone, which here is `+08:00`: the first commit counted as
October is `396ddeb8` at `2026-10-01T00:29:36+08:00`.

| Measurement | Value |
| --- | --- |
| First-parent commits, all history | 539 |
| Of those, merge commits, all history | 74 |
| Merge commits before 2026-10-01 | **1** — `3c932585`, 2026-09-11, `Merge pull request #15` |
| First-parent commits before the first merge, inclusive | 265 |
| First-parent commits since 2026-10-01 | 274 |
| Of those, merge commits | **73** |
| Of those, commits whose subject ends in `(#NNN)` | 201 |

Both conventions are in use, and the second is new. Before the single September merge,
`main`'s first-parent line held 264 non-merge commits, 98.5% of the 268 at that point: a
linear history with one exception. From 2026-10-01 the same line holds 73 merges in 274
commits, 26.6%, and 201 commits that still end in a pull-request number. The sample matters
here too — 700 commits are reachable from `main` in that window once the merged branch
histories are counted, so measuring "the newest 300 commits" overstates the merge share,
because most of those commits live on the branches rather than on `main`'s own line.

Two things follow, and both belong to the same finding as
[issue #579](https://github.com/flagos-ai/FlagQuantum/issues/579):

- **`main` advances by a push per merge, and a second push used to delete the first
  reading.** A pull request is read on its own merge ref while it is open, and that reading
  is of a tree that is not yet a revision of `main`: it is not a reading of `main` with
  every other change that landed in between. The merge push is therefore the only reading
  of the revision `main` actually holds, and with the push trigger's concurrency group
  keyed to the branch, the next push cancelled it: the merges of #590 and #591 landed 17
  seconds apart on 2026-10-08 and all 23 check runs attached to the first of them were
  `cancelled`, `quality` among them. The workflows no longer cancel a superseded run of
  `main`, and the required checks on the pull request are what must pass before the merge
  is accepted.
- **The `(#NNN)` convention no longer describes `main`.** 73 of the 274 first-parent
  commits since 2026-10-01 are merge commits and `required_linear_history` is disabled, so
  nothing rejects the shape. A reader who takes the five-step flow above as "one pull
  request, one commit" will misread the history this finding is about.

### The declared flow

| Document | What it declares | Line |
| --- | --- | --- |
| `AGENTS.md` | Sessions "must use the branch and linked worktree assigned in `team-ownership.toml`"; `refactor/flagquantum-vnext-architecture` is "the reserved name of the only authoritative integration branch", and the migration layout is optional | `244-287` |
| `docs/development/MULTI_TEAM_DEVELOPMENT.md` | 304 lines: linked worktrees, cross-team merges only through Integration, `git merge --no-ff` one team at a time, handoff records | full document |
| `team-ownership.toml` | 11 teams, each with a `branch` and a `worktree`, all marked reserved | `77-190` |
| `.github/workflows/ci.yml` | `push` triggered on `main`, `develop`, and `refactor/flagquantum-vnext-architecture` (corrected to `main` only by the change that edited this line) | `5` |
| `.github/workflows/ci.yml` | `pull_request` triggers on `main` only | `7` |

### The two sets do not intersect

| Declared | Actual | Evidence |
| --- | --- | --- |
| 11 team branches | remote has **4** branches total: `main` plus 3 short-lived ones | `git ls-remote --heads origin` |
| `refactor/flagquantum-vnext-architecture` is the integration branch | the branch **does not exist** on the remote | `git ls-remote --exit-code --heads origin refactor/flagquantum-vnext-architecture` fails |
| `refactor/vnext-team-core`, `-compiler`, `-runtime`, and the other 8 | none exist | same, per branch |
| `develop` is a CI push branch | `develop` **does not exist** | `git ls-remote --exit-code --heads origin develop` fails |
| 11 `worktree` names | no directory matching `FlagQuantum-vNext*` exists beside the checkout | `ls -d ../FlagQuantum-vNext*` |
| team branches deliver merged work | **exactly one** merged pull request ever came from a `vnext-team-*` branch: **#15**, `refactor/flagquantum-vnext-architecture` into `main`, 2026-09-11, "refactor: establish FlagQuantum v0.2.0 architecture" | GitHub API over all 271 merged pull requests |
| team branches are long-lived integration lines | the head branches of squash merges are abandoned; `feat/team-scope-changed-path-gate`, for example, still exists on the remote although its work landed as `4f4adbd` (#282) | `git ls-remote --heads origin` |

The single exception is decisive rather than incidental. PR #15 was the v0.2.0
architecture migration, and it is the only pull request in 271 that used the
integration branch. The 270 that followed used per-change branches off `main`. The
declared workflow is the **migration device** that produced the current layout; it is
not the steady state that maintains it.

The branch roster above is the 2026-10-01 measurement and it has since moved, which
strengthens the row rather than weakening it. On 2026-10-08 `git ls-remote --heads origin`
returns 9 branches — `main` and 8 others — and not one of the eight is a `refactor/vnext-team-*`
team branch:

| Branch | Ahead of `main` | Behind `main` |
| --- | --- | --- |
| `parity/w6-recovery` | 96 | 554 |
| `feat/cpu-inplace-capacity` | 3 | 230 |
| `feat/cpu-dense-native-one-qubit` | 2 | 230 |
| `feat/qboson-kaiwu-provider-core` | 2 | 0 |
| `bench/compiler-optimization-levels-vs-qiskit` | 1 | 0 |
| `feat/cpu-qft-diagonal-memory` | 1 | 230 |
| `fix/construction-acceptance-root-export-count` | 1 | 122 |

`fix/main-push-reading-is-not-cancelled` is omitted because it is the branch that produced
this revision and its work is this change. The roster grew from 4 branches to 9 while the
number of team branches stayed at zero, and six of the eight are hundreds of commits behind
`main` — the abandoned-head-branch pattern the open questions below describe, at a larger
scale than the 2026-10-01 table shows.

### Why the stale declaration is not merely untidy

`ci.yml:7` sets `pull_request: branches: [main]`. A pull request whose base is any other
branch therefore matches no workflow and reports **no checks at all**. This has already
happened, six times:

| Pull request | Base | Head |
| --- | --- | --- |
| #129 | `perf/1-cpu-bench-lane` | `perf/2-statevector-kernels` |
| #130 | `perf/2-statevector-kernels` | `perf/3-z-expectation-memory` |
| #131 | `perf/1-cpu-bench-lane` | `perf/4-joint-marginal` |
| #132 | `perf/1-cpu-bench-lane` | `perf/5-row-sampling` |
| #133 | `perf/1-cpu-bench-lane` | `perf/6-characteristics-docs` |
| #284 | `fix/team-ownership-protect-ir-and-assign-domains` | `feat/team-scope-changed-path-gate` |

Measured directly: the commit of #284 has **1 check run**; the merge commit of #281,
based on `main`, has **23**. A stacked pull request is not merely slower to verify — it
is unverified. Since 2026-10-08 `main` requires eight status checks, so the shape of this
failure changed rather than disappeared: a pull request based on another branch reports
none of them, GitHub leaves each one `Expected — Waiting for status to be reported`, and
the merge is **blocked** rather than accepted. The gate notices now; what it still cannot
do is run, because no workflow matches a base branch other than `main`. The remedy for a
stacked change remains to retarget it at `main` (or to land its base first), not to merge
it and hope.

This is exactly the failure mode the declared workflow would produce for *every* pull
request, since it routes work through an integration branch that CI does not watch.

### What `--validate` currently enforces

`tools/check_team_scope.py:64-76` requires every team to declare a unique `branch` and a
unique `worktree`, and `ci.yml:76` runs that validation on every pull request. So the
repository has a machine-checked invariant over 22 values that no process consumes. The
only reader of either field is the uniqueness check itself. The two fields are kept as a
reserved record of the migration layout, now labelled as reserved in both
`team-ownership.toml` and `AGENTS.md`, rather than removed.

### Branch protection now supplies the control, and did not before

`gh api repos/flagos-ai/FlagQuantum/branches/main/protection`, measured on 2026-10-01 and
again on 2026-10-08:

| Setting | 2026-10-01 | 2026-10-08 |
| --- | --- | --- |
| `required_approving_review_count` | 2 | **0** |
| `dismiss_stale_reviews` | false | false |
| `require_code_owner_reviews` | false | false |
| **`enforce_admins.enabled`** | **false** | **true** |
| `required_status_checks` | **absent** | the eight names in `.github/required-checks.json` |
| `required_status_checks.strict` | — | **true** |
| `required_linear_history.enabled` | false | false |

At the 2026-10-01 measurement, two required approvals were configured, the account that
authors and merges these pull requests holds `admin: true`, and the requirement was
bypassable: the last 14 merges were authored and merged by the same account with zero
reviews. The count of two could not be the mechanism that distinguishes the declared
workflow from the observed one, and it was recorded as nominal rather than corrected.

Two decisions were taken on 2026-10-08, in the change that closes
[issue #579](https://github.com/flagos-ai/FlagQuantum/issues/579):

1. **The two-approval requirement was removed rather than enforced.** A requirement that
   no second reviewer exists to meet is not a rule; the number now says what this
   repository does. The control that replaces it is the required checks.
2. **Eight required checks, strict, enforced against admins.** `quality`,
   `cpu-core (3.10)`, `cpu-core (3.11)`, `cpu-core (3.12)`, `package`, `distributed-cpu`,
   `coverage`, and `pre-commit` must each complete successfully on a head branch up to date
   with `main` before a merge is accepted — including a merge performed by the maintainer.
   `.github/required-checks.json` records the names, `tools/validate_required_checks.py`
   proves each name is one a workflow job produces, and `ci.yml`/`pre-commit.yaml` no
   longer cancel a superseded run of `main`, so the merge push of a revision is read
   instead of discarded.

The eight names are the names of the **check runs**, not a path to them. A check run is
reported under its job key — or the job's `name:`, where a job declares one — with the
matrix suffix GitHub appends, so `cpu-core (3.10)` and not `CI / cpu-core (3.10)`. This
was measured rather than assumed, because assuming it was wrong once here: branch
protection was given the `"<workflow> / <job>"` spelling of all eight names, and pull
request #597 stayed `blocked` with all nineteen of its check runs `success`, an empty
review decision, and nothing else outstanding. No check run is ever reported under that
spelling, so each of the eight sat `Expected` forever. Written as the job names alone, the
same eight contexts were recognised and the pull request went `clean`.
`tests/unit/test_gpu_workflow_watchdog_policy.py` pins the contract against the nineteen
names GitHub reported on that pull request's head commit, which is what makes the mistake
reproducible as a test failure rather than as a second silent outage.

Nothing in this repository compares the two, and that is the remaining hole in the gate.
`.github/required-checks.json` is a *record* of what branch protection holds —
`externally_configured: true` says so — and `tools/validate_required_checks.py` reads it
against the workflow documents, not against GitHub. The settings are changed through the
API, so a later edit to branch protection can leave the file describing a rule that is no
longer in force, and no check fails. Re-reading the two together is a manual step, and it
takes two commands:

```bash
diff <(gh api repos/flagos-ai/FlagQuantum/branches/main/protection \
         --jq '.required_status_checks.contexts[]') \
     <(python -c "import json; print(*json.load(open('.github/required-checks.json'))['checks'], sep='\n')")
gh api repos/flagos-ai/FlagQuantum/branches/main/protection --jq '.required_status_checks.strict'
```

An empty diff and `true` are the agreeing state. A non-empty diff means the record and the
enforcement have parted, and the record is the half that is wrong: branch protection is the
rule, and the file is what the repository believes about it.

The approved API change proposals written before this date still describe
`tools/validate_required_checks.py` as pinning "six externally configured required
checks". That was true of the contract at the time each was approved, and those proposals
are records of an approval rather than a description of the current roster, so they are left
as written. The roster is eight here and in `.github/required-checks.json`.

What this does not do: it does not make a red `main` impossible, and it does not require a
human to read a pull request. A red `main` blocks a release
([release policy](RELEASE_POLICY.md#a-red-main-blocks-the-release)); a green pull request
is what the eight checks make it mean.

## Decision Candidates

**Candidate A — Declare the observed pull-request flow authoritative.** Rewrite
`AGENTS.md:247-274` and `docs/development/MULTI_TEAM_DEVELOPMENT.md` to describe branch
off `main`, pull request, squash merge; remove the `branch` and `worktree` fields from
`team-ownership.toml` and the uniqueness check with them; trim `ci.yml:5` to `main`.
`team-ownership.toml` keeps its actual job — path ownership — and loses the two fields
that describe a process that is not running. Consequence: the machine-checked invariant
matches the enforced reality, and the removal of the uniqueness check is a subtraction
rather than a new rule.

**Candidate B — Adopt the declared workflow.** Create the integration branch and the 11
team branches, change `ci.yml` to watch the integration branch, and route all pull
requests at it. Consequence: it must be staffed as a real process — one Integration
owner merging one team at a time with `--no-ff` and per-team handoff records — and the
265-of-271 measurement says that is not what has been happening.

**Candidate C — Declare the pull-request flow authoritative and keep the worktree
workflow as an explicitly optional mode for a coordinated multi-team round.** This is
Candidate A with `team-ownership.toml` retaining the `branch`/`worktree` fields marked
as reserved, no longer required to be live. Consequence: the documents stop presenting
an optional mode as mandatory, at the cost of keeping 22 values nothing reads.

**Candidate D — Keep all four declarations and change nothing.** Consequence: an agent
following `AGENTS.md` creates a team branch and worktree and delivers a handoff record
that cannot merge, and the cost is rediscovered each time.

**Candidate E — Make the flow machine-checked before rewriting the prose.** Add a check
that a pull request's base is `main` and that its head branch follows the observed
prefix convention, then correct the prose once the check exists. Consequence: the
defect that produced six unverified pull requests is closed by construction, but it adds
a new rule before the subtraction review that Candidate A performs.

### Decision recorded

Candidate C was chosen, with the `ci.yml:5` trim from Candidate A applied at the same
time:

- `team-ownership.toml` keeps `branch` and `worktree`, and the file now marks them as
  reserved names rather than current instructions. `AGENTS.md` and
  `docs/development/MULTI_TEAM_DEVELOPMENT.md` say the same.
- `ci.yml:5` no longer lists `develop` or `refactor/flagquantum-vnext-architecture`.
  Neither has ever existed, so both entries described events that can never fire.
- Candidate A's remaining subtraction — deleting the two fields and the uniqueness check
  — is **not** applied. Candidate B stays open. The two unstaffed branches above remain
  the record of why.

Candidate E was not adopted, and on 2026-10-08 the case for it narrowed. Its stated
motivation was that a pull request whose base is not `main` reports no checks at all and is
therefore *silently* unverified. Since the eight required checks are enforced on `main`,
such a pull request reports none of them, GitHub leaves each `Expected — Waiting for status
to be reported`, and the merge is blocked rather than accepted. The failure is loud now, so
what remains of Candidate E is a base-branch convention check rather than a repair of a
silent hole. It is not added here: no evidence yet shows a maintainer attempting a stacked
merge since the checks were enforced.

One part of the stale declaration was corrected on its own terms. Of the eleven `owns`
lists, five named directories that do not exist in this repository
(`flagquantum/ops`, `flagquantum/_compiler`, `flagquantum/compilation`,
`flagquantum/devices`, `flagquantum/encoding`) and the `docs` team named three more
(`docs/site`, `docs/roadmap/VNEXT_EXPERIENCE.md`,
`docs/development/DOCUMENTATION_STRATEGY.md`). A rule that matches no tracked path
assigns nothing, and the `docs` team, holding only those three, held nothing at all, so
it was removed. `tests/unit/test_team_scope_policy.py` now asserts the property for
`owns` as it already did for `protected_paths`.

## Prohibited Practices

The following are recorded so that the declaration cannot decay while it is decided:

1. **Opening a pull request against a base other than `main`.** CI does not watch any
   other base, so such a pull request is unverified. If stacking is genuinely needed,
   the base-branch filter must be fixed first, as a separate change.
2. **Treating a `branch` or `worktree` value in `team-ownership.toml` as a current
   instruction.** Those fields describe the v0.2.0 migration. A new value added today
   would be a new unread value.
3. **Adding a third description of the integration flow.** Four sources already exist
   and three of them disagree with the repository; one declaration is the point.
4. **Recording a workflow in `AGENTS.md` that CI cannot run.** The operating manual is
   the highest-authority instruction an agent receives; an instruction there that
   produces unmergeable work is worse than silence.
5. **Using the `(#NNN)` squash convention as evidence that a pull request was
   verified.** #284 followed the convention and had one check run.

## Compatibility

- **`AGENTS.md` and `docs/development/MULTI_TEAM_DEVELOPMENT.md` are prose.** Correcting
  them changes no executable behavior. Both are protected/shared paths with different
  owners: `AGENTS.md` is a protected integration path (`team-ownership.toml`), while
  `docs/development/**` is shared, so any team may edit the second and only Integration
  may edit the first.
- **Removing `branch` and `worktree` from `team-ownership.toml` is a policy change.**
  `tools/check_team_scope.py:64-76` reads them and `ci.yml:76` runs the validation on
  every pull request, so the removal must land with the checker and be reflected in
  `tests/unit/test_team_scope_policy.py`. `team-ownership.toml` and `tools/**` are both
  protected integration paths.
- **`ci.yml:5` triggers.** Removing `develop` and `refactor/flagquantum-vnext-architecture`
  from the `push` trigger removes two events that can never fire. If Candidate B is
  chosen instead, that line must keep the integration branch and line 7 must be changed
  to watch it, which is a CI behavior change rather than a cleanup.
- **`docs/development/MULTI_TEAM_DEVELOPMENT.md` has no machine reader.** Nothing
  imports it and nothing asserts its content, so it can be rewritten or reduced without
  a compatibility path. Its 304 lines are the largest single statement in the repository
  about a process that is not running.
- **`ci.yml:7`'s base-branch filter is load-bearing for the finding above.** It is the
  reason a stacked pull request reports nothing, and it is the one part of the stale
  declaration that has already caused repeated, measurable harm.

## Acceptance Tests

A declaration of the workflow is complete when:

1. **One authority.** `AGENTS.md`, `docs/development/MULTI_TEAM_DEVELOPMENT.md`, and
   this document state the same steps, and a reader can find no conflicting instruction.
2. **The branch fields have a reader or are gone.** Every field in `team-ownership.toml`
   is either read by a check or has been removed; there is no machine-checked invariant
   over values no process consumes.
3. **The CI triggers match the flow.** Every branch named in `ci.yml`'s `push` and
   `pull_request` triggers either exists or is documented as intentionally reserved.
4. **A stacked pull request cannot be silently unverified.** Either the base-branch
   filter admits the documented alternative bases, or the practice is prohibited in a
   document a contributor reads before opening one.
5. **The measured flow is stated as measured.** This document's claims carry the
   commands and dates that produced them, so the declaration can be re-verified rather
   than trusted.

## Open Questions

1. **Is the worktree workflow retired, or dormant?** Candidate A retires it; Candidate C
   keeps it as an option. The answer determines whether the 24 `branch`/`worktree`
   values are deleted or marked reserved, and that is a subtraction decision.
2. **Who is the Integration owner in the observed flow?** Path ownership still names an
   integration team for protected paths, and `--team integration` is the supported bypass
   for a cross-team change. Is that role a reviewer, a merger, or only a path
   classification?
3. **Should squash merges be required by rule rather than convention?** At the 2026-10-01
   measurement `main` was linear because every merge had been a squash, with
   `required_linear_history` disabled. The convention changed in October 2026 — 73 of the
   274 first-parent commits since 2026-10-01 are merge commits — so the question is now
   whether the current shape is the intended one, rather than whether an accident would be
   rejected.
4. **What happens to abandoned remote branches?** Squash merging leaves the head branch
   behind, so the remote accumulates branches whose work is already in `main`. As of
   2026-10-08 none of the eight non-main branches is in that state:
   `feat/team-scope-changed-path-gate`, whose work landed as `4f4adbd`, has since been
   deleted, and all eight that remain are ahead of `main` by at least one commit. The
   question narrows from "what do we do with branches whose work has landed" to what the
   roster addendum above measures instead: four branches holding one to three unmerged
   commits while 122 to 230 behind, two holding one or two commits and not behind at all,
   and one carrying 96 commits from 554 behind.
5. **Does the existing `refactor/` and `dev/` prefix usage mean a team-branch naming
   scheme is wanted after all?** The observed prefixes are per-change, not per-team, but
   18 `refactor/` and 4 `dev/` branches suggest a looser convention that has never been
   written down.

Closed on 2026-10-08: **should the two-approval requirement be enforced or removed?** It
was removed, and the eight required checks — strict, and enforced against admins — are the
control that replaces it. See [Branch protection now supplies the
control](#branch-protection-now-supplies-the-control-and-did-not-before).

## Owner and approvals

- Owning domain: `integration`. This document, `AGENTS.md`, `team-ownership.toml`,
  `tools/check_team_scope.py`, and `.github/workflows/ci.yml` are all integration-owned
  protected paths, and Candidate A changes nothing outside them.
- `docs/development/INTEGRATION_WORKFLOW.md` is a shared documentation path, so this
  file may be added by any team; the corrections it proposes to `AGENTS.md` and
  `team-ownership.toml` may not.
- Required approvals before implementing a candidate: **integration owner** for all of
  them, plus **the maintainer who owns CI** if `ci.yml` triggers change.
- Sequencing: this declaration, then the candidate choice, then the prose corrections,
  then — only if Candidate A is chosen — the removal of the `branch`/`worktree` fields
  with the check that validates them, in one change so the checker and the policy never
  disagree.
- Smallest useful step: **Candidate A's prose corrections alone**. They change no
  executable behavior, they remove the instruction that produces unmergeable work, and
  they leave the `team-ownership.toml` field removal to a separate review.
