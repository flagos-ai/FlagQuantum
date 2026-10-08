# Governance

How decisions are made in this repository, who makes them, and what is deliberately not
defined. Every statement about the current state below is measured; the measurements are
given so they can be re-run.

For the workflow a change follows to reach `main`, see
[Integration workflow](docs/development/INTEGRATION_WORKFLOW.md). This document covers who
decides, not how a change travels.

## Roles

| Role | What it is | Who is in it |
| --- | --- | --- |
| Contributor | Anyone who opens an issue or a pull request. No membership step and no form to fill in. | Anyone |
| Team | One of the 12 entries in `team-ownership.toml`. **A path-ownership classification, not a people roster.** | Unstaffed; see below |
| Integration | The role that resolves protected paths and cross-team changes. `--team integration` is the supported bypass in `tools/check_team_scope.py`. | The maintainer |
| Maintainer | Merge rights on `main`; decides releases, API authorization, and this document. | One account today |

### The teams are not staffed

`team-ownership.toml` defines 11 teams, and each owns a set of paths. They are a
classification mechanism: `tools/check_team_scope.py` resolves *which domain owns this
path*, which is what makes a boundary checkable. They are **not** eleven working groups.

Read the roster honestly: there is one maintainer. No branch named in that file has ever
existed, the work of the 561 merged pull requests was done by one person plus two one-off
contributions, and a team whose `owns` list matched nothing was removed rather than kept
as a claim. `docs/development/INTEGRATION_WORKFLOW.md` records the measurements.

## Decision classes

| Class | Examples | Who decides |
| --- | --- | --- |
| Ordinary change | A domain's internal implementation, its tests, and its documentation. | The required checks on the pull request, merged by the maintainer |
| Protected surface | `AGENTS.md`, `architecture.toml`, `team-ownership.toml`, `contracts/**`, `.github/**`, `tools/**`, `pyproject.toml`, `flagquantum/core/ir.py` | Integration owner. A separate integration change, before the team implementation proceeds |
| Stable Core API | The 34 exports in `docs/public_api_v1.json` | An API change proposal plus recorded authorization, per `AGENTS.md` |
| Capability level | A level in `capability-maturity.toml` | Evidence that satisfies `tools/check_capability_maturity.py`, approved by Integration |
| Release | `flagquantum/version.py`, tags, release notes | Maintainer |

The distinction that matters is the second row. `tools/check_team_scope.py` refuses a
team-authored change to a protected path, which is why a change that *starts* as an
implementation can become an integration change. The rule is mechanical rather than a
matter of judgment.

## The current maintainer

Re-measured 2026-10-08 on a full clone (`git fetch --unshallow`); the 2026-10-01 figures
are kept beside them because the earlier numbers are the baseline this document is
compared against.

| Measurement | 2026-10-01 | 2026-10-08 |
| --- | --- | --- |
| First-parent commits on `main` | 214 | 540 |
| Authored by `Wei LIU <liuwei.chem.phys@gmail.com>` | 213 | 537 |
| Authored by anyone else | 1 (`Qiming Teng <tengqm@outlook.com>`, 2026-09-18) | 3 (`Qiming Teng <tengqm@outlook.com>` 2, `Yulong Ao <aoyulong@outlook.com>` 1) |
| Repository collaborators | 1 (`FlagQuantum`, `admin`) | 1 (`FlagQuantum`, `admin`) |
| GitHub API contributors | 3 (`FlagQuantum` 1543 commits, `tengqm` 2, `aoyulong` 1) | 3 (`FlagQuantum` 2069 commits, `tengqm` 2, `aoyulong` 1) |
| Merged pull requests | 271 | 561 |
| Closed without merging | 2 | 22 |
| Open pull requests | 7 | 1 |

The API's contributor counts and the commit authors on `main` do not agree, because a
squash merge attributes a whole pull request to its merger and history has been rewritten
at least once. The git-side figures are the ones that describe `main`, and since October
2026 they include the merge commits themselves, which are attributed to the account that
pressed merge: 73 of the 275 first-parent commits since 2026-10-01 are merges, so the
maintainer's own share now counts merges as well as authored work.

The open-pull-request count is quoted with the moment it was read, and this row was
recorded wrong once. It was first written as **0**, and zero never held on 2026-10-08:
[#598](https://github.com/flagos-ai/FlagQuantum/pull/598) was opened at 02:40Z and
[#597](https://github.com/flagos-ai/FlagQuantum/pull/597), the pull request that closes
[issue #579](https://github.com/flagos-ai/FlagQuantum/issues/579), was open from 01:46Z
until it merged at 05:47Z, so the queue held two at the moment of that measurement and
holds one now. The paragraph this replaces said the queue had been empty and that #597 was
the first entry it had held since, which was a recalled state rather than a read one. That
is the failure this document is written against, one row up: an unmeasured number in a
table whose subject is measured ones.

## What protects `main`

Measured with `gh api repos/flagos-ai/FlagQuantum/branches/main/protection` on
2026-10-08, after the change recorded in [issue #579](https://github.com/flagos-ai/FlagQuantum/issues/579):

| Setting | Value |
| --- | --- |
| `required_status_checks` | the eight names in `.github/required-checks.json` |
| `required_status_checks.strict` | `true`: the head branch must be up to date with `main` |
| `enforce_admins.enabled` | **`true`** |
| `required_approving_review_count` | **`0`** |
| `required_linear_history.enabled` | `false`, because `main` carries merge commits |
| `allow_force_pushes.enabled` / `allow_deletions.enabled` | `false` |
| `required_conversation_resolution.enabled` | `false` |

Until that date the same endpoint returned `required_approving_review_count: 2` with
`enforce_admins` `false` and **no required status checks at all**. Two required approvals
were nominal in two ways at once: `enforce_admins` was off while the authoring account
holds `admin: true`, so the requirement could be bypassed, and there was no second
reviewer to meet it. The measured consequence was that the fourteen pull requests merged
immediately before the 2026-10-01 measurement, #271 through #284, were each authored and
merged by `FlagQuantum` with zero reviews, and that reading a merged tree was optional:
two merges to `main` 17 seconds apart on 2026-10-08 cancelled all 23 check runs attached to
the first of them.

Both halves are now stated rather than left nominal:

- **Zero required approvals.** A requirement nobody can meet is not a rule. The count now
  says what this repository does, and the control that replaces it is the next bullet.
- **The required checks are the review of record for `main`.** Each of the eight must
  complete successfully, on a head branch up to date with `main`, before a merge is
  accepted — including a merge by the maintainer, because `enforce_admins` is on.
  `.github/required-checks.json` records the names and `tools/validate_required_checks.py`
  proves each name is one a workflow job produces, so a required check cannot be a name
  that reports nothing. The name is the check run's own name — the job key, or the job's
  `name:` where it declares one, with GitHub's matrix suffix — and not a
  `"<workflow> / <job>"` path to it. That distinction is not a preference: branch
  protection was first given the `"<workflow> / <job>"` spelling of these eight names and
  pull request #597 stayed `blocked` with all nineteen of its check runs `success`,
  because no check run is reported under that spelling. Written as the job names alone,
  the same contexts reported as required and the pull request went `clean`.
- **A red `main` blocks release.** The answer to the third question in issue #579 is that
  a revision of `main` whose required checks did not all complete successfully is not a
  release candidate; [Version and release policy](docs/development/RELEASE_POLICY.md)
  carries the operative sentence.

What this does not do: it does not make a red `main` impossible, and it does not require a
human to read a pull request. Both are stated consequences rather than unreported ones.

## Becoming a maintainer

There is no committee and no vote, because there is no second maintainer to hold one. The
path is:

1. Land changes that stay within one domain and survive review without follow-up fixes.
2. Review other changes in that domain, and be right about them.
3. Be recorded here as the domain's integration approver, which is a decision the
   maintainer records in this file.

Step 3 is intentionally a recording step rather than a ceremony: the useful artifact is a
name against a domain that other contributors can check, and that belongs in this
document rather than in a parallel one.

## Files deliberately not added

Two files that a governance checklist expects are absent on purpose. Both would duplicate
an authority that already exists.

**`MAINTAINERS.md` — not added.** `docs/development/MULTI_TEAM_DEVELOPMENT.md` states of
the machine-readable authority that "This document does not maintain a second roster." A
`MAINTAINERS.md` beside a `team-ownership.toml` is exactly such a second roster, and the
repository already has a machine-checked one. The maintainer and the domain approvers are
recorded above, in the document that explains how the decision is made.

**`.github/CODEOWNERS` — not added.** `team-ownership.toml` plus
`tools/check_team_scope.py` already are the path-ownership authority, and the classification
is enforced on every pull request by `ci.yml`. A `CODEOWNERS` file would be a second
statement of the same fact, which `AGENTS.md` prohibits, and it would have no effect on
merging today because `require_code_owner_reviews` is `false`. Generating it from
`team-ownership.toml` would be a defensible alternative and is an open question below;
hand-maintaining it would not be.

## Security

Vulnerability reporting, supported versions, and scope are in [SECURITY.md](SECURITY.md).
Security fixes are merged on the expedited schedule recorded in `dependency-policy.toml`
rather than the monthly dependency cadence.

## Amending this document

This file is a protected integration path in `team-ownership.toml`, so amending it is an
integration change. A change that names a new maintainer, a new domain approver, or a new
decision class should state the measurement that supports it, in the same way the tables
above do.

## Open questions

1. **Adopt the Contributor Covenant, and at which version?** 2.1 and 3.0 both exist, and
   the text requires naming an enforcement contact and carries its own license, which
   would place a differently licensed document in an Apache-2.0 repository. The current
   [code of conduct](CODE_OF_CONDUCT.md) states the standard this project already had
   and says plainly that it is not the Covenant.
2. **Generate `.github/CODEOWNERS` from `team-ownership.toml`?** There is one obvious
   benefit — GitHub would surface the owning domain in the review UI without any rule
   change — and one obvious cost: a generated file needs a check that it is current, which
   is a new protected tool.
3. **Should the 11 teams be staffed, or renamed?** A team name that maps to no person is
   a truthful description of a path-ownership domain. It is a misleading description of an
   organization, and today the file can be read either way.
4. **What replaces the "one maintainer" state?** The honest answer is more contributors.
   The measurement above is the baseline to improve on, and it is recorded so progress is
   visible rather than asserted.

## Resolved questions

- **Should the two-approval requirement be enforced or removed?** *(closed 2026-10-08)*
  Removed. The eight required checks, enforced against the maintainer as well, are the
  control that replaces it; see [What protects `main`](#what-protects-main).
