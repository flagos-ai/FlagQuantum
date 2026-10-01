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

`team-ownership.toml` defines 12 teams, and each owns a set of paths. They are a
classification mechanism: `tools/check_team_scope.py` resolves *which domain owns this
path*, which is what makes a boundary checkable. They are **not** twelve working groups.

Read the roster honestly: there is one maintainer. Five of the twelve team branches in
that file have never existed, and the work of the last 271 pull requests was done by one
person plus two one-off contributions.

## Decision classes

| Class | Examples | Who decides |
| --- | --- | --- |
| Ordinary change | A domain's internal implementation, its tests, and its documentation. | One reviewer; currently the maintainer |
| Protected surface | `AGENTS.md`, `architecture.toml`, `team-ownership.toml`, `contracts/**`, `.github/**`, `tools/**`, `pyproject.toml`, `flagquantum/core/ir.py` | Integration owner. A separate integration change, before the team implementation proceeds |
| Stable Core API | The 34 exports in `docs/public_api_v1.json` | An API change proposal plus recorded authorization, per `AGENTS.md` |
| Capability level | A level in `capability-maturity.toml` | Evidence that satisfies `tools/check_capability_maturity.py`, approved by Integration |
| Release | `flagquantum/version.py`, tags, release notes | Maintainer |

The distinction that matters is the second row. `tools/check_team_scope.py` refuses a
team-authored change to a protected path, which is why a change that *starts* as an
implementation can become an integration change. The rule is mechanical rather than a
matter of judgment.

## The current maintainer

| Measurement | Value |
| --- | --- |
| Commits on `main` | 214 |
| Authored by `Wei LIU <liuwei.chem.phys@gmail.com>` | 213 |
| Authored by anyone else | 1 (`Qiming Teng <tengqm@outlook.com>`, 2026-09-18) |
| Repository collaborators | 1 (`FlagQuantum`, `admin`) |
| GitHub API contributors | 3 (`FlagQuantum` 1543 commits, `tengqm` 2, `aoyulong` 1) |

The API's contributor counts and the commit authors on `main` do not agree, because a
squash merge attributes a whole pull request to its merger and history has been rewritten
at least once. The git-side figures are the ones that describe `main`.

## Review requirements are nominal

Branch protection on `main` sets `required_approving_review_count: 2`, but
`enforce_admins` is `false` and `require_code_owner_reviews` is `false`, and the
authoring account holds `admin: true`. The measured consequence is that the last 14
merges were authored and merged by the same account with zero reviews.

This is recorded rather than resolved. The two coherent options are to enforce the
requirement — which needs a second reviewer who does not exist yet — or to remove it and
state the actual rule. The current combination, a requirement that is neither met nor
removed, is the option that cannot be audited. It is an open question below.

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
2. **Enforce the two-approval rule, or remove it?** Either answer is defensible; see
   above. Enforcing it requires a second reviewer.
3. **Generate `.github/CODEOWNERS` from `team-ownership.toml`?** There is one obvious
   benefit — GitHub would surface the owning domain in the review UI without any rule
   change — and one obvious cost: a generated file needs a check that it is current, which
   is a new protected tool.
4. **Should the 12 teams be staffed, or renamed?** A team name that maps to no person is
   a truthful description of a path-ownership domain. It is a misleading description of an
   organization, and today the file can be read either way.
5. **What replaces the "one maintainer" state?** The honest answer is more contributors.
   The measurement above is the baseline to improve on, and it is recorded so progress is
   visible rather than asserted.
