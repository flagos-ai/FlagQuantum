# PR checks and main acceptance

## Current policy

Pull requests targeting `main` or a `refactor/qec-restructure-*` stack base run
the existing complete CI job graph once. QEC stack targets receive the same
automatic checks as main-target PRs; manual runs remain supplementary evidence.
`CI complete` depends on every CI job, including source hygiene (`pre-commit`),
compatibility matrices, packaging, coverage, optional SDKs and distributed proofs.
A failed, cancelled or skipped dependency fails the aggregate. Required jobs are
never filtered out by changed-file paths.

Main protection requires the GitHub Actions `CI complete` check and an up-to-date
base, including for administrators. Review approval count remains zero for the
current administrator-driven workflow. Force pushes and branch deletion remain
prohibited. The merge queue is not required. When ordinary auto-merge is enabled,
a specifically authorized PR merges after its required checks succeed.

If another PR advances `main`, update the branch and validate that new integration
before merging. A single administrator can still have concurrent PRs; strict
checks retain this protection. New pushes to one PR cancel its obsolete run.

## Workflow ownership

- `ci.yml`: full main/QEC-stack pull-request CI and manual full validation. It has no `push` or
  `merge_group` trigger. Test selections and thresholds are unchanged.
- `pre-commit.yaml`: manual source-hygiene diagnostics. Both it and the CI job use
  `.github/actions/source-hygiene`, so there is one implementation of the hooks.
- `local-gpu.yml`: one automatic GPU workflow per PR, plus manual dispatch.
  Branch pushes and queue refs do not create duplicate hardware workflows.
- `main-smoke.yml`: a small acceptance job for every main push. It builds and
  inspects a CPU wheel, installs it into a private environment, verifies that the
  import comes from that environment and the native extension loads, then runs
  the existing minimal circuit and autograd smoke tests outside the source tree.
  Changes to this workflow or its smoke tests also validate the job on the PR.
- Existing scheduled hardware and release workflows keep their validation.
  Manual full CI retains the non-PR interpreter and slow-coverage selections.

Hosted jobs retain their capacity buckets. Self-hosted CPU capacity remains
bounded by provisioned runners rather than a workflow parallelism cap. Device
locks protect shared hardware measurements. A runner may be idle after all
eligible jobs have started or completed; tasks are not automatically subdivided.

## Safe settings migration

1. Validate the workflow diff and aggregate failure cases. Run full PR CI and the
   new installed-wheel smoke job on the candidate revision.
2. Read and save the live branch protection, ruleset and queued PR identities.
   Preserve unrelated protection fields and concurrent changes.
3. After candidate checks succeed, replace the classic required contexts with
   GitHub Actions `CI complete`, keeping strict checks and administrator
   enforcement. The existing ruleset already requires that aggregate.
4. Remove only the merge-queue rule from the existing ruleset, preserving its
   strict aggregate requirement and empty bypass list. The reviewed target
   configuration is `ci/pr-gate-main.json`; update the existing ruleset instead
   of creating a second rule authority.
5. Enable ordinary repository auto-merge. Preserve merge authorization for
   existing queued PRs when converting them to ordinary auto-merge; do not enable
   it on unrelated PRs.
6. Merge the CI change through the protected PR flow, without administrator
   bypass. Verify its main acceptance run. Record remote settings and run URLs.

The prior queue rollout used one build at a time and repeated full validation on
PR, merge-group and main-push events. Historical run links remain in the PRs;
this document describes the replacement policy rather than activating settings
merely by being checked into the repository.

## Contributor commands

After a PR has merge authorization:

```bash
PR=123  # replace with the authorized pull request
gh pr checks "$PR" --required
HEAD=$(gh pr view "$PR" --json headRefOid --jq .headRefOid)
gh pr merge "$PR" --auto --merge --match-head-commit "$HEAD"
```

Expected: required checks and strict base freshness remain enforced. A failing
check blocks the merge; a stale branch must be updated and checked again.

For an explicit full validation of the current default branch:

```bash
gh workflow run ci.yml --repo flagos-ai/FlagQuantum --ref main
```

Expected: the complete non-PR matrix runs independently of the short main smoke.
The FlagQuantum SDK workflow and public APIs are unchanged.
