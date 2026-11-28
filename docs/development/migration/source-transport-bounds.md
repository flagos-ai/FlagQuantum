# Bound source transport and replace an offline serial recipient

Owner: Integration. Commit this declaration before the protected implementation.

## Recorded failure

PR #1360 head `d3a07915533f788cda501b42b5e35ea0400ac461`, Main run
`46142914625`, attempt 1: five jobs lost runner communication without usable
server-side step/test records. One remaining source download on
`fq-bw1100-12-cpu-pilot` stayed in the official incremental-artifact action for
more than twenty minutes; another job waited for the offline
`fq-bw1100-14-cpu16-validation` selector. The already failed owned run was
cancelled only after preserving job/annotation/state evidence. All failed,
missing, cancelled and over-target results remain unqualified. No other run or
runner was cancelled or administered.

## Complete source handoff

The actual consumer inventory contains 41 raw source-artifact download steps:
35 in `ci.yml` and six in `local-gpu.yml`, all lacking individual timeouts.
Bound incremental downloads to one minute and full-source downloads to two.
Allow exactly one independent retry, bounded to two minutes, in a clean,
different directory with the same exact artifact name and `digest-mismatch:
error`. The retry must fail the job if unsuccessful. Promote only a successful
retry to the original canonical source directory, preserving the abandoned
first directory separately for the job. Keep all original source SHA/tree/blob,
Git-object closure, cache/repair/full fallback, checkout and cleanup checks.
No test starts from an unverified partial transfer.

Reuse identical step definitions with standard YAML anchors and aliases,
supported by GitHub Actions, rather than duplicating the same definitions in
every job. Do not use merge keys. Expansion must preserve the exact per-job
conditions, artifact selector, destination and non-transport steps. See
[GitHub's workflow reuse reference](https://docs.github.com/en/actions/reference/workflows-and-actions/reusing-workflow-configurations).

Keep the source-bootstrap and distribution-verifier bodies byte-exact. Add no
new HTTP client, generic manager, remote self-repository action, package,
registry, old executable path or dependency. The earlier coverage REST transport
and its original-artifact proof remain scoped to their actual published input.

## Same-admission serial recipient

Change only the existing `_SERIAL_BW_RUNNERS` recipient for the complete
`test_invalid_singles_angles_fail_before_execution` workload from
`fq-bw1100-14-cpu16-validation` to `fq-bw1100-2-cpu16-validation`. The new recipient
is an existing qualified peer and is not pinned by another Runtime or serial
coverage workload. Keep four effective CPUs, 24 GiB, one worker/four numerical
threads and every original node, call, budget, seed, width and precision.
Do not rewrite historical qualification records or change a runner/service/label.
The independently committed Runtime35 same-admission replacement remains in
place, with its original 143-node hash unchanged.

## Acceptance

Compare parsed/expanded YAML while normalizing only declared source transport
steps and Runtime35's recipient. Compare the complete planner AST while
normalizing only the one serial recipient string. Existing routing/source/
offline-policy cases must retain their assertions' meaning. Exercise real Bash
promotion with both source kinds, present/absent partial data and collision
refusal; validate same selector/digest, independent retry paths, bounded statuses
and no partial-success path. Run source quality and the existing static gates.
The next source must independently execute all original Runtime/coverage nodes,
scientific calls, resource checks, installed-source and complete CI gates with
all executed jobs below 600 seconds before qualification. Prior local default,
cold source and immutable-original-artifact proofs are not relabelled as that
new head's acceptance. No merge, force push, workload cut or Rust is included.
