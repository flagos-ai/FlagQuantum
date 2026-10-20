# Main merge queue rollout

## Status and purpose

Workflow support is implemented in the existing CI and pre-commit workflows.
The repository setting is **not enabled by this change**. The reviewed template
in `ci/merge-queue-main.json` has enforcement disabled. Enable only after the
workflow PR passes its latest checks, is merged through the existing protection,
and its main workflow runs finish successfully. Local configuration tests alone
are not evidence that GitHub dispatched or completed a real merge-group run.

Frequent Update branch operations create new heads and cancel superseded PR
runs. A native merge queue validates a temporary integration revision containing
main and the queued change, without repeatedly rewriting every PR branch. It does
not reuse a successful result from a different revision or eliminate all rebuilds:
removing or reordering queue entries can require another integration run.

## Reuse and check preservation

- Both existing workflows accept merge_group.checks_requested targeting main.
  No parallel workflow, queue runner, result copier or additional dependency is
  introduced. Default checkout validates the merge-group SHA, not the PR head.
- All existing jobs, compatibility matrices, package checks, coverage floors,
  optional SDK checks and distributed proofs remain. The required names stay
  quality, cpu-core (3.10), cpu-core (3.11), cpu-core (3.12), package,
  distributed-cpu, coverage and pre-commit.
- The additive CI complete job waits for all twenty-one other CI jobs, requiring
  success from each, and reuses a light-a bucket. The staged ruleset adds it as
  a GitHub-Actions-bound requirement beside the original eight. Native queue
  merging must not turn the prior full-workflow acceptance into eight checks
  passing while an SDK or wheel lane still runs. Pre-commit remains separately
  required.
- A queue event follows the current non-PR selection: full smoke/unit on Python
  3.10/3.11 and full maintained-runtime coverage including slow cases. Python
  3.12 unit/runtime, package and other lanes retain their existing commands.
- Both change-based quality gates use the queue base SHA. A missing or zero
  queue base fails instead of skipping ownership or evidence-retention checks.
- Workflow concurrency distinguishes queue SHAs. Cancellation stays limited to
  superseded PR runs. Five GitHub-hosted buckets bound the hosted share; eight
  A800 CPU runners bound the CPU share. The organisation's hosted runner limit
  is unchanged; merge-group refs isolate unrelated integration candidates.
- Existing strict branch protection, required checks with their GitHub Actions
  app binding, administrator enforcement, review policy and force-push/deletion
  restrictions remain. The added ruleset contains the queue rule and only the additive CI complete
  requirement, with no bypass actors. It neither replaces nor copies the
  original status-check authority.

## Staged activation

1. Validate the workflow diff, shell gate cases, event routing, YAML and existing
   policy tests. Run the repository's default and CPU distributed/contract tiers.
   Require all latest-head CI checks and complete workflows to succeed before
   merging the CI PR. Do not use an administrator bypass.
2. After merge, verify both workflow definitions are on main and the main CI and
   pre-commit runs are successful. Re-read current protections and rulesets;
   preserve concurrent changes and do not create duplicate queue rulesets.
3. Confirm CI complete is present and successful on main. In-flight PRs may
   need one bootstrap synchronization to acquire the new workflow; a missing
   check must block admission rather than receive a fabricated status. Apply
   the reviewed template with enforcement changed to active, keeping the
   existing branch protections intact. Queue builds start at one concurrent
   candidate with one entry per merge, ALLGREEN and a 120-minute response timeout.
   This bounds runner demand and does not relax any test or required status.
   MERGE preserves original development commits and their authors; GitHub owns
   the metadata of the resulting merge commit.
4. Verify the live rule, no bypass actors, the unchanged required checks and
   strict setting. For the first separately authorized PR, observe an actual
   merge_group run at its integration SHA. Verify every required check and the
   complete CI/pre-commit workflows before treating the rollout as proven.
   A pending, skipped, missing or cancelled result is not success.
5. Record the activation rule id, configuration, integration SHA and run URLs in
   the CI PR. If event dispatch or status names fail, remove the candidate from
   the queue, fix the workflow and revalidate. If activation must be rolled back,
   disable only the added queue rule; retain the original strict protection.

No PR is enrolled automatically by this change. In particular, monitoring #652
and implementing its approved contract do not authorize its merge. Native queue
entry can lead directly to a merge, so use it only for a separately authorized PR.

The template can be inspected without changing GitHub settings:

```bash
python -m json.tool ci/merge-queue-main.json
```

Expected: enforcement is disabled, the queue strategy is ALLGREEN, its build/merge limits are one, bypass_actors
is empty, and the only added status requirement is CI complete.
An operator should read current remote state before applying the active version;
this repository does not add a second settings-management framework.

## Contributor and user workflows

After activation, finish the review and required checks, then use the PR's
**Merge when ready / Add to merge queue** control. Avoid Update branch on every
waiting PR. In the CLI, after a specific PR has merge authorization:

```bash
PR=123  # replace with the authorized pull request
HEAD=$(gh pr view "$PR" --json headRefOid --jq .headRefOid)
gh pr checks "$PR" --required
gh pr merge "$PR" --match-head-commit "$HEAD"
```

Expected on a queue-protected branch with successful required checks: GitHub
queues the PR, reports new checks on the integration revision, and merges only
when its protection requirements pass. A conflict or failing check blocks it.
This is the built-in GitHub queue, not a FlagQuantum SDK interface.

The existing local quantum-computing workflow is unchanged:

```python
import torch
import flagquantum as fq

circuit = fq.Circuit(1, dtype=torch.complex128).h(0)
probabilities = circuit.state().abs().square()[0]
print([round(value, 6) for value in probabilities.tolist()])
```

Expected: `[0.5, 0.5]`. No queue operation or network access is needed by SDK users.

## References

- [GitHub merge queues](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue)
- [merge_group workflow event](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#merge_group)
- [Repository ruleset API](https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset)
