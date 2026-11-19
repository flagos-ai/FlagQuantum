# QEC acceptance evidence revision origin

The shared quality gate rejects
`artifacts/development/qec_colour_distance_acceptance_20261118.json` because its
producer revision has no obtainable ref or origin declaration. This failure
appeared in the compiler routing authorization PR #1218, before any compiler
implementation was included. The artifact entered `main` through QEC PR #1212,
squash-merged as `305b58f5f1d862cdad0770ffdd35dfaf1713ee67`.

The producer is `dc90ce6622b2e6b014a9ae4bc3bf966df13bf1fa`. On 2026-11-18,
both repository-scoped commit endpoints returned that exact object. Enumerating
the current branch and tag refs returned 55 tips. After resolving annotated tags
to commits, comparing the producer with every tip found no tip whose merge base
was the producer: 40 comparisons were divergent and 15 tips were behind it.
The commit is therefore an `unreferenced_object`, rather than a revision from an
unavailable external producing history. A future ref change may require reviewing
this declaration again; these counts describe this inspection, not a permanent
property of the repository.

The inspection uses these existing GitHub API operations:

```bash
gh api repos/flagos-ai/FlagQuantum/commits/dc90ce6622b2e6b014a9ae4bc3bf966df13bf1fa
gh api repos/flagos-ai/FlagQuantum/git/commits/dc90ce6622b2e6b014a9ae4bc3bf966df13bf1fa
gh api repos/flagos-ai/FlagQuantum/git/matching-refs/heads/
gh api repos/flagos-ai/FlagQuantum/git/matching-refs/tags/
```

For each annotated tag, resolve its object with `git/tags/<object-sha>`. For
each resulting commit tip, query `compare/<producer>...<tip>` and check whether
`merge_base_commit.sha` equals the producer. Checking only branches whose head
equals the producer is insufficient: a branch could contain it as an ancestor.

The repair adds one artifact-scoped row to `evidence-revision-origins.toml`.
It does not alter the numerical artifact, its recorded producer or base commit,
the QEC implementation, or any provenance checker. The unchanged artifact's
SHA-256 is `f6afd3bd07f93275dd83bccbd292399d78c436d43b92ebd2977e7ab9a2783659`.
This disclosure does not certify the recorded scientific results or make an
unreferenced commit part of ordinary clone history.

The affected existing contributor workflow is:

```bash
python -m pytest tests/unit/test_evidence_revisions.py -q
```

In a complete clone with the current remote refs, the expected result is that
this artifact no longer causes the revision-accounting test to fail. Other
evidence, formatting, dependency, and runtime checks retain their existing
requirements. Local validation must distinguish synthetic gate behavior from
complete-clone provenance acceptance; a partial local history cannot prove the
latter by itself.

Local validation on 2026-11-18: TOML parsing and the artifact hash check pass.
Using the object/ref answers from the inspection above in the gate's existing
resolver caches, artifact-scoped validation changes from one error without the
row to zero with it; the gate still rejects that row if a ref reaches the
producer. This isolates the declaration's behavior without claiming a complete
local history. The full evidence-revision test file reports 57 passed and one
failed: the checked-in history test reports 98 other revision/ref discrepancies
in this development clone. Complete-clone CI acceptance remains required.
