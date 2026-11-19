# Toric QEC evidence revision origin

The compiler routing PR #1227 quality job fails because
`artifacts/development/qec_toric_distance_acceptance_20261118.json` records two
producer commits that an ordinary clone cannot obtain. The artifact was already
present on the compiler branch's main baseline; this is an independent
Integration-owned provenance repair, not a compiler acceptance exception.

| Artifact field | Recorded commit | Repository commit endpoint | Current containing branch/tag tips |
| --- | --- | --- | ---: |
| `commit` | `7f89948efd1d6dc63b29d00934cc6738cd440829` | Returned exact object | 0 of 55 |
| `full_regression.commit` | `29aa778ab5604059c5a4337f9b250c6c4b606d9c` | Returned exact object | 0 of 55 |

Measured on 2026-11-19 using the repository-scoped GitHub API. Enumerate
`git/matching-refs/heads/` and `git/matching-refs/tags/`, resolve any annotated
tag with `git/tags/<sha>`, then query `compare/<producer>...<tip>` for every
producer/tip pair. A tip contains the producer only if
`merge_base_commit.sha == producer`. All 110 comparisons completed and none
satisfied that equality. The commit endpoint `git/commits/<producer>` returned
the exact producer object for both revisions. These are therefore
`unreferenced_object` origins under the existing gate, rather than dependency
commits or inaccessible producing-host history. The counts describe this
inspection, not a permanent property of changing remote refs.

The initial local Git object probe timed out on an unrelated missing branch tip
in this partial development clone. It is not evidence of unreachability. The
successful remote API comparisons above establish the classification without
using that partial-history result.

The repair adds two artifact-scoped declarations. It leaves the recorded values,
numerical evidence, QEC code and provenance checker unchanged. It does not
certify scientific correctness or make the producer commits part of a normal
clone. Future branch/tag changes may require reevaluating the declarations.

Existing contributor workflow:

```bash
python -m pytest tests/unit/test_evidence_revisions.py -q
python tools/check_evidence_revisions.py
```

Expected result in a complete clone with current refs: both toric producer
origins are accounted for, and these two records no longer fail the existing
quality gate. In this partial development checkout, numerous unrelated stale
refs already cause the full repository provenance scan to fail; local isolated
declaration tests are not reported as complete-clone acceptance. Required CI
must pass before normal merge.

Local isolated validation uses the measured object/reachability answers in the
existing validator caches: the two missing-origin errors become zero with the
declarations, and both declarations are rejected if their commits become
reachable. The raw artifact matches its checked-in bytes; its SHA-256 remains
`e0ca2bf7c6f4a7cfdbf2b6358769839f1e33d2ba8ec97dfd6531bc4655b1b4eb`.

The existing evidence-revision test file reports 57 passed with the single
ambient-history test deselected. This verifies validator behavior, not complete
remote-history acceptance; the latter remains a required CI check.
