# Runtime duration balancing

The existing runtime tier can exhaust its job budget after pytest succeeds,
leaving the required launcher checks unexecuted. This integration-only change
updates the existing duration hints from recorded observations. It changes no
test, selection marker, acceptance threshold, worker count, job timeout or
production code.

## Observed failure

[Run 44487669516, job 135479850494](https://github.com/flagos-ai/FlagQuantum/actions/runs/44487669516/job/135479850494)
tested source `abde94db5cef007495cc1bce41ce36f673747cff`, attempt 2.
The integration command completed with 2,829 passed and 255 skipped in
3,554.88 seconds. The 60-minute job then cancelled before the two required
launcher checks could complete. This was not a passing job, even though its
pytest step reported no assertion failures.

Four resiliency-planning tests consumed 1,295.60 to 1,439.62 seconds each.
Their node IDs had no recorded duration and were therefore priced at the
0.05-second default by the existing scheduler. The log's slowest 50 entries
supply 49 previously unknown node IDs and refresh one existing measurement.
The hints retain the observed source/run/job, pytest outcome and normalized log
hash. Setup time is added when a node has a recorded setup entry. Unreported
phases are not inferred. These are shared-runner observations, not intrinsic
algorithm costs or evidence of a compiler performance regression.

## Scheduling check

A fresh collection at main `45468b934a18e552a2a69dcc0b8f1db0d8fa16f9`
on macOS ARM64/Python 3.12.14 selected 12,152 integration tests, with optional
collection skips. Replaying the unchanged four-shard scheduler with old and
updated hints preserves every selected node exactly once. All 50 measured
nodes exist in this collection. The four longest resiliency cases now occupy
four different shards.

[The retained plan comparison](evidence/runtime-shard-duration-balance.json)
records both assignments and counts. The known measured duration sums range
from 1,630.66 to 5,782.78 seconds before the update and from 3,588.39 to
4,135.77 afterwards. These sums describe only 50 observed nodes reassigned
into a different collection. They are not wall-clock predictions: workers run
concurrently, other tests have unknown costs, Linux collection can differ, and
shared-runner contention is not controlled. In particular, the local old plan
is not claimed to reproduce the failed Linux shard's exact assignment.

## Existing contributor workflow and verification

```bash
python tools/ci_tier.py pr-runtime --shard-count 4 --shard-index 0
python -m pytest tests/unit/test_pytest_shard.py \
  tests/unit/test_ci_coverage_plan.py -q
```

The runtime command keeps the same selection and invokes the same scheduler;
only the recorded cost inputs change. Run all four shard indices to execute
the complete tier. The focused scheduler and coverage-plan tests passed
18 cases in 0.55 seconds, retaining the existing disjointness, coverage and
balance bounds. Full CI, including both launcher proofs, is still required.
The cancelled job must not be reclassified as passing.


## Source provenance after integration

The observed source `abde94db5cef007495cc1bce41ce36f673747cff` belonged to
the former compiler cleanup branch. GitHub still serves the full object by
commit ID, but the original branch was deleted after squash merge and the
commit is not an ancestor of main. Its origin is declared as
`unreferenced_object` in the existing evidence-revision manifest. The original
failed run, source pin, log hashes, measured durations and scheduler evidence
remain unchanged. A full-history CI checkout detected the missing declaration;
adding it accounts for the source rather than weakening the provenance gate.
