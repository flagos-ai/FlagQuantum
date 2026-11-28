# Runtime shard 35: replace an offline slot with an admitted peer

Owner: Integration. Commit this declaration before changing protected CI routing.

## Observed failure and preserved evidence

Published head `d3a07915533f788cda501b42b5e35ea0400ac461`, Main run
`46142914625`, attempt 1, job `140972753464`: GitHub reports runner
`fq-bw1100-14-cpu-pilot` lost communication with the server. It records no
steps, exposes no downloadable job log, and marks the job failed after 600
seconds. API runner 326 is now offline. Absence of server-side steps/logs is not
proof that no local process executed; no scientific result is claimed for this
failed attempt. Retain the original job, annotation, unavailable-log observation
and all other current-run results. Do not rerun that job or administer the runner.

## Narrow routing change

Change only `cpu-runtime.strategy.matrix.include[35].runner` from
`fq-bw1100-14-cpu01-validation` to
`fq-bw1100-2-cpu15-validation`. The recipient is an existing online, idle service
at observation, has the original four locked profile qualifications, and is not
already pinned by another Runtime matrix row or the seven exclusive coverage
workloads. Its current runner is `fq-bw1100-2-cpu15`, API ID 433.

Seven digest-reverified original Runtime/coverage reports across both services
show the same dedicated admission: four effective CPUs and 24 GiB. Runtime's
four workers/one numerical thread remain unchanged. The original qualification
record and locked images remain immutable. This is an equivalent admitted-slot
replacement, not an A800-to-BW reduction, a new service, label or resource quota.
Online/idle status is a timestamped observation, not a future availability claim.

Keep all 128 original shard indexes, hosted buckets, every selected node and
original call phase, timeouts, budgets, seeds, widths, precision, test and product
code, selection hashes, workflow steps and coverage routes. Do not introduce a
failover manager, fallback, new registration or rewritten qualification record.
Only the one routing value and this declaration may change from the published
head. The earlier complete local default (21,959 passed/864 original skips),
source tree and transport proof remain scoped to their exact d3a079155 input.

## Acceptance

Compare complete YAML after normalizing only the one declared routing scalar.
Confirm the recipient remains in the original qualification record and all 96
BW Runtime recipients stay unique. Run the existing routing, hosted capacity and
offline-budget policy cases without weakening assertions. Preserve the prior
complete original shard35 node hash and call inventory; the next head must
execute the entire original workload and independently prove resources and
outcomes. Its own complete CI is required before qualification. Observe the
current run through terminal state, including the new coverage transport; no
failed result, missing log, timeout or over-target duration is reclassified.
No merge, force push, runner administration, workload cut or Rust is included.
