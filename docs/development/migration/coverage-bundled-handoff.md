# Coverage plan and source handoff

Owner: Integration. Commit this declaration separately before implementation in
existing PR #1360; this repair does not create another migration PR.

The original attempt for head 710491542 recorded coverage shard 21 at 603 seconds
(job 141022090941, run 46158710226). It passed the unchanged complete 1,070,000-draw
study. The test step took 515 seconds, while the job separately downloaded its
source and coverage plan and installed the exact checkout offline. Retain that
successful but over-target original observation; do not rerun the same source to
seek a favourable duration.

Remove the redundant artifact download from every coverage recipient. In the
coverage-plan producer, retain its verified initial incremental source.sha and
source.pack before any full-source fallback can replace them. Publish those exact
bytes beside the original plan.json and all 256 unchanged shard text files in the
existing coverage-plan artifact. The producer must reject a mismatched source
identity or empty/missing initial pack. Do not package a full fallback pack as the
incremental bootstrap.

Coverage recipients download that existing exact-source coverage-plan artifact
through the same bounded initial attempt and isolated retry used for source.
Retain digest-mismatch rejection, the same GITHUB_SHA check, complete-tree and Git
object verification, exact bootstrap helper, independent GitHub source repair,
full-source fallback and source checkout. Read the plan from that already
verified transfer; delete the second plan-download step. Keep all other source
consumers on their existing source artifact, without changing their expanded
steps. No YAML merge keys, new executable wrapper, second planner, or duplicate
source authority is needed.

Preserve every original test node, phase, fixture invocation, RNG draw, seed,
width, optimizer/sample budget, clock, result/status, scientific input, coverage
policy and timeout. Preserve the existing 128 Runtime and 256 coverage
partitions, all 83 scientific singletons, runner selectors and actual admitted
CPU/memory/worker/thread resources. No native compiler, runtime implementation,
image pin, dependency, runner service or source repair policy changes.

Exercise the real producer shell scripts with valid and invalid bootstrap
inputs, and exercise retry promotion with the complete bundled files. Verify
actual old-run source/plan bytes survive packaging, existing cold full-source
fallback retains the plan, and the plan parser rejects stale source identities.
Run all original source, transport, coverage-plan and offline-environment policy
consumers plus the static gates. Next-head own CI must independently verify all
original reports, call phases, resources and whole executed job times including
prepare/upload. Green checks and the under-600-second target are separate facts;
report both and retain every over-target observation. No merge, force push,
scientific reduction, Rust, or runner administration.

The same attempt failed coverage aggregation after verifying 208/256 databases
in its 180-second download window. Increase only the bounded artifact I/O worker
count from eight to sixteen. Preserve the 180-second overall deadline, fifteen-
second request timeout, at most two transient retries, exact run/head/attempt/
source/index set, HTTPS credential isolation, byte bounds, CRC/SHA checks and
atomic complete-set publication. CPU/memory admission and scientific workers
remain unchanged. Independently replay the failed run's immutable 256 artifacts
with observed active I/O, original ZIP/database hashes and SQLite integrity.
This is transport diagnosis, not a CI/test rerun or runner/hardware qualification.

## Existing temporary directories on a reused recipient

Own65 CI exposed a recipient with an existing RUNNER_TEMP/coverage-plan directory
(job141032922089). A fixed name must not assume prior job cleanup succeeded.
Use freshly created task-owned temporary roots for the producer bootstrap, the
producer plan and the recipient plan. The plan writer creates a child below its
fresh parent; publish only successful step output paths and keep source fallback
independent from the preserved plan. Do not remove or overwrite old directories,
relax Git/source/plan checks, or change original scientific inputs, selection,
resources, clocks or output identities. Exercise existing misleading fixed-name
directories in the real producer/recipient script regressions.

The core312 partition also exposed an unsynchronized policy consumer that still
looked for the removed second plan download. Synchronize that test with the
bounded initial/retry/promotion graph and successful fresh plan output, keeping
its digest, strict retry, path isolation and two-consumer assertions. Update the
stale eight-transfer tool documentation. These remain the same declared PR1360
handoff repair; preserve all current65 observations before another publication.
