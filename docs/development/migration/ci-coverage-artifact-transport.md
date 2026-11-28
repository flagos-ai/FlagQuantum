# Coverage artifact transport repair

Owner: Integration. This declaration precedes implementation changes to the
protected CI surface. Continue PR #1360; do not rerun its failed attempt.

## Observed failure

Head `bc8d135aea648738a006e16c523f4410ec040c67`, actual source
`bc7ddb9fe159e2db2c5740f973e826bf84584d25`, Main run `46137997651`, attempt 1:
all 394 original test partitions passed, maximum 589 seconds. The coverage job
`140963979044` failed after 601 seconds. Its first data-download composite hit
120 seconds and its second hit 180 seconds. Combining data and enforcing the
coverage policy did not execute. CI complete therefore also failed. Four other
applicable workflows passed. Do not label this head qualified.

The locked official `actions/download-artifact@v8.0.2` is also the latest release
at inspection; official main has identical source. It invokes
`artifactClient.downloadArtifact` for every selected artifact before chunking
those promises into groups of five. The eager starts are verified in source;
their contribution to this failure remains an inference, not measured physical
network utilization. Preserve the full failed log and all original reports.

## Authorized replacement

Replace only coverage's multi-artifact transport with a narrow standard-library
CI tool at `tools/ci/ci_coverage_artifacts.py`. Bound actual network initiation,
not just waiting on previously started work. Use GitHub's supported REST artifact
API with read-only actions permission; reuse established CI network/redirect
behavior where it meets this boundary. Never forward authorization across hosts
or permit an HTTPS downgrade. Keep tokens and signed artifact URLs out of logs.

Run network transport with the host interpreter: the existing offline CPU
Python wrapper intentionally has no external networking. Select only the exact
current run and all expected `coverage-shard-<index>-<source-sha>` artifacts.
Validate source/run identity, complete index set, uniqueness, non-expiration,
API SHA-256, ZIP member identity and safe regular-file extraction into a fresh
private staging directory. Bound response size, per-request time, retries and
whole command time. Preserve already verified original shard bytes. Fail closed
on missing/extra/duplicate/corrupt/wrong-source artifacts and publish no partial
successful output. A retry must not reuse an unverified partial file.

Keep the original count bounds, exact `.coverage.<index>` set, read-only SQLite
integrity checks, coverage combine/XML generation, policy check, floors, source
attribution and final report retention. Do not alter test node lists, sharding,
markers, call phases, scientific inputs, runner labels, numerical budgets,
resource profiles, dependencies, frozen public contracts or native inputs.
The single-source bootstrap download and unrelated action consumers retain their
current implementation; do not patch runner-owned action caches or vendor a
modified SDK distribution. No runner administration is in scope.

## Acceptance

Exercise actual bounded-start concurrency, exact run/source/index selection,
corrupt/different/unsafe ZIP and digest refusal, transient-only bounded retry,
credential stripping and fail-closed incomplete output. Retain the failed
original execution; use its immutable original artifacts to qualify the transport
without rerunning tests. Compare every restored SQLite file byte-for-byte with
its own SHA-256-bound original artifact. Run the original CI-policy/source
consumer tests, ownership/architecture/hygiene/documentation checks and source
quality checks. Preserve every old assertion's meaning. The changed CI head
requires its own original selection/scientific/resource and installed-source
qualification, every applicable check green and every executed job below 600
seconds including prepare/upload, excluding queue. No favorable rerun substitutes
for that new-head qualification.

This repair does not complete pytest/Coverage configuration consolidation,
conftest/reference-tool ownership, artifacts, #1084 integration, native ABI,
optional developer distributions or G0. Rust remains deferred. No merge or force
push is authorized.
