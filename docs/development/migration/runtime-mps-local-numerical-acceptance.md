# Runtime MPS local numerical acceptance

Status: formal source and native installed artifact qualified. Parent #1339
`6ea206c3b4aa82fe73f33e8515f018d2fef7a1ec` passed all applicable checks
(243 successful, 3 existing conditional skips). This child still
requires its own exact published-head combined CI.

## Source proof

Four product files gain five private numerical functions in one existing
Simulation module. The original sixteen qualified lexical definitions were
reviewed; eleven non-target definitions remain AST-identical. Five normalized
numerical donor checks pass. Existing signatures and independent boundary-gate
validation remain unchanged. Original tests and scientific assertions are not
rewritten. Evidence-only tensor allocation is deliberately retained at Runtime.

## Replacement and original-parent comparison

Five defining-owner replacement cases cover CPU sites/gradients, actual VJP
cotangent inputs and both Runtime collective callbacks. The collective contract
fake checks wiring only; it is not hardware or scalability evidence.

The task-local optional environment uses JAX/jaxlib 0.10.2, within the
repository's declared `jax>=0.10,<0.11` range, and does not modify the shared
environment or production dependencies. The final CPU/JAX focused run passes
120 tests, with six existing hardware skips, including all five defining-owner
replacements. Eleven preview quality gates pass. An earlier preview had two
additional evidence-tensor replacements; those are outside the reviewed final
scope and are not counted here. One initial new optimizer assertion targeted
the retained backward gradient field instead of optimizer ownership records;
that fixture was corrected without changing production behavior or old tests.
Original logs remain available. Tests skipped without optional JAX are not
counted as passing numerical execution.

The original-parent fixture covers 360 positive executions and 22 intended
refusals. It preserves all result fields and mapping order. Runtime clocks are
explicit deterministic fixtures advancing 0.125 seconds per event; those
fixtures certify data flow, not measured timing or performance. The CPU-vmap
inner rank kernels exercise pure numerical expressions and named-axis exchange,
not actual accelerator pmap or multi-GPU capacity. Original snapshots are
immutable and comparison failures must remain recorded.

## Local resource interruption

The first full Unit/API/Team attempt was interrupted by local ENOSPC and
Python log-flush exit 120; its raw log and incomplete result are retained.
After the failed task process group had exited, only its temporary fixtures
and rebuildable task caches were removed. The complete three-phase run then
used pytest builtin `tmp_path_retention_policy=failed` and a task-owned base
directory, keeping the same selections, four workers, assertions, scientific
samples and 900-second phase bounds. Disk-space heartbeats are retained.
No original parent snapshots, contracts, raw incoming sources or wheel artifacts
were removed or normalized. A first isolated-basetemp startup also stopped
before test execution because its parent directory had not been created;
that startup trace is retained and directory preparation was corrected.
These local environmental retries are separate from CI.

## Complete formal source regression

Three phases ran in the actual Git checkout with the same parent dependency
environment, four workers and unchanged credential/hardware conditions.
The separate task-local optional-JAX phase additionally passed 120 tests,
with six existing hardware skips, in the actual checkout. The base Unit phase
has 957 inherited conditional skips plus one new optional-JAX VJP replacement
case; that case passes in the mandatory JAX phase, together with all five
replacement cases. No test selection or scientific assertion is removed.

| Phase | Passed | Conditional skips | Pytest seconds |
| --- | ---: | ---: | ---: |
| unit-api-team-final | 19429 | 958 | 386.37 |
| qec-full | 2801 | 329 | 75.7 |
| root-local-integration | 1592 | 223 | 87.84 |

Eleven formal quality and five additional gates pass. Publication also requires
five pinned source hooks, four final document/diff checks and exact commit
source/wheel/installed binding. Existing tests and frozen evidence are unchanged.

## Native installed artifact

The new native wheel SHA-256 is
`f0f6dc2e71396f12f44b9602ccf5b624b965e428bec037999c2a2b75aad5141e`.
All 1,262 product Python files match source, wheel and clean installation
byte for byte. The installed artifact passes 120 tests with six existing
hardware skips, with all 329 loaded product modules verified from its own
installation and no product source in the test root.

All 382 original-parent scenarios match byte for byte, including all result
fields, derivative/state data and refusal messages. Snapshot SHA-256 is
`8c21f7c56b806f572b89b01257b7b813bca3be1d9b87e70cb24e88ce461a2321`.
Eighteen separate CPU scenarios also match with JAX/JAXlib imports actively
blocked and neither package loaded. Cold catalog, both native extensions and
CLI probes have separate acceptance gates.

All 11,916 other parent blobs, including 1,174 frozen contract/fixture blobs
and 3,046 original test-tree blobs, remain identical in the preview. No old
assertion or scientific fixture is changed.

## Minimal-dependency collection followup

The complete three-phase source counts above were recorded at migration commit
`a6d4cc284fe61274fbecc61ba465b0fa0ac48205`. Its Python 3.10 and 3.11
minimal CPU smoke jobs subsequently failed during collection because this new
unit module imported optional NumPy before pytest selected its markers.
Both original failure logs are retained. Product numerical execution had not
started in those failed collection jobs.

The followup changes only this test's dependency admission and this record.
NumPy is loaded by an execution-time fixture, and the fake ready-array type is
constructed inside its collective contract factory. Unit markers, all five
test cases and every original assertion remain unchanged; no module-level skip,
test filter, mandatory dependency or scientific threshold is added.

A simulated NumPy import refusal reproduces the original collection error and
accepts the fixed module with no NumPy import attempt. This probe initializes
Torch first and is not a clean missing-dependency environment certificate.
The smoke/base-unit supplement passes eight tests with one optional JAX skip.
Real CPU/JAX focused acceptance passes 120 tests with six existing hardware
skips, including all five defining-owner replacement cases. Both Black version
bounds and Ruff pass. The 1,262 product Python files and qualified native wheel
remain byte-identical; the exact new-head CI must validate the real minimal
Python 3.10/3.11 environments.

## Remaining acceptance

The child still requires its own exact published-head combined CI. Final source
hooks, document gates and commit/artifact binding are enforced before
publication. Parent or local qualification never replaces that exact-head check.
