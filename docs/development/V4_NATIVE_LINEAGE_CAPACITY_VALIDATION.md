# Bounded v4 native lineage capacity validation

The approved two-budget change removes the v4 evidence bottleneck for the two
inspected DNN artifact workflows. All six inspected workflows pass the unchanged
SDKBench evaluator. This is local development evidence on known inputs, not a
held-out result, an SDK ranking, or formal resource qualification.

## Implementation and compatibility

The [approved proposal](../api-changes/FQ-V4-NATIVE-LINEAGE-CAPACITY-20261123.md)
changes only native input records and aggregate rewrite input/output memberships
from 4,096 to 8,192 in v4 constructors and readers. The implementation revision
is `ceccc29816014a43d3ee1fcbf4445433c3a6846c`, built on the separately approved
source circuit payload fix in PR #1326. The source-capacity dependency must land
first; this change does not broaden that proposal.

Physical outputs, transitions, rewrite records and ancestry indices retain their
4,096 caps. Per-block inputs/outputs retain 64/96 limits; plan and bundle byte
limits remain 16 MiB. Canonical serialization, hash inputs, schema versions,
public signatures and v1-v3 readers remain unchanged. Existing golden identity
checks pass. Readers must upgrade before producers send expanded v4 evidence;
older readers reject it. No existing artifact needs conversion.

## Boundary and safety checks

225 focused tests passed after formatting (4.97 seconds), including constructor,
dictionary and JSON round trips, both exact 8,192 boundaries, 8,193 rejections,
early raw-array rejection, independent unchanged 4,096 caps, per-block budgets,
ancestry bounds, old identities, hash/membership/layout/schedule tampering, byte
limits and automatic target artifact workflows. The protected public API
snapshot checker passed. Two existing small-budget tests now patch the owning
membership budget rather than the unrelated shared record constant; their
rejection assertions are preserved. New exact-size tests use actual limits.

The synthetic structural cases describe canceling adjacent CZ pairs and optional
zero-angle RX identity outputs. Core evidence has no gate angles and checks
structure rather than quantum equivalence. Complete quantum semantics belong to
the existing Compiler and independent SDKBench evaluator, not this parser test.

## Six inspected application inputs

SDKBench is pinned unchanged to
`5ba03de33f8a9193f86b26d2cb3fa841ede7c98a`. Adapters, numerical acceptance,
resource policy and fixtures were not modified. Its revision is explicitly
classified as an external dependency in the evidence metadata. Each workflow ran in a fresh
process with a 20-second wall limit. Original six-workflow batch results were
also retained locally; the checked-in evidence records the fresh-process run.
The fixture identifiers and SHA-256 values bind the exact inputs.

| Input | Topology | Result | CZ count | Two-body depth | Semantic check | Maximum error |
| --- | --- | --- | ---: | ---: | --- | ---: |
| DNN, 8 wires | grid | accepted | 183 | 86 | three probes only | 2.4709e-15 |
| DNN, 8 wires | line | accepted | 190 | 82 | three probes only | 3.4758e-15 |
| QAOA, 6 wires | grid | accepted | 66 | 40 | full operator | 2.6946e-15 |
| QAOA, 6 wires | line | accepted | 90 | 58 | full operator | 2.7777e-15 |
| Simon, 6 wires | grid | accepted | 36 | 33 | full operator | 2.1184e-15 |
| Simon, 6 wires | line | accepted | 34 | 31 | full operator | 1.7772e-15 |

The earlier rejected DNN attempts remain in
[the source-capacity workflow record](evidence/program-artifact-delivered-workflows.json)
and [the original record-count diagnostic](evidence/program-artifact-v4-record-counts.json).
No validator was bypassed in this run. DNN probe acceptance is weaker than a
complete operator comparison and is reported as such. The other four results
retain their earlier gate counts and depths. No speedup or memory regression
comparison is inferred from these uncontrolled wall times.

## Observed resource guardrails

[Resource observations](evidence/v4-native-lineage-capacity-resources.json) retain
all 18 fresh-process results: six old/new/over-limit cases repeated three times.
Each process constructs evidence, and accepted cases serialize and read it back.
The predeclared development guardrails were a 20-second process wall limit and
512 MiB observed peak RSS. All cases passed their expected acceptance/rejection
and those guardrails. RSS comes from the operating system, not traced Python
allocations. The new native boundary uses a 1,787,165-byte bundle; the new
membership boundary uses 1,826,598 bytes, both below unchanged byte caps.

macOS does not enforce a memory cgroup here. The host was not exclusively
reserved and other local verification was running. These are development
observations, not a hard memory guarantee or a substitute for SDKBench's Linux
host qualification. The local guardrail is not a new SDKBench threshold.

[Workflow evidence](evidence/v4-native-lineage-capacity-workflows.json) contains
full returned payloads, adapter audits, compiler evidence, exact input hashes,
revision pins, drivers, exit status and independently observed process RSS.
Drivers use local fixture paths captured for reproducibility; no credentials,
remote jobs or private host/process inventories are included.

## Required verification status

The local default tier stopped after 840.39 seconds and the runtime tier after
15.00 seconds: the supervisor terminated their owned process groups when free
disk fell below the 512 MiB headroom guardrail (approximately 307/304 MB free).
Both are incomplete, with exit code -15, not passing checks. The configured wall
budgets were 1,500/1,200 seconds. Logs and receipts remain in the local
`/private/tmp/fq-v4-capacity-required-tiers-20261123` directory; only terminated
test scratch was removed after confirming the recorded processes had exited.
The production implementation is the revision named above; documentation and a
stronger native-overflow fixture were added while the broad tiers ran. These
attempts do not certify the later documentation/test head.

An initial metadata check found that the SDKBench pin lacked an explicit
external-dependency origin. After adding that sibling metadata field without
changing numerical results, the evidence checker reports the same 100 historical
errors as the earlier baseline and zero errors naming either new capacity file;
its citation check reports zero errors. Those historical errors are not waived.
The strengthened overflow fixture keeps all other budgets valid: 8,192 canceled
CZ inputs plus one unchanged RX tail, one output and 8,192 rewrite memberships.
All 11 exact-boundary tests passed again (2.24 seconds). After the provenance
correction, the combined boundary, documentation, privacy and capability check
passed all 60 tests (17.12 seconds).

Exact-head GitHub CI and completed required tiers remain necessary before merge.
The source-capacity dependency and this separate contract change must each meet
the normal merge gate. Release qualification and prospective held-out neutral
comparison remain pending.
