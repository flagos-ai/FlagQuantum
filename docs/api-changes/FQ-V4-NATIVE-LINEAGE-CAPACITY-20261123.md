# Bound expanded native lineage independently in v4 compilation evidence

Status: approved explicitly by the user on 2026-11-23; implementation and release qualification pending.
Owner: Integration and Core; Compiler is the producing consumer.
Classification: single_device_fast_path; serialization and validation only.

## Problem and verified workflow

An application compiles a source circuit artifact for a target using the existing
`compile_circuit_artifact_for_target` workflow. Native decomposition expands
source operations before CZ block resynthesis reduces the executable output.
The approved v2 circuit payload fix admits the source but cannot admit its v4
lineage. No validator was disabled to obtain these observations.

At FlagQuantum revision `64c381a467685dc4e198411195ae0bea9df4d9b0`, unchanged
SDKBench revision `5ba03de33f8a9193f86b26d2cb3fa841ede7c98a` reports:

| Inspected DNN input | Source operations | Native inputs | Outputs | Block memberships | Maximum topology index |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8-wire grid | 2,640 | 5,544 | 2,336 | 4,904 | 2,663 |
| 8-wire line | 2,640 | 5,586 | 2,320 | 4,880 | 2,665 |

Both stop with `native_input_instructions exceeds record limit 4096` in
`PhysicalPlanEvidenceV4.__post_init__`. Their membership totals would independently
exceed the existing 4,096 cap. The diagnostic records sizes before invoking the
original constructor validator unchanged. Both have 64 rewrite blocks; maximum
source index is 2,639. Complete outcomes and diagnostic are checked into PR #1326
under `docs/development/evidence/program-artifact-delivered-workflows.json` and
`program-artifact-v4-record-counts.json`. These are inspected development inputs,
not held-out evidence or comparative resource measurements.

## Proposed contract delta

Change exactly two independent v4 budgets, in both constructor and reader:

| Scope | Existing | Proposed |
| --- | ---: | ---: |
| Native input instruction records | 4,096 | 8,192 |
| Aggregate block input plus output memberships | 4,096 | 8,192 |

Use explicit private limits at their owning validation sites. Do not change the
shared `_MAX_RECORDS` default. The power-of-two ceiling bounds expansion above
the observed requirement; it does not promise admission of arbitrary programs.

Retain 4,096 limits for physical output records, mapping transitions, rewrite
block records, source/topology ancestry indices, layouts, critical-path entries,
and predecessor lists. Retain per-block limits of 64 inputs and 96 outputs,
16 MiB plan and bundle caps, all target/native-growth budgets, wire legality,
ordered ancestry, disjoint complete membership, schedule validation, residual
thresholds and rejection of unsupported cases. v1-v3 remain unchanged. Do not
add a public configuration option, new fields, or a new version.

## Compatibility and migration

Existing accepted v4 objects retain canonical bytes, identity payloads, hashes,
fields, signatures and exception classes. Existing rejection behavior is retained
outside the two expanded budgets. Older readers reject newly admitted large v4
objects, so receiving environments must upgrade before producers send them.
Document this in the next minor-release compatibility notes; installations
requiring older receivers must retain the earlier size constraints. No existing
accepted document needs conversion or deprecation. Explicit API-owner approval
is required for in-place expansion under version 4.0; if version negotiation is
required instead, reject this proposal and design that separately.

Update the v4 candidate contract with a narrow, dated approval record while
preserving its original approval. No API snapshot regeneration to hide impact.

## Alternatives

Keeping the cap preserves the current fail-closed DNN failure, which remains
an honest supported choice. Omitting native records, truncating memberships,
bypassing evidence, falling back to older evidence, or disabling CZ optimization
to conceal capacity failures would alter the verified workflow and are rejected.
A global record-limit increase expands unrelated schemas and is unnecessary.
A new evidence version provides explicit negotiation but duplicates an otherwise
identical schema; it is a separate decision if owners require that migration.

## Acceptance and resource bounds

Before merge, prove exact acceptance at 8,192 and rejection at 8,193 for each
expanded budget with semantically valid plans; test constructor, dictionary and
JSON paths. Reject oversized raw lists before per-record reconstruction. Prove
all unchanged 4,096 limits and v1-v3 golden identities remain unchanged. Test
negative ancestry, duplicate/out-of-range membership, omissions, schedule and
layout tampering, hash tampering and byte limits. A membership-boundary test
must retain individual block-size limits rather than weakening them.

Use fresh workers with explicit 20-second wall limits for old-limit, new-limit,
over-limit, and complete real-program workflows. Observe peak RSS independently
of traced Python allocations. Establish acceptance guardrails from measurements
before proposing release; do not claim hard memory enforcement on macOS. Preserve
failures and timing/memory regressions. Any formal comparative claim requires
the unchanged SDKBench protocol and qualified host/resource enforcement.

Rerun all six inspected external artifact workflows without monkeypatching or
changing adapters/thresholds. DNN success is required for the specific capacity
claim; failures elsewhere remain reported. Add generated workloads near the
limits so implementation does not dispatch on benchmark identity. Run protected
API, v4/legacy evidence and compiler workflow tests, then default/runtime tiers
and the full exact-head CI gate. No merge while required verification is pending.

## Approval scope

Approval authorizes only the two v4 limits and their necessary reader/writer,
contract, tests, and release documentation updates. It does not authorize
changing any other limit, algorithm, SDKBench policy, public signature, source
artifact budget, or making a general performance/superiority claim.

## Recorded authorization

On 2026-11-23 the user explicitly approved both v4 limit changes described above.
The original 2026-11-18 v4 contract approval remains intact. This approval does
not certify performance, release readiness, or any additional contract change.

## Release compatibility note

The next minor release may emit v4 evidence with up to 8,192 native inputs and
8,192 aggregate rewrite memberships. Upgrade all evidence readers before using
these larger artifacts. Older readers continue to reject them. Artifacts within
the previous limits retain identical canonical representations and identities.
The release version remains unassigned; approval does not authorize a release.
