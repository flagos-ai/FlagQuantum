# Separate the circuit payload and envelope entry budgets

Status: approved by the API owner on 2026-11-23; implementation in review; release qualification pending.
Owner: Integration and Core, with Compiler, Runtime, and Deployment as consumers.

## User journey

An application can build and directly compile a circuit of hundreds of gates,
yet fail to serialize that same source for artifact-based compilation. Issue
[FlagQuantum #1322](https://github.com/flagos-ai/FlagQuantum/issues/1322) isolates
the failure at ProgramArtifactV2.from_circuit_ir, before target selection or
routing. The SDKBench campaign recorded 5,637 payload entries for a 702-operation
QAOA input and 21,143 for a 2,640-operation DNN input. Both fit within the existing
16 MiB payload byte limit.

The existing reproducer uses the artifact constructor that failed in the
measured workflow:

    import flagquantum as fq
    from flagquantum.core._artifacts import ProgramArtifactV2

    circuit = fq.Circuit(2)
    for _ in range(600):
        circuit.h(0)
    artifact = ProgramArtifactV2.from_circuit_ir(
        circuit.to_ir(), producer="capacity-example"
    )
    assert len(artifact.payload["instructions"]) == 600

At the proposal baseline, the constructor raises ValueError for the entry budget. The approved
reader and writer would support this round trip. Target compilation remains
subject to independent target and provenance limits.

## Contract delta

For version 2.0 and profile circuit-ir-1.0 only, replace the shared aggregate
entry counter with two independent budgets.

| Counted scope | Current maximum | Proposed maximum |
| --- | ---: | ---: |
| Canonical CircuitIR payload mapping and list entries | 4,096 | 65,536 |
| Structured envelope entries, excluding payload descendants and including its field | 4,096 shared with payload | 4,096 |
| Aggregate upper bound for the circuit profile | 4,096 | 69,632 |
| Executable v2 and structured v3 entry budgets | 4,096 | 4,096 |

The payload limit is a power-of-two ceiling with headroom above the measured
2,640-operation circuit. It is an entry limit, so it does not promise a fixed
number of operations or admission of every circuit permitted by a target. Retain
the depth limit of eight, the 16 MiB payload cap, the 18 MiB envelope cap, the
structured string limit of 4,096 bytes, the closed value algebra, canonical IR
validation, sensitive structured field rejection, and identity verification.

Core should pass the explicit entry budget down the existing recursive validator
from the circuit-profile payload call site. It should enforce that budget before
normalization or reconstruction of the circuit and avoid the second recursive
payload walk, which today exists to enforce the combined count. The structured
walk continues counting one entry for the payload field. Validate byte caps
before IR reconstruction where practical and bound cumulative encoding work
before admitting oversized input. Keep the existing 4,096 default at all other
validator call sites. The approved v2 contract record needs an additive,
profile-specific revision after approval. Preserve the earlier approval record.

## Compatibility and rollout

Keep version, profile, fields, canonical bytes, hash inputs, values and exception
classes. All previously accepted envelopes remain readable and retain identical
digests and artifact identities. A reader with the earlier limit rejects an
expanded artifact with its existing ValueError. Upgrade receiving environments
before sending expanded envelopes. Version 1 reading, v2 executable admission,
and v3 and v4 validation stay on their existing limits. This expands admission
under an existing version and requires an explicit compatibility decision.

The next minor release should document reader-first rollout and provide the
current and proposed limits in its compatibility notes. Existing documents need
no migration or deprecation period. Producers targeting older readers must
continue to stay within the earlier budget. No writer override or new public
configuration is proposed.

An explicit new version would let senders advertise the new resource contract,
at the cost of another version of structurally identical payloads and changes to
consumers that dispatch by version. If API owners require version negotiation,
reject the in-place proposal and define that evolution separately. A global
constant increase also broadens v2 executable and v3 metadata admission.
Byte-only admission removes an independent bound on traversal and allocation.
Neither alternative is recommended.

## Acceptance before implementation can merge

- Assert exact acceptance at 65,536 payload entries and rejection at 65,537.
  Exercise cumulative counts across lists and mappings, both structured and
  payload depth, and independent payload and envelope byte limits.
- Exercise an over-limit input with an instrumented reader and establish that
  validation rejects it before CircuitIR reconstruction. Include short-entry
  payloads, very wide containers, deep structures and long scalar strings.
- Round-trip the 600-H example through dictionary and JSON readers. Preserve
  existing golden artifact bytes, payload digests, and identities and verify
  tampered hashes and unexpected fields are still rejected.
- Show the same bounds for executable v2 and v3 structured data. Verify v1 and
  v4 reading and existing v4 provenance refusal with unchanged fixtures.
- Measure construction, read time and peak process RSS in fresh processes on
  old-limit, proposed-limit, and rejected payload cases. Establish budgets from
  those measurements before release. Include the production serializer and
  receiver together, along with real workloads and explicit failure outcomes.
- Run the protected API contract, affected Core and compiler end-to-end tests,
  resource boundary tests, and default and runtime tiers. Run the unchanged
  SDKBench acceptance policy for any promoted comparison.

Approval covers only the profile-specific entry budget and its reader/writer
validation. Downstream provenance limits, target budgets, public signatures,
and SDKBench acceptance remain outside this proposal.

## Implementation and development evidence

The implementation is classified as `single_device_fast_path`: local artifact
validation and serialization, without distributed execution. The validator takes
an explicit private entry budget only at the circuit payload call. Other callers
keep the 4,096 default. The envelope counts the payload field once and no longer
walks payload descendants a second time. Payload byte encoding is incremental
and stops at 16 MiB before IR reconstruction. Source string/key lengths bound a
single escaped JSON token; these checks cannot reject a previously accepted
payload because JSON cannot encode a string shorter than its source length.
Canonical encoder settings and identity inputs remain unchanged. The writer
also bounds canonical encoding before hashing, and the receiver reuses the
verified canonical payload digest as the identical CircuitIR content digest.
The preliminary implementation and observations remain in the evidence file;
the final rerun follows this encoding-order change, not a favorable-sample rule.

The [raw resource observations](../development/evidence/program-artifact-circuit-capacity.json)
retain 66 final fresh-process runs and all 66 preliminary runs, the complete driver, baseline/current source
hashes, source revision, fixture hashes, outcomes, construction/serialization/read
times and peak process RSS. Each of eleven cases ran three times for both the
previous implementation and this candidate. Cases include 4,000, 65,536 and
65,537 payload entries, 600 H gates, a 16 Mi-character escaped string, and six
previously inspected external program fixtures. Receiver and serializer run
together in each fresh worker. The baseline module is loaded from the recorded
main revision; there is no change to the installed repository or SDKBench policy.

The current implementation accepts 65,536 entries and rejects 65,537. It accepts
and round-trips all six inspected external inputs, including 2,640-operation
DNN inputs with 30,935 entries and 702-operation QAOA inputs with 8,231 entries
under the driver's ordinary tensor-angle construction. Entry counts depend on
the concrete serialized IR, not only operation count. The previous reader rejects
these four DNN/QAOA inputs and the 600-H case. Both versions accept the smaller
Simon inputs. Previously accepted cases retain exactly the same identities.

The candidate's largest observed construction/serialization/read total is
0.282 seconds; its largest observed process RSS is below 419 MiB, including
imports and input construction. The maximum escaped-string case is rejected,
not reconstructed. These shared macOS observations establish development
triage guardrails of five seconds worker wall time and 512 MiB peak RSS for
these cases, selected after measurement. All candidate samples satisfy them.
The subprocess has an enforced 20-second wall timeout; RSS is observed, not an
OS-enforced memory cap. These bounds do not qualify arbitrary payloads,
other operating systems, downstream compiler evidence, a release, a participant
ranking, or scalability. There was no qualified host reservation for this work.

Focused validation covers exact entry and byte boundaries, cumulative and wide
containers, excessive depth, escaped strings, rejection before IR reconstruction,
independent executable-v2/v3 envelope limits, unchanged structured string/depth
limits, golden v1/v2 fixtures, identity tampering and v4 provenance reading.
An initial new envelope-byte test used dataclass replacement on a normalized
executable whose existing wire validator expects JSON lists; it was corrected
to exercise the supported dictionary reader. No production wire validation
contract was changed to accommodate the test.

Final-source focused artifact, public API, compiler pipeline, legacy identity
and v4 evidence checks passed: 976 tests in 28.80 seconds. The public API
baseline, source provenance and documentation links also pass. Required
default/runtime tiers are running with bounded budgets; their pending status
is not a passing verification claim.

Before release, rerun default/runtime tiers and the protected API checks on the
merged revision, repeat resource checks on supported receiver environments, and
upgrade readers before distributing expanded artifacts. Downstream provenance
limits and formal SDKBench acceptance remain independent release blockers.


## Delivered target-workflow regression check

The [complete outcomes](../development/evidence/program-artifact-delivered-workflows.json)
run the unchanged SDKBench target-artifact adapter and routing acceptance on six
previously inspected inputs at candidate revision
`64c381a467685dc4e198411195ae0bea9df4d9b0`. SDKBench is pinned to
`5ba03de33f8a9193f86b26d2cb3fa841ede7c98a`. Four workflows pass: QAOA grid and
line have 66/90 CZ gates and dependency depths 40/58; Simon grid and line have
36/34 CZ gates and depths 33/31. These are development regression checks, not
held-out comparisons, qualified resource evidence, or superiority claims.

Both DNN inputs pass source-artifact admission and then fail closed in the v4
physical-plan constructor: `native_input_instructions exceeds record limit
4096`. A [read-only diagnostic](../development/evidence/program-artifact-v4-record-counts.json)
observes array lengths and calls the original unchanged validator. It does not
raise any limit or bypass validation.

| DNN input | Native inputs | Physical outputs | Transitions | Rewrite blocks | Nested input/output memberships |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grid, 8 wires | 5,544 | 2,336 | 24 | 64 | 4,904 |
| Line, 8 wires | 5,586 | 2,320 | 26 | 64 | 4,880 |

Native ancestry topology indices peak at 2,663/2,665 and source indices at
2,639, within their existing 4,096-entry address space. Both native-input and
aggregate block-membership counts exceed separate v4 limits. This is a distinct
serialization contract, outside the approved change. The capacity fix therefore
does not establish DNN target-artifact support. Changing the downstream contract
requires its own proposal and explicit authorization. All six source artifacts
remain subject to the existing target, native growth and evidence byte budgets.

The check ran locally on macOS with one numerical thread requested, existing
per-stage timeouts, and no enforced memory cgroup or exclusive host reservation.
The evidence retains failed traces and the complete diagnostic driver. No SDKBench
acceptance threshold or adapter was changed.

After synchronizing the main CI temporary-directory isolation fix, the focused
v4 evidence and CI-isolation tests pass: 116 tests in 3.92 seconds. This does not
replace the pending exact-head default and runtime qualification.
