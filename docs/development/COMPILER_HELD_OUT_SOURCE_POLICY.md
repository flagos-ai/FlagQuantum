# Prospective external compiler source holdout

Status: population policy prepared before a draw or campaign circuit-body retrieval.
This document is not an executed qualification, a novelty certificate, an SDK
ranking, or evidence of improved performance. Execution classification is
`single_device_fast_path`; no distributed workload is involved.

## Requirement and remaining gap

Validate whether compiler changes generalize to real programs that were not
used to develop them. The earlier external DNN/QAOA/Simon campaign is now
inspected development evidence, including its input refusals and failed artifact
workflows. New angles or widths of an inspected recipe do not close this gap.
Keep the independent SDKBench evaluators, resource enforcement and thresholds
unchanged. Freeze the participant builds before revealing the selected programs.

This is an evaluation of delivered CPU compilation workflows. It cannot
establish that a scientific application is useful, that every possible input
state is correct at widths eight/twelve, or that one SDK is globally superior.

## Population determined without circuit outcomes

The [complete metadata census](evidence/compiler-held-out-source-population.json)
pins [VeriQBench](https://github.com/Veri-Q/Benchmark) at
`9be108313844e30c5df5780989a68cac4d966cfe`. Its source README describes
combinational, reversible and variational application circuits. Only public
repository metadata, filenames, sizes, blob identities, README and LICENSE were
retrieved for this policy. No selected circuit body, parser outcome or
participant compilation was inspected when choosing this population. Git blob
API URLs explicitly identify file objects; their final path component is the
SHA-1 consumed by the unchanged preparer. Source commits remain separate
`revision` fields. These identities do not establish scientific usefulness or
successful source admission.

The complete tree contains 966 QASM blobs. The filename suffix
`_(?:n)?(6|8|12)\.qasm$` and the existing 1-MiB preparation bound identify 21
metadata candidates. Filename widths are provisional; the unchanged converter
must confirm the actual declared width. Retain every classification in the
census. Restrict this static W03 study to `combinational/` and `variational/`:
dynamic sources are outside its coherent-prefix task. Exclude QFT and QAOA
families already inspected during development. These metadata rules leave eleven
eligible paths: two at width six, three at eight and six at twelve. The eligible
paths cover BV, phase estimation, reversible circuits, EfficientSU2 and QCNN;
algorithm labels do not imply a claim that their mathematics is unfamiliar.

These are application-motivated benchmark circuits, not evidence of an
industrial workload distribution. This corpus can test source-level
generalization only within its declared scope. The broader compiler objective
also requires independently selected, held-out research/application exports
with their scientific task data and generator provenance; favorable results on
these small benchmark programs alone cannot discharge that requirement.

The prior pinned QASMBench population already exposed all seven filename-matched
programs at these widths. It supplies no remaining source holdout under that
same eligibility rule. The alternative source above was chosen from metadata,
not after looking for favorable compiler results. Widths outside six/eight/twelve,
large sources and dynamic tasks remain explicit scope gaps. Do not pad sources,
repair syntax or generate replacement programs to increase successful coverage.

## Publication, build freeze and single draw

First publish this policy and census, then publish an immutable execution
manifest. That manifest must bind the policy/census hashes, unchanged SDKBench
revision `5ba03de33f8a9193f86b26d2cb3fa841ede7c98a`, every preparer/driver hash,
participant wheels and installed dependency inventories, configurations, CPU
profile, host admission, reservation end, historical audit inventory, required
negative controls, deterministic order rotation and complete phase budgets.
The FlagQuantum build must come from integrated main after its required exact-head
checks. Candidate capacity changes under review are not an integrated baseline.
Builds and configurations remain frozen through source exposure and execution.

Reuse SDKBench's existing entropy convention: call `secrets.token_bytes(32)` once
only after that complete public freeze. Immediately retain and publish the nonce,
freeze hash and draw record before retrieving any selected source. There is no
retry, replacement nonce, manual seed, result-driven subset or best draw.
This operator-generated draw is auditable from its record but is not proof of
independently witnessed entropy; disclose that limit.

For each width independently, rank every metadata-eligible row by the following
SHA-256 bytes, breaking ties by original path, and take the first two. This
produces six sources, not six independent algorithm families. Cross-width family
correlations remain visible in reporting. The expression specifies the ranking;
it does not draw the campaign nonce:

```python
import hashlib

def source_rank(path: str, nonce: bytes) -> tuple[bytes, str]:
    if len(nonce) != 32:
        raise ValueError("The frozen draw requires exactly 32 nonce bytes")
    domain = b"fq-compiler-external-holdout-20261123"
    payload = domain + b"\x00" + nonce + b"\x00" + path.encode("utf-8")
    return hashlib.sha256(payload).digest(), path
```

A source refusal, exposure match, missing notice or timeout never triggers another
draw or fills a vacancy. Preserve all six drawn sources in the intended census.
A subsequently repaired compiler or adapter can use these programs for
development, but cannot call a later run on them an unexposed confirmation.

## Source provenance, compatibility and exposure gates

Fetch only the drawn immutable blobs, verify Git blob/size and SHA-256, and retain
the original path and exact bytes. Stage deterministic aliases
`source-00_n6.qasm`, etc., in draw order for the unchanged preparer's `_nN` filename
contract. An alias changes no source byte, include, register, gate, wire or
parameter. Record a bijection from every alias to its upstream path and blob;
the reported original source remains the upstream program. The existing
converter must verify source width and source hash after parsing. The census is
not a preparation request: derive the unchanged preparer's existing `population.json`
from exactly the six drawn rows, their aliases and source hashes, the fixed
`["line", "2-by-n/2-grid"]` targets, and the staged LICENSE/NOTICE hashes.
Do not pass excluded or undrawn census rows to source preparation.

Retain the original MIT LICENSE byte-for-byte and all source notices. This pinned
tree has no upstream NOTICE. The existing preparer requires a NOTICE file, so
stage a clearly labeled supplemental provenance NOTICE identifying VeriQBench,
the original repository/path/revision and the alias mapping, explicitly stating
that the file is generated packaging information rather than an upstream notice.
Hash it before preparation. Inspect selected source headers for additional
attribution/license obligations before redistribution; unresolved rights block
that source, rather than being silently covered by the repository license.

Before compiler execution, audit every selected original source and converted
ordered operation list against the frozen materialized evidence inventory and
relevant obtainable Git history for FlagQuantum and SDKBench. Retain raw hashes,
canonical hashes, exact matches, inaccessible records, unsupported representations
and review of declared developer exposure. A match or declared prior inspection
marks that row exposed. A coverage gap marks novelty unestablished. Neither row
qualifies as a verified holdout or is replaced. Preparation itself reveals source
meaning; no compiler change may follow before the frozen evaluation completes.

The previous audit's 242 gaps are not waived. A limited hash search cannot prove
absence from private campaigns, upstream compiler development or human knowledge.
State the exact audit boundary and remaining uncertainty. At most claim a
prospective source holdout relative to the covered inventory and declared
exposure; do not turn external origin or absence of a match into a universal
novelty certificate. License/source admission and exposure are separate verdicts
from SDK implementation failures and numerical acceptance.

## Unchanged execution and comparison

Use the existing `sdkbench.candidates.external_bundle` preparation and W03 cases
without editing the pinned checkout. Preparation retains the 300-second/16-GiB
source supervisor, 1-MiB source bound and 20,000 expanded-operation bound. Source
syntax, noncoherent prefixes, dynamic operations, unsupported definitions and
filename/actual-width mismatches are input refusals, never repaired fixtures.
The converter's independent source checks remain mandatory.

Each drawn source has line and two-row grid targets. Register the same four
workflows as the earlier external study: FlagQuantum ordinary restored layout,
FlagQuantum target artifact, Qiskit release restored layout and Cirq release
restored layout. Seal exact releases/options before the draw. Every SDK receives
the same converted operation list, target and initial layout, with no ancillas.
Restore logical output order and include that work in its declared timed region.
The FlagQuantum artifact workflow additionally validates provenance and emits a
serialized executable artifact. Compare that delivered workflow separately from
ordinary compilation; do not attribute its total cost to routing alone.

Keep W03 semantic absolute tolerance `1e-8`, relative tolerance zero,
normalization error `1e-10`, and zero illegal target edges. Width six checks the
complete operator up to one common global phase. Width eight/twelve retain the
existing three output probes, which cannot prove arbitrary-input equivalence.
Preserve all original probe preparations and negative controls. Every accepted
sample must meet the same independent criteria; no accepted-only coverage score.

Retain 120 seconds per compile and the width-specific 600/900/1,800-second case
limits, with 16 GiB aggregate cgroup memory, no swap, one physical core/thread
and the original external-network denial. Use the same exclusive qualified Linux
host, immutable environments, SMT/core checks and ten fresh process blocks,
two warmups and three measured positions. Intended coverage is 48 workflow rows,
480 process blocks, 960 warmups and 1,440 measured positions before refusals.
Source failures and incomplete execution reduce observed coverage, never this
intended denominator. Record requested seeds and unsupported seed control.

Before launch, bind every preparation/campaign/replay/summary/cleanup budget and
a reservation sufficient for their sum. Campaign-wide exhaustion stays an
incomplete campaign and produces no replacement selection. Reuse the existing
supervisor and phase helper; this policy does not authorize a new host booking,
runner-label change, reboot, kernel/driver update or service interruption.
A renewed host admission and adequate remaining reservation are prerequisites,
not implied by an old successful campaign.

## Outputs and admission decision

Publish all source/refusal/exposure decisions, original and converted inputs,
notices, hashes, installed identities, exported programs, final permutations,
per-sample semantics, CZ counts, two-body dependency depth, time and process peak
memory, errors, unsupported cases, resource failures and operational recovery.
Keep raw attempts; operational recovery does not erase an earlier attempt.
Detailed internal host/process data require their own disclosure authorization.

Report each workflow, width and original program, including regressions and
quality/time/memory tradeoffs. Report eligible, drawn, source-admitted,
holdout-established, executed and independently accepted denominators separately.
Do not fabricate an optimality gap or scalar winner. Match each proposed compiler
improvement to its prospective baseline and confidence/measurement limits.
A favorable subset, warm result, probe-only check or passing source CI cannot
certify the full goal. No support level changes in this policy.

The current user workflow supported by this validation plan remains:

```bash
python -m examples.target_aware_compilation
```

Expected: legal target edges, restored logical mapping and matching CPU
statevector execution. This policy adds no public API or production compiler
behavior. Source retrieval, draw, exposure audit, build/environment freeze,
qualified execution and independent result replay remain outstanding.

## Policy verification

The first local check had 72 passes and one evidence-origin failure because
bare 40-digit file-blob identifiers were interpreted as commit revisions. The
census now uses explicit upstream Git blob API URLs, preserving every object
identifier without misdeclaring a file as an external commit. No validator or
origin rule changed. After correction, all 73 selected revision-history,
field-vocabulary, file-budget, Markdown-link and public-evidence-privacy tests
passed in 105.75 seconds in an isolated checkout based on integrated main.
The existing target-aware compilation example passed with legal edges, restored
mapping and local statevector execution. The documented ranking was exercised
with three fixed control nonces, retaining two unique paths at every width and
rejecting malformed nonce length; these controls are not the campaign draw.
Every census path, size and blob reference matches the immutable tree, and the
LICENSE blob identity was independently recomputed. Three changed text files
pass the repository language check. No full local default/runtime tier or
new-head CI pass is claimed; those gates remain required before integration.
