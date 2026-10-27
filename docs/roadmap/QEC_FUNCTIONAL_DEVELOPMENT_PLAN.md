# QEC functional development plan

Owner: QEC maintainers. Date: 2026-10-24 (Asia/Shanghai).
Status: sequencing approved by the user; delivery in progress. Approval of this
roadmap does not replace review and approval of protected API change proposals.

## Relation to the FTQC mission

This is the QEC foundation subplan of the [FTQC simulation master plan](FTQC_SIMULATION_MASTER_PLAN.md). The master plan maps the full mission to R01–R15, FTS work packages and staged acceptance, including mixed simulation, dynamic feedback, domestic hardware and experimental validation. Existing QEC contract approvals and compatibility gates remain authoritative; the broader roadmap does not approve pending interfaces. Every PR must include runnable public-user code, prerequisites, an execution command and expected results, including documentation, tests and performance changes.

## Starting point

[PR #749](https://github.com/flagos-ai/FlagQuantum/pull/749) delivered the
responsibility-based package restructuring. [PR #766](https://github.com/flagos-ai/FlagQuantum/pull/766)
removed 23 historical modules for 0.4.0 and retained one registered
`RotatedSurfaceCode`. Functional expansion is a new phase, not unfinished
structural migration. See the [layout and migration guide](../guides/QEC_PACKAGE_LAYOUT.md).

The baseline already includes code families, memory declarations, DEM
construction and sampling, matching/BP/BPOSD decoders, streaming windows and an
optional PyMatching adapter. These are bounded research/reference capabilities;
their presence does not establish scalable construction, threshold estimates,
hardware feedback or production performance. Use implementation and tests to
resolve stale historical roadmap statements.

## Delivery sequence

Each row is a separately reviewable PR directly targeting `main`. Start an
implementation PR after its required contracts and dependencies have merged.
Do not use temporary integration bases. Numbers below are plan identifiers,
not GitHub issue or PR numbers. Split a row if its reviewed scope requires it.

| ID | Proposed title | Dependency | Deliverable and acceptance |
| --- | --- | --- | --- |
| QEC-FUNCTIONAL-001 | `docs(qec): record mainline acceptance and development roadmap` | #766 | Verify merged-main CI, source/install/examples and immutable fixtures; record exact evidence, then clean only merged, clean temporary workspaces and branches. |
| QEC-FUNCTIONAL-002 | `docs(qec): propose scalable experiment and decoder contracts` | 001 | Propose construction, batch decoding and experiment interfaces, failures, compatibility and ownership; obtain explicit approval before protected API implementation. |
| QEC-FUNCTIONAL-003 | `feat(qec): derive Clifford fault signatures by stabilizer propagation` | 002 approval | Remove the full-statevector requirement from the supported Clifford fault-signature path. Compare every small-model mechanism with the existing execution reference, and measure resource growth at larger sizes. |
| QEC-FUNCTIONAL-004 | `feat(qec): model correlated Pauli circuit faults` | 003 | Support two-qubit Pauli faults and explicit exclusive distributions, including depolarization. Verify probabilities, detector/observable flips and sampling; never silently replace exclusivity with independent faults. |
| QEC-FUNCTIONAL-005 | `feat(qec): add batch decoding across supported decoders` | 002 approval | Implement the approved batch contract for supported algorithms; match per-shot decoding and validate empty batches, invalid inputs and algorithm-specific limits. |
| QEC-FUNCTIONAL-006 | `perf(qec): accelerate graphlike batch decoding with PyMatching` | 005 | Use optional native batch support; independently cross-check results, report throughput and memory with the accepted translation domain and tie/tolerance rules. |
| QEC-FUNCTIONAL-007 | `feat(qec): add reproducible logical error rate experiments` | 004, 005 | Compose sampling, decoding and logical statistics, with confidence intervals, stopping rules, reproducible seeds and captured experiment configuration. |
| QEC-FUNCTIONAL-008 | `feat(qec): add code distance sweeps and threshold analysis` | 007 | Sweep distance and noise, export reusable results and validate fits/uncertainty against controlled data. Refuse threshold conclusions when the evidence is insufficient. |
| QEC-FUNCTIONAL-009 | `test(qec): benchmark decoding accuracy and resource scaling` | 006, 008 | Establish accuracy, refusal/failure rate, throughput and memory evidence. Label declared-model sampling separately from executed-circuit sampling; preserve both. |
| QEC-FUNCTIONAL-010 | `docs(qec): propose subsystem and Floquet execution contracts` | 009 | Separately specify time-dependent measurements, gauge semantics, detectors and logical tracking; review boundaries before implementation. |
| QEC-FUNCTIONAL-011 | `feat(qec): execute a subsystem memory experiment` | 010 approval | Complete one small subsystem-code schedule-to-DEM-to-sampling-to-decoding workflow, independently checked. |
| QEC-FUNCTIONAL-012 | `feat(qec): execute a Floquet memory experiment` | 010 approval | Complete one Floquet memory workflow with inter-cycle detectors and logical tracking. Distinguish instantaneous code distance from spacetime fault distance. |

001–002 were delivered in #774 and #776. Decision A was explicitly approved on
2026-10-24; 003 merged in #779 and 004 merged in #789 after its approval.
The exact channel extension for 004 is
specified in [Proposal 088](../development/API_CHANGE_PROPOSAL_088_QEC_CORRELATED_PAULI_FAULTS.md)
and was explicitly approved for implementation on 2026-10-24. The complete cross-decoder Decision B contract was approved on 2026-10-26.
Decision C remains design-only, with implementation explicitly deferred. The separately approved Tensor-only PyMatching batch method
on main is recorded in [its own proposal](../development/API_CHANGE_PROPOSAL_087_QEC_BATCH_DECODING.md);
it does not authorize the broader cross-decoder contract. 004–009 depend on
their respective approved contract scope.
010–012 are later candidates; their contract review may require additional PRs.
This is a delivery sequence, not a promise of exactly twelve PRs or fixed dates.

## Cross-cutting acceptance

- Titles follow `type(scope): description`; branches do not use `codex/`.
- Every PR includes executable public-user code, prerequisites and expected
  output. Document-only PRs may use an existing supported example.
- Preserve the existing ordered 108 root exports, signatures/defaults, numerical
  behavior, single registered rotated class, surviving pickle identities and
  registration policy. Any intentional protected API change needs its own
  approved proposal; extending functionality is not permission to rewrite a snapshot.
- Do not change or regenerate the original 15 restructuring fixtures. New
  evidence must be separately named and traceable to its producer and version.
- Required full CI must pass at the reviewed head. Do not reduce thresholds,
  suppress failures or introduce skips to obtain a passing result. Report
  pre-existing conditional skips separately from required checks.
- Make numerical claims with independent reference evidence. Benchmark claims
  state model, hardware, size, precision, seeds and resource measurements.
- Keep QEC semantics in QEC, compilation in Compiler, orchestration in Runtime,
  numerical execution in Simulation and hardware adapters in Remote/Compute.
- Do not modify or merge DynamicCircuit PR #731. Do not merge PRs, create release
  tags or publish packages without explicit user authorization.

## Example retained throughout the first tranche

From the repository root with an installed package:

```bash
python examples/qec/installed_imports.py
```

Expected output:

```text
Public exports: 110
Registered class: True
Code pickle path: flagquantum.qec.codes
Correction: (0,)
Logical error rate: 0.0
```

Source checks use `PYTHONPATH=.`. Installation acceptance runs outside the checkout
with `PYTHONPATH` unset and verifies the imported `site-packages` location.

## Deferred scope

GPU decoding, asynchronous execution, device-verified realtime feedback,
lattice surgery, code deformation and magic-state protocols need separate
proposals after the relevant foundation is measured. General non-Pauli noise
must not be represented as an exact Pauli DEM without an explicitly approved
approximation policy. External plugin discovery and exclusive-group decoder
normalization remain candidate work, not capabilities implied by this plan.

## Progress

The separate continuing compatibility/performance deliveries are summarized in
the [Stim/PyMatching progress ledger](QEC_STIM_PYMATCHING_DELIVERY_PLAN.md#continuing-delivery-progress--2026-10-26).
As of 2026-10-26, #872, #877 and #885 are merged; #885's merge-specific Main smoke
and container publishing succeeded. The central progress correction #889 is
merged with complete head CI/GPU and exact-merge Main smoke success. Main smoke
is not full main CI. These private optimizations did not themselves approve 005 or Decision C. The owner separately approved
the complete 005 contract on 2026-10-26; C implementation remains deferred.

The [remaining-gap inventory and owner decisions](QEC_STIM_PYMATCHING_DELIVERY_PLAN.md#remaining-compatibility-and-functional-gaps)
separate delivered compatibility from unapproved functionality. Nested-repeat bridge-bound conformance is delivered in #896 with exact-main
Main smoke success. The separately approved 005 contract is delivered through
[#904](https://github.com/flagos-ai/FlagQuantum/pull/904), with full reviewed-head
CI/GPU and exact-merge Main smoke success. The central ledger records exact
revisions, local verification limits and the cancelled container publication.
The finite schedule/feedback Groups A-E matrix is already accepted through
[#837](https://github.com/flagos-ai/FlagQuantum/pull/837), as recorded in its
[approved conformance proposal](../development/QEC_SCHEDULE_FEEDBACK_CONFORMANCE_PROPOSAL.md).
Every future implementation/performance delivery must update the central ledger
and this entry point, preserving pending checks and scope boundaries.

| ID | State | Evidence |
| --- | --- | --- |
| 001 | Merged in [#774](https://github.com/flagos-ai/FlagQuantum/pull/774) | [Post-retirement mainline acceptance](../development/QEC_RETIREMENT_MAINLINE_ACCEPTANCE.md): main and PR full CI/GPU checks passed; old clean merged QEC worktrees and branches were cleaned. |
| 002 | Proposal merged in [#776](https://github.com/flagos-ai/FlagQuantum/pull/776) | Decision A explicitly approved by the API owner on 2026-10-24; Decision B complete contract approved 2026-10-26; C design-only. |
| 003 | Merged in [#779](https://github.com/flagos-ai/FlagQuantum/pull/779) | [Implementation and resource evidence](../development/QEC_FUNCTIONAL_003.md); final PR head passed complete CI and single-/two-GPU checks. [Post-merge acceptance](../development/QEC_FUNCTIONAL_003_MAINLINE_ACCEPTANCE.md) records exact revisions, installations and checks. |
| 004 | Merged in [#789](https://github.com/flagos-ai/FlagQuantum/pull/789) after explicit merge authorization | [Proposal 088](../development/API_CHANGE_PROPOSAL_088_QEC_CORRELATED_PAULI_FAULTS.md) specifies the channel grammar, exclusive distributions, compatibility and independent numerical acceptance. [Implementation](../development/QEC_FUNCTIONAL_004.md) covers exclusive circuit-noise construction and sampling; [post-merge acceptance](../development/QEC_FUNCTIONAL_004_MAINLINE_ACCEPTANCE.md) records exact main, fresh artifacts and checks. |
| 005 | Approved 2026-10-26; merged and accepted in [#904](https://github.com/flagos-ai/FlagQuantum/pull/904) | [Batch delivery contract](../development/QEC_FUNCTIONAL_005_BATCH_CONTRACT.md) preserves complete single-shot results, specifies the supported concrete types and differing duplicate semantics, and refuses unknown/stateful plugins. Actual five-type SDK and source/wheel/sdist installation profiles passed; full reviewed-head CI/GPU and exact-merge Main smoke succeeded. The central ledger preserves local default/build and container-publishing limits; no speed claim. |
| 006 | Existing logical-only native batch path; broader optimization pending | The approved PyMatching Tensor operation already exists. Do not reimplement it or use it to invent full correction records; any further optimization must preserve the approved 005 result contract. |
| 007 | Complete fixed-shot contract approved; implementation candidate under verification | [Decision C refinement](../development/QEC_FUNCTIONAL_007_EXPERIMENT_DESIGN.md) proposes explicit batch size, default-only decoder configuration and caller-owned evidence without persistence. #784/#788 sampling and fixed-shot counts/Wilson intervals do not activate this public API or adaptive stopping. Implementation remains deferred. |
| 008–009 | Pending respective contracts | No threshold, broad experiment API or performance claim follows from the partial deliveries above. |
| 010–012 | Candidate tranche | Separate contract review required. |

## Decision C contract preparation after design delivery

[#912](https://github.com/flagos-ai/FlagQuantum/pull/912) delivered the design
review as merge `b9b4f51963a4ab85e9253b3d070e798423082ee8`. Exact-head full CI
`40289425554` passed all 54 jobs; GPU `40289425526` attempt 3 passed three jobs
with three original optional skips. All 60 checks were 57 success / 3 original
skips. Its exact merge Main smoke `40300707460` succeeded; that is not full
main CI. Earlier GPU variance failures remain preserved; thresholds were not
changed. The owner specifically authorized cancelling one blocked Scheduled
hardware run to release the shared queue; no general cancellation permission
follows.

The owner then requested proceeding with the recommended three choices:
required explicit batch size, default-only registered decoder selection and
caller-owned evidence without initial persistence. The
[complete C/007 contract draft](../development/QEC_FUNCTIONAL_007_EXPERIMENT_CONTRACT.md)
now makes the signature, result, strict actual-law source path, support and
exception tables concrete for review. This is document preparation, not
approval of the complete contract or implementation. Older design-choice
wording above records the #912 publication snapshot. Decision B/005 remains
complete; C implementation, adaptive stopping, threshold fitting and persistence
remain deferred. No new experiment, performance study or hardware run was done.

## Approved C/007 implementation candidate

After the #924 contract artifact was delivered and its exact merge Main smoke
`40314835571` succeeded, the owner explicitly approved the complete
[C/007 contract](../development/QEC_FUNCTIONAL_007_EXPERIMENT_CONTRACT.md) for
implementation, verification and delivery on 2026-10-26 (Asia/Shanghai).
This supersedes earlier C design-only/deferred wording in historical progress
snapshots above. B/005 remains delivered. Adaptive stopping, threshold analysis,
nondefault decoder configuration and persistence remain separate decisions.

The candidate adds only `MemoryExperimentResult` and `run_memory_experiment` in
the canonical `qec.execution.experiments` module. It reuses original-source
export/strict actual-law DEM/one compiled sampler, five default registered exact
decoders through complete-record 005, exact final partial batches, all-logical
counts and one shared private Wilson formula. No root or parent re-export,
registry entry, dependency, channel law, RNG schedule or numerical tolerance is
changed. Zero-noise PyMatching isolated-detector refusal remains explicit.
Focused correctness and actual-SDK acceptance are in progress; exact-head full
CI/GPU, reviews/rules, normal pinned merge and exact merge/Main smoke acceptance
are still required. No performance, threshold, GPU or capacity claim is made.
