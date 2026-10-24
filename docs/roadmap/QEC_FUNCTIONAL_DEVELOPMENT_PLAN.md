# QEC functional development plan

Owner: QEC maintainers. Date: 2026-10-24 (Asia/Shanghai).
Status: sequencing approved by the user; delivery in progress. Approval of this
roadmap does not replace review and approval of protected API change proposals.

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
and was explicitly approved for implementation on 2026-10-24. Cross-decoder Decision B and experiment Decision C
remain unapproved. The separately approved Tensor-only PyMatching batch method
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

| ID | State | Evidence |
| --- | --- | --- |
| 001 | Merged in [#774](https://github.com/flagos-ai/FlagQuantum/pull/774) | [Post-retirement mainline acceptance](../development/QEC_RETIREMENT_MAINLINE_ACCEPTANCE.md): main and PR full CI/GPU checks passed; old clean merged QEC worktrees and branches were cleaned. |
| 002 | Proposal merged in [#776](https://github.com/flagos-ai/FlagQuantum/pull/776) | Decision A explicitly approved by the API owner on 2026-10-24; B/C remain unapproved. |
| 003 | Merged in [#779](https://github.com/flagos-ai/FlagQuantum/pull/779) | [Implementation and resource evidence](../development/QEC_FUNCTIONAL_003.md); final PR head passed complete CI and single-/two-GPU checks. [Post-merge acceptance](../development/QEC_FUNCTIONAL_003_MAINLINE_ACCEPTANCE.md) records exact revisions, installations and checks. |
| 004 | Merged in [#789](https://github.com/flagos-ai/FlagQuantum/pull/789) after explicit merge authorization | [Proposal 088](../development/API_CHANGE_PROPOSAL_088_QEC_CORRELATED_PAULI_FAULTS.md) specifies the channel grammar, exclusive distributions, compatibility and independent numerical acceptance. [Implementation](../development/QEC_FUNCTIONAL_004.md) covers exclusive circuit-noise construction and sampling; [post-merge acceptance](../development/QEC_FUNCTIONAL_004_MAINLINE_ACCEPTANCE.md) records exact main, fresh artifacts and checks. |
| 005 | Concrete contract awaiting explicit owner approval | [Batch delivery contract](../development/QEC_FUNCTIONAL_005_BATCH_CONTRACT.md) preserves complete single-shot results, specifies the supported concrete types and differing duplicate semantics, and refuses unknown/stateful plugins. No implementation is authorized yet. |
| 006 | Existing logical-only native batch path; broader optimization pending | The approved PyMatching Tensor operation already exists. Do not reimplement it or use it to invent full correction records; any further optimization must preserve the approved 005 result contract. |
| 007 | Example statistics delivered; public experiment contract pending | #784 provides compiled sampling; #788 provides fixed-shot counts and Wilson intervals in the example. These do not activate Decision C or adaptive stopping. |
| 008–009 | Pending respective contracts | No threshold, broad experiment API or performance claim follows from the partial deliveries above. |
| 010–012 | Candidate tranche | Separate contract review required. |
