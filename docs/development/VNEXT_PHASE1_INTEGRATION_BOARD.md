# FlagQuantum vNext Phase 1 Integration Report

> Status: Core contract reconciliation, architecture ADR consolidation, and
> Compiler performance fixes reviewed, merged, and jointly verified.
>
> Integration branch: `codex/flagquantum-vnext-architecture`
>
> Phase 0 baseline: `99d5a92091bff35fdc573f4b401e3ebaaf5ffcbf`
>
> Phase 1 integration commit: `11c473cccbf9ef65b4a4743dd0921d582066c361`
>
> Integration completed: 2026-09-03 (Asia/Shanghai)

## 1. Round Conclusion

Phase 1 completed three interdependent minimum slices:

1. Reconcile current Core `ProgramArtifact` facts and scope.
2. Record long-term evidence levels, migration economics, Simulation/Runtime
   negotiation, Agent/LLM boundaries, and verification paths as Proposed ADRs.
3. Fix repeated `CircuitIR` import/verification performance gates without changing
   budgets, statistics, public APIs, or default compilation semantics.

No second Artifact envelope was introduced, and Proposed ADRs were not treated as
implemented contracts. `ProgramArtifact` v1 remains the sole envelope authority,
but covers only envelope behavior and the executed `kind=circuit` path. It cannot
losslessly replace Compiler sealed artifacts or Deployment packages.

## 2. Review and Merge Records

| Order | Slice | Team commit | Integration merge | Result |
| ---: | --- | --- | --- | --- |
| 1 | Core Artifact reconciliation | `c33292fb2b27bfb6dc94cbf58c53f05dd1966043` | `288cbd0b` | Passed |
| 2 | Phase 1 contract ADRs | `bc129bc668fa78ec4069483b90002d42e16ad291` | `23bb1c32` | Passed after revision |
| 3 | Compiler successor authorization | `80d3234c66bbf14f953ffc5ebb87e3807af71740` | `a7a9e8e6` | Passed |
| 4 | Compiler performance implementation | `cba6bad207d49fdc43acc6640e7cd67c0919c314` | `11c473cc` | Passed |

The initial Compiler implementation was constrained by historical SHA-256 workflow
records. Current executable verification uses implementation, scenario tests, and
performance budgets; Deployment Bridge phase approvals no longer serve as code contracts.

## 3. Key Decisions

### Artifacts and Metadata

- `ProgramArtifact` v1 is the sole artifact envelope authority; v2 must evolve
  within the same contract family.
- v1 `content_hash` is full-envelope identity, including producer, required
  capabilities, ordered parents, payload, and metadata.
- Metadata's value domain is not closed and is not a security boundary. A closed
  value algebra, namespaces, and size limits require a later API Change Proposal.

### Long-Term Architecture Decisions

- Evidence levels are `basic < observable < certification`, orthogonal to support
  status and fact exposure.
- Migration may `migrate`, `adapt`, `freeze_legacy`, or `retire`; immediate deletion
  of every legacy implementation is not mandatory.
- Simulation may supply typed resource candidates and cost suggestions; Runtime
  retains resource selection and scheduling authority.
- Agent Services remains deterministic and protocol-independent. Optional external
  LLM/reasoning capabilities do not enter stable Core/Compiler/Runtime contracts.
- Verification uses fast, standard, and certification paths, with machine rules
  tracking documented constraints.

### Compiler Performance Fix

- Cache only successful imports that passed full verification; never cache failures.
- The cache is process-local, bounded, weak-reference-based, and concurrency-protected,
  not a persistent compilation cache.
- Nested content changes, in-place tensor mutation, and equal-valued but distinct
  trainable tensors invalidate entries, triggering reimport, verification, and
  autograd-owner binding.
- Public APIs, default paths, accepted inputs, diagnostics, identities, round trips,
  budgets, and statistical methods remain unchanged.

## 4. Performance Results

In the standard Linux CPU container, five repeated-import rounds reduced 10,000-gate
p95 from approximately `98.304–131.938 ms` to `16.310–17.444 ms`. Full-gate pass rate
improved from `4/5` to `5/5`.

Independent integration review measured:

| Gates | p95 |
| ---: | ---: |
| 10 | 0.041 ms |
| 100 | 0.186 ms |
| 1,000 | 1.680 ms |
| 10,000 | 17.718 ms |

Latency, peak host memory, identity determinism, and normalized growth all met the
original budgets. One timing fluctuation occurred in the full default suite's
historical performance test. Thresholds were unchanged; per-scale review and a
subsequent full rerun passed. Retain this as a stability observation.

## 5. Unified Verification

- Architecture boundaries: passed.
- Combined Core, ADR, Compiler, Agent, import/round-trip/default-path tests after
  merge: `121 passed`.
- Linux Docker `pr-runtime`: `175 passed, 33 skipped`.
- Final Linux Docker `pr-default` rerun: `1867 passed, 12 skipped`.
- Compiler focused semantics/authorization chain: `110 passed`.
- Full Compiler `tests/internal_ir`: `867 passed`; one separate Stage 4 performance
  item fluctuated in the large suite, then passed isolated review: `2 passed`.

Skips reflect unavailable optional dependencies or accelerators in the container.
They do not establish hardware, distributed, or real QPU verification.

## 6. Next-Round Entry Points

1. Consolidate minimum `TargetCapabilities`, separating requirements, discovery
   facts, support status, and execution evidence.
2. Define minimum Execution Request and Result/Evidence contracts.
3. Approve Platform and Execution Provider method sets after contracts stabilize.
4. Remove Runtime's first pure-type dependency on Compiler.
5. Extract Simulation's first replaceable single-device statevector engine.
6. Observe historical Phase 1/Stage 4 performance-gate stability on shared CPUs,
   without relaxing budgets or changing statistics to remove fluctuations.

Each slice requires one owner, a minimal diff, contracts/characterization first,
team-scope and architecture gates, individual integration merges, and joint retesting.

## 7. Current Phase 2 Status (2026-09-04)

This section records integration of the minimum internal TargetCapabilities
Phase 2 slice. Runtime matching, fallback, decision records, and synthetic
producers are not public, default, or production capabilities. Closure review
baseline: `0f63c3b4`.

| Order | Phase 2 slice | Team commit | Integration commit | Current conclusion |
| ---: | --- | --- | --- | --- |
| 1 | Core TargetCapabilities v1 values, canonical identity, pure matcher | `69816f42`, corrected through `34580bcd` | `d7fad5e4`, `2083dd2e`, `35f1adb0` | Merged; internal, no policy/fallback |
| 2 | Compiler `_compiler.TargetCapabilities` adapter with loss accounting | `8c3e367e` | `1a501d8a` | Merged; uncovered fields still require the old comparator; Core matcher alone cannot admit |
| 3 | CPU Platform snapshot adapter | `38e8bf61` | `c30316b9` | Merged; independent CPU snapshots only, no discovery/selection/fallback authority |
| 4 | Runtime matching seam | `935adb25` (series begins at `4c8732e9`) | `b6403927` | Merged/verified; internal policy seam outside the default path |
| 5 | Synthetic Execution second-producer replacement conformance | `110ce4a1` (series begins at `6f87da7d`) | `0f63c3b4` | Merged/verified; producer replacement with unchanged Runtime consumer, not real remote/QPU |

ARCH-003/004/005/007 remain **Proposed**. Phase 2 machine authorization permits only
internal TargetCapabilities v1, narrow adapters, and the Runtime matching seam.
It does not approve new public Execution Request/Result/Evidence, stable plan
fields, default backends/fallbacks, or failure stages.

## 8. Runtime Matching Seam Acceptance Checklist

`[x]` means implementation/test evidence exists for the internal slice; `[ ]`
means deferred or incomplete evidence. Checks neither promote capability maturity
nor create public, default, hardware, or production claims.

### 8.1 Input Authority and Scope

- [x] Runtime consumes Core `RequirementSet`, `TargetCapabilitySnapshot`, and pure
  matcher results directly, without copying, re-exporting, subclassing, or creating
  parallel capability contracts.
- [ ] Preserve Compiler `requires_legacy_comparator` and every loss record; when
  true, reject candidates unless the old `_compiler` comparator passes. The seam
  has not integrated Compiler projection; this is required technical debt.
- [x] Consume requirements, snapshots, and explicit authorizations only. Do not
  fill platform facts from user expectations, `world_size`, environment variables,
  Simulation estimates, or backend registry booleans.
- [x] Core rejects deferred domains and unknown extensions without registered
  matchers. Do not expand public contracts here for dynamic execution, topology,
  communication, checkpointing, or realtime work.

### 8.2 Mandatory Rejection and Preference Ranking

- [x] Core fully checks each candidate's identity, scope, freshness, evidence
  references, and mandatory predicates. Missing/stale data, scope/identity
  mismatch, `unknown`, `unmeasured`, `unsupported`, `not_exposed`, unaccepted
  exposure, insufficient evidence, and unknown extensions make it nonexecutable.
- [x] Runtime policy cannot override mandatory blockers because a candidate is
  cheaper, higher-priority, or the only one available.
- [x] Preferences rank only candidates satisfying every mandatory requirement.
  Missed preferences lower rank; they do not authorize fallback or offset failure.
- [x] If all candidates fail, return aggregated typed blockers. Do not choose the
  nearest match or silently enter an existing default backend.

### 8.3 Deterministic Candidate Order

- [x] Identical requirements, snapshots, policy inputs, and evaluation time yield
  identical candidate order, selection, and decision identity.
- [x] Use versioned explicit sort keys: executability, preference satisfaction,
  controlled policy priority, then stable target/snapshot identities. Never use
  dict/set order, concurrent completion, discovery timing, addresses, or hash seeds.
- [x] Tests cover input permutations, duplicate candidate/snapshot/target
  identities, and equal scores. Different semantics under duplicate identity fail
  rather than use last-write-wins. Independent process hash-seed tests remain in
  section 8.8; explicit sort-key behavior is already established.

### 8.4 Independent CPU Candidates and Full Rematching

- [x] Explicit CPU requests enter ordinary matching. Moving from non-CPU to CPU
  creates a fallback candidate only with `fallback_authorizations.cpu=true`.
- [x] CPU uses its own target identity, scope, freshness, facts, evidence, and
  blockers and rematches every Core mandatory predicate. Joint legacy Compiler
  comparison remains deferred under 8.1.
- [x] Do not reuse the original candidate's memory, precision, device count,
  route, evidence, lease, or matcher success. Nullable unknown CPU memory/precision
  fails corresponding mandatory requirements.
- [x] Failed device resolution, empty candidates, unavailable probes, or local
  CPU availability cannot make CPU an implicit resolver default.

### 8.5 Independent Fallback Authorizations

- [x] Read explicit `backend`, `device`, `cpu`, `precision`, `algorithm`, and
  `approximation` authorizations independently; all default false. One axis never
  implies another.
- [x] Fully rematch each explicit fallback candidate with its own snapshot and
  constraints. Do not mutate failed candidates, delete mandatory requirements,
  or convert preferences into authorization.
- [x] Runtime policies, provider declarations, and historical broad
  `allow_backend_fallback` cannot imply CPU, precision, algorithm, or approximation authority.
- [ ] Selected fallback appears in internal decisions, but attempt evidence and
  claim-ceiling association are unimplemented. Stable plan/result field, identity,
  or exception changes require a separate API Change Proposal.

### 8.6 Typed Blockers and Decision Identity

- [ ] Blockers preserve Core codes, messages, capability names, and source context
  and aggregate in deterministic candidate order. Canonical blocker deduplication
  is not frozen. Do not reduce blockers to strings/booleans or hide them in logs.
- [x] Retain every evaluated candidate's snapshot identity and blockers. Final
  success cannot overwrite rejected-candidate evidence; preference misses are not
  fatal blockers.
- [ ] Internal decisions bind `requirement_set_id`, ordered snapshot identities,
  selected snapshot/target, evaluation time, per-axis authorization identity,
  matcher blockers, sort-key version, and actual fallback axes. Independent
  policy/candidate provenance identity remains unfrozen required debt.
- [x] Canonically derive identity from those semantic fields. Exclude display
  text, runtime objects, credentials, vendor handles, and unordered containers.
  Changes to requirements, snapshots, authorizations, or candidate order change identity.
- [x] Decisions exist only in the internal seam. Before ARCH-004/005 approval,
  do not call them public Execution Request/Result/Evidence or alter existing
  plan/result identities.

### 8.7 Public API, Default Path, and Layering Protection

- [x] Root exports, stable signatures/defaults/Literals, serialized schemas,
  exception categories, and failure stages remain unchanged.
- [x] Existing `fq.run`, `fq.plan`, default backend/device selection, and local
  CPU/single-device fast paths retain golden/compatibility behavior. The seam does
  not take over defaults without separate approval.
- [x] This slice does not integrate Simulation or rewrite algorithm constraints/
  cost models. Estimates are not available capacity; Simulation does not read
  environments or select physical targets. Layering defines ownership, not a ban
  on typed information exchange.
- [x] Implementations pass team scope and architecture checks. Production changes
  outside Runtime/Core/Compiler/Platform ownership or touching protected surfaces
  return to the owner or Integration/API proposal; do not hide them in this slice.

### 8.8 Required Runtime Test Evidence

- [x] Core/Runtime negative tests for mandatory status, exposure, evidence,
  freshness, scope, identity, and unknown extensions prove policy cannot bypass Core.
- [ ] Preference-only ranking, all-candidate failure, ties, permutations, and
  decision identity are covered. Independent process hash-seed tests remain
  determinism regression debt.
- [ ] CPU full-rematch covers memory, identity, unauthorized CPU, backend-only
  authorization, unavailable/missing facts. Device-count and native/effective
  precision negative matrices remain incomplete.
- [x] Default prohibition and isolation tests cover six fallback axes; authorized
  candidates still use the same complete Core matcher.
- [ ] A synthetic second producer replaces the CPU producer without Runtime
  consumer changes. Compiler loss accounting/legacy comparison remains outside
  the seam and deferred.
- [x] Stable API, default `fq.run/plan`, and local fast-path boundaries remain.
  Accelerator, QPU, multinode, and scalability claims stay unverified by this slice.

## 9. Specification Gaps and Admission Decisions

1. ARCH-004/005 propose future request/decision/result/evidence boundaries; current
   authorization does not freeze a stable decision schema. Minimal internal
   records may prove identity binding, but cannot replace or enter stable
   plans/results. Such a need pauses this slice for an API Change Proposal.
2. Compiler projections remain lossy for gate parameter domains, ancilla policies,
   and other fields and explicitly require the legacy comparator. Core
   `executable=true` is not complete Compiler legality.
3. CPU Platform and synthetic Execution producers passed replacement conformance
   with one Runtime consumer, satisfying the minimum two-producer boundary.
   Synthetic fixtures are not network/provider/QPU/hardware evidence and cannot
   promote maturity.
4. Dynamic execution, checkpoints, realtime, full topology/communication, and
   gradient/optimizer distribution remain deferred. This checklist defines
   rejection and extension gates, not feature or release authorization.

## 10. Minimum Phase 2 Workflow and Independent Verification

The internal workflow is complete: Core contracts/pure matcher -> Compiler adapter
with loss accounting -> CPU Platform producer -> Runtime policy seam -> synthetic
Execution second-producer replacement conformance. This proves cooperation among
internal types, matching, ranking, fallback authority, decision identity, and
producer replacement. It changes neither public APIs nor defaults and generates
no actual execution observations.

Independent integration verification:

- After Runtime merge `b6403927`: `121 passed`; architecture, team scope, Ruff,
  and diff checks passed.
- After synthetic producer fix/merge `0f63c3b4`: `118 passed`; architecture,
  team scope, Ruff, and diff checks passed.

Real providers/hardware, public APIs, defaults, execution results/evidence, claim
promotion, and deferred domains are not authorized. `synthetic_qpu` is an anonymous
test value, not QPU integration or remote execution capability.

## 11. Recommended Next Phase (Not Automatically Started)

### 11.1 Required Technical Debt

1. **Candidate provenance and trusted non-CPU fallback binding:** support origins,
   changes from the original request, and each fallback axis through verifiable
   diffs/adapter provenance. Do not permanently trust caller-reported
   `fallback_axes`. Freeze minimum internal blocker aggregation and policy identity.
2. **Joint Compiler legality:** consume projection losses in the seam without
   changing defaults; require legacy comparison whenever
   `requires_legacy_comparator=true`.
3. **Acceptance evidence:** add independent hash-seed and CPU device-count/
   native/effective-precision full-rematch negatives. These complete the approved
   seam rather than add product capabilities.

### 11.2 Future Features and Contracts

4. **Execution observation/evidence proposal:** first consolidate a side envelope
   for attempt identity, actual target/path, fallback, precision, distribution,
   and claim ceilings. ARCH-004/005 remain Proposed; do not alter stable plans/
   results before approval.
5. **Real domestic Platform certification prerequisites:** define vendor-neutral
   probes, physical identity, native precision, kernel residency, routes/no-fallback,
   evidence digests, and certification environments. A800/synthetic results cannot
   substitute for domestic hardware. Real adapters, remote jobs, and claims need
   separate authorization.
6. Keep dynamic, checkpoint/realtime, full topology/communication, and training
   distribution in separate proposals outside TargetCapabilities v1.

Recommended order: 1 -> 2 -> 3 -> 4 -> 5. Start item 6 per domain as real use cases
and evidence mature. This board orders work; it does not create or authorize
next-phase implementation tasks.

## 12. Current Integration Direction (2026-09-04)

The user explicitly authorized changing vNext's focus from horizontal contract
expansion to a minimum vertical product workflow. Follow this order without
starting parallel horizontal abstraction work:

```text
Freeze new horizontal abstractions
 -> Finish current Compiler consolidation
 -> Complete the minimum CPU vertical workflow
 -> Move proven code into authoritative directories
 -> Remove historical/transitional code in every round
```

Execution gates:

1. **Compiler consolidation:** complete joint legality using existing comparators,
   pass determinism/tamper/scenario tests, and remove unused wrappers and duplicate
   representations before delivery.
2. **Minimum CPU workflow:** one user-visible path spans input, compilation
   legality, Runtime selection, CPU simulation, results, failures, and basic
   evidence. Do not simultaneously build generic workflows, remote task centers,
   plugin marketplaces, or configuration frameworks.
3. **Real directory migration:** move only the proven path; update callers,
   ownership, and dependency checks together. No empty target directories or
   long-lived duplicate authorities.
4. **Subtraction each round:** handoffs separately list additions, reuse, removals,
   and frozen items, with old-entry owners, exit conditions, and target versions.
   Adding paths without resolving old ones is not completed migration.

The only active implementation task at this point is Compiler legality
consolidation. Do not begin production implementation of the minimum CPU workflow
before its integration review passes. Subsequent Platform, Execution, Simulation,
Ecosystem, and Agent work must arise from real needs of that workflow, not directory
coverage or contract counts.
