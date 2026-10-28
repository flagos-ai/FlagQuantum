# General-Purpose High-Performance Fault-Tolerant Quantum Simulation

## Master Plan, PR Roadmap, and Requirement-by-Requirement Acceptance Checklist

Version: 1.0, English master roadmap. Date: 2026-10-24, Asia/Shanghai.

Status: the user authorized the delivery sequence and conditional mainline merges on 2026-10-24. Detailed API contracts, suggested quantitative targets, approximation policies, package admission, and hardware scope retain the separate approval gates specified below.

Project: `flagos-ai/FlagQuantum`. Document owner role: project lead / Integration. Domain owners are assigned by work package below. Named contributors, hardware, experimental partners, and quantitative commitments must be established at the project entry gate.

**Status: a complete task plan, not capability certification, a performance commitment, or public API approval.** This document changes no implementation, existing PR, frozen fixture, or merge authorization. Implementation must follow approved contracts; new public interfaces, serialization, cross-domain semantics, approximation policies, and hardware execution scope require their respective approvals.

The plan includes a conditional sibling `ftqc` package proposal in Appendix B and mandatory runnable user examples for every PR in Appendix C. These are planning requirements, not authorization to create a package or activate an API. Standing merge authorization applies only to in-scope PRs whose exact reviewed head passes all required CI, review, and conflict gates; it does not bypass branch protection or approve unresolved contracts.

## 1. Mission and Overall Assessment

The original task, translated into English:

> Address the needs of large-scale quantum error correction and fault-tolerant quantum computing by researching key technologies for general-purpose, high-performance simulation of fault-tolerant quantum circuits. Advance efficient simulation of circuits combining Clifford and non-Clifford operations, dynamic measurements, and classical feedback control to simulate complex fault-tolerant circuits efficiently. Research efficient sampling of large-scale fault-tolerant circuits and overcome simulation-capacity and computational-performance bottlenecks. Adapt and optimize domestic computing backends. Build a general-purpose, high-performance fault-tolerant quantum simulation system supporting multiple quantum error-correcting codes, logical operations, and noise models. Provide high-performance simulation and verification tools for QEC algorithm research, logical-operation verification, and fault-tolerant architecture design, and cross-validate against experiments.

The deliverable is a **verifiable, reproducible fault-tolerant circuit simulation system with explicit support boundaries**. QEC package restructuring, batch decoding, detector error model (DEM) construction, and memory experiments are foundations. Completing any one foundation does not complete the mission.

The full delivery chain is:

```text
Codes and logical tasks
  -> error-correction / fault-tolerant circuits
  -> shared IR and compilation
  -> execution planning
  -> Clifford / mixed / noisy numerical execution
  -> measurement and classical feedback
  -> sampling and decoding
  -> logical outcomes and statistics
  -> performance evaluation and experimental cross-validation
```

“General-purpose” means verified combinations of codes, logical tasks, noise, and backends, with explicit refusal of unsupported combinations. It does not promise efficient classical simulation of arbitrary universal quantum circuits. “High-performance” requires measurements under frozen workloads, precision, hardware, and comparison conditions. Directory organization, API counts, green CI, and device discovery do not establish it.

In this document, **domestic computing hardware** means the Chinese domestic computing backends requested by the mission. It does not mean merely local execution, a device alias, or CUDA execution behind a compatibility wrapper.

## 2. Planning Basis, Baseline, and Evidence Levels

### 2.1 Revisions and Existing Work

The implementation baseline independently accepted in detail is main commit `4be2375eff47c112ce8d8e24f8206ed5592130d8`. The main revision queried while preparing the source plan was `ba4ee53d21993a5ef2c802ad30a6576ebfffe6cb`. The difference adds only M3 scenario acceptance and shared roadmap documentation, without new numerical implementation for this task. These are dated snapshots, not live status. Recheck latest main and concurrent PRs before starting each work package.

FTS-01 is prepared against main `cecf96b260cb81d3c34cc26469ada34fc390c81f`, which merged #794 after its 57 successful checks and clean review/conflict state. This adds acceptance/contract documentation, not batch implementation. The concrete 005 API remains unapproved. Main `cecf96b260cb81d3c34cc26469ada34fc390c81f` passed [complete CI, all 54 jobs](https://github.com/flagos-ai/FlagQuantum/actions/runs/40075840297), [source/single-GPU/required two-GPU validation](https://github.com/flagos-ai/FlagQuantum/actions/runs/40075851244) and [main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/40075781010). Three pre-existing optional manual GPU-pool checks were unselected. Fresh isolated sdist and wheel-from-sdist contain all 814 package Python members exactly; a fresh Python 3.12 core installation at the supported Torch 2.10.0 lower bound passes dependency consistency, native/Bell sampling and both public examples outside the checkout. This is one fresh core profile, not a claim that the prior 12-profile matrix was rerun. The initial default Torch 2.14.1 download hit local disk exhaustion; task-generated environments/cache were cleaned, and the explicit supported lower-bound profile was then accepted without changing project bounds or checks. Maximum-version validation remains covered by required CI.

| Area | Verified baseline fact | Reuse in this plan |
| --- | --- | --- |
| QEC restructuring and retirement | #749 and #766 established responsibility-based packaging and the approved retirement of historical entry points; one registered `RotatedSurfaceCode` remains. | Reuse the structure rather than restructuring it again. |
| Clifford foundations | Simulation contains stabilizer sampling, Pauli readout, positioned-noise sampling, and private fault propagation; QEC-003 merged in #779. | Reuse numerical boundaries; add end-to-end correctness and scale evidence for mission workloads. |
| Correlated Pauli faults | QEC-004 merged in #789, including exclusive fault groups. Existing independent decoders are not thereby certified for every correlated model. | Preserve exclusivity, propagation semantics, and capability refusals. |
| Compiled sampling and example statistics | #784 supplies reusable sampling; #788 supplies fixed-shot logical failure counts and Wilson intervals. | Reuse them. Public experiment entry points and adaptive stopping require separate approval. |
| Batch decoding | Native PyMatching Tensor logical prediction already exists. The #794 complete-result batching contract document has merged; implementation still awaits explicit approval. | Keep their result semantics distinct. #794 is not implementation approval. |
| Dynamic circuits | Proposal 084 records completed stable local dynamic execution through `fq.plan` / `fq.run`, including measurement, conditional operations, and existing execution strategies. | Reuse dynamic IR and Runtime; add QEC integration and mixed execution rather than another dynamic framework. |
| Domestic backends | Compute, Kernels, and domestic-accelerator / numerical-trust plans already exist. | Reuse device, precision, operator, and communication boundaries. This planning exercise does not certify domestic-device performance. |
| Multiple codes and logical tasks | Several code definitions and bounded execution capabilities exist. Definitions and declared distances do not establish complete executable fault-tolerant workflows for every code. | Publish per-code execution coverage; distinguish declarations from execution evidence. |
| Concurrent work | At the source-plan snapshot, #792 extends the Surface fixed-shot baseline and #786 evaluates AI decoding. #794 subsequently merged acceptance and batch-contract documentation. | Recheck current state, reuse delivered work, and avoid duplicate work or unauthorized modifications and merges. |
| M3 scenario acceptance | Latest-main M3 documentation distinguishes bounded physical reference scenarios, hardware certification, and remaining boundaries. | Other physical reference examples are not automatically FTQC experiments or QPU validation. |

The post-#789 baseline passed all 54 full-CI jobs, single-GPU and required two-GPU checks, source regression with 2,051 passes and one pre-existing inapplicable-case skip, and acceptance across 12 installation profiles and user examples. **These establish engineering acceptance for that revision, not completion of the mission's scale, domestic-hardware, or experimental-validation requirements.**

### 2.2 Status Vocabulary

- **Delivered:** a named revision, support domain, tests, and environment-specific evidence exist; the claim covers only that domain.
- **Partial foundation:** reusable implementation exists, but required workloads, combinations, or evidence are missing.
- **Research / implementation pending:** selection, implementation, or acceptance remains incomplete.
- **External dependency pending:** hardware, experimental data, permissions, or resource budgets are unavailable.
- An unchecked `[ ]` means the overall requirement remains open. An existing foundation API is not sufficient to check it off.

## 3. Requirement-by-Requirement Satisfaction Checklist

Each requirement needs a traceable chain: requirement -> use case -> implementation revision -> independent reference -> quantitative metric -> result -> limitations. The last column references this plan's FTS work packages, not GitHub PR numbers.

| ID / mission requirement | Current state | Required acceptance checklist | Primary work packages |
| --- | --- | --- | --- |
| R01 Large-scale QEC needs | Partial foundation | [ ] Execute workflows for repetition, Surface, another independent CSS code family, subsystem codes, and Floquet codes. Code declarations do not replace execution acceptance. Report resource curves against distance, rounds, and shots. | 02, 06–09, 23–24 |
| R02 Fault-tolerant computing needs | Partial foundation | [ ] Extend memory experiments to logical operations and measurement-based fault-tolerant protocols. Verify fault propagation, logical action, and failure conditions. Distinguish ideal logical correctness from fault tolerance. | 16–24 |
| R03 General-purpose FT circuit simulation technologies | Partial foundation | [ ] Connect at least three classes of end-to-end scenario through the same IR and execution entry. Replace a numerical implementation without changing consumers; user workflows must not depend on external SDK objects. | 03–04, 10–19, 36 |
| R04 Efficient mixed Clifford / non-Clifford simulation | Research and implementation pending | [ ] Compare algorithms and select a strategy. Check a bounded exact path against independent small-system references. Any adopted approximation has its own error budget. Demonstrate performance or capacity benefits on predeclared Clifford-dominated workloads. | 10–15 |
| R05 Dynamic measurement | Local foundation delivered; QEC integration pending | [ ] Preserve mid-circuit measurements, reset, repeated rounds, record ordering, and conditional semantics. Verify branch probabilities and conditional states. Define invalid def-use, zero-probability selection, and resource-limit behavior. | 04, 14, 16–17, 19 |
| R06 Classical feedback control | Local foundation delivered; FT loop pending | [ ] Complete measurement -> decoding -> classical decision -> physical correction / frame update -> subsequent logical operation. Handle frames correctly at non-Clifford boundaries. Report queueing and tail latency. | 14, 18–19, 22 |
| R07 Efficient complex FT circuit simulation | System acceptance pending | [ ] One task combines repeated correction, at least one logical two-qubit operation, non-Clifford injection, and feedback. Supply both a small end-to-end reference and target-scale resource evidence. | 15, 20–22, 35–36 |
| R08 Efficient large-scale FT circuit sampling | Partial foundation | [ ] Identify executed-circuit and declared-DEM sampling separately. Support compilation reuse, chunking, reproducible random streams, and bounded memory. Cross-check statistics against independent circuit execution. | 04, 07, 09, 19, 29–30 |
| R09 Overcoming simulation-capacity bottlenecks | Quantitative evidence pending | [ ] Expand the executable workload under equivalent semantics, precision, and resource budgets. Report qubits, rounds, non-Clifford resources, branches, and memory separately. Replicating tasks must not be called single-circuit capacity expansion. | 09, 12–15, 31, 36 |
| R10 Overcoming computational-performance bottlenecks | Quantitative evidence pending | [ ] Show end-to-end improvement on at least two representative workload classes with baselines and uncertainty intervals. Separate compilation, execution, sampling, decoding, transfer, and I/O. Report unsupported regions and regressions. | 07, 09, 15, 19, 29–31, 36 |
| R11 Domestic backend adaptation | Framework foundation exists; physical resources pending | [ ] Deploy on at least one preselected domestic device, verify numerical consistency, and preserve the same user workflow. Identify device, drivers, compiler, and libraries. No undeclared CPU fallback. | 27–28 |
| R12 Domestic performance optimization | Physical-device evidence pending | [ ] Optimize at least one complete target workflow based on measured bottlenecks, including conversion, compilation, and transfer. Meet precision and performance gates together. Qualification on a second device is a separately tracked extension. | 29; 32 is an extension |
| R13 Multiple codes, logical operations, and noise models | Partial foundation | [ ] Publish a support matrix by combination. Cover multiple codes, logical one-/two-qubit operations, and a non-Clifford protocol. Route independent/correlated Pauli, readout, coherent, and non-Pauli channels correctly or refuse explicitly. | 20–26 |
| R14 Tools for algorithm research, logical verification, and architecture design | Partial foundation | [ ] Deliver three reproducible user studies: decoder / logical-error comparison, logical-protocol verification, and architecture tradeoffs involving correction periods, decoding latency, and resources. | 05–09, 20–22, 35 |
| R15 Experimental cross-validation | Dataset and collaboration conditions pending | [ ] Use at least one traceable experimental dataset. Separate calibration from held-out validation. Compare detector statistics, correlations, and logical metrics, with uncertainty, residuals, and applicability limits. | 26, 33–34, 36 |

Report mission completion only after R01–R15 are closed and Section 12's system gates pass. Requirements blocked by external conditions remain open with a named blocker and owner.

## 4. System Scope, Architecture, and Ownership

### 4.1 Functional Boundaries

First vertical slice: Surface / independent Pauli / Clifford memory -> existing circuit sampling -> batch decoding -> logical statistics -> CPU baseline. Reuse completed components.

Second vertical slice: small encoded logical task -> intermediate measurement and conditional correction -> non-Clifford injection -> independent reference -> larger mixed-circuit execution.

Third vertical slice: a verified task above -> domestic device -> complete numerical and performance evidence -> held-out experimental comparison.

Each iteration completes an executable path from input through failure handling and evidence before expanding algorithms or code families. Domains must not build parallel IRs, controllers, device registries, or result systems.

| Domain | Authoritative responsibility | Responsibilities it must not duplicate |
| --- | --- | --- |
| QEC codes / circuits | Codes, checks, measurement schedules, detectors, and code-level logical/frame semantics. | General numerical evolution and vendor device control. |
| QEC dem / decoders / execution | Model construction, decoding, QEC experiment composition, and logical statistics. | A second Runtime or classical-control IR. |
| Compiler | Validate, lower, and legally optimize the existing IR, preserving measurement dependencies and noise positions. | Execution scheduling and shot policy. |
| Simulation | Clifford and mixed representations, trajectories, statevector / MPS / TN numerical algorithms. | Code registration, job scheduling, and vendor business logic. |
| Runtime | Execution plans, shot/branch orchestration, lifecycle, and bounded-resource execution. | Reimplementation of quantum algebra or external SDK semantics. |
| Kernels / Compute | Operators, devices, precision, synchronization, memory, and communication adaptation. | Complete per-vendor copies of simulation algorithms. |
| Remote / Ecosystem | Experimental/provider data and deployment format conversion with preserved provenance. | Presenting local simulation as QPU or experimental results. |
| Benchmarks / Integration | Independent references, benchmark protocols, evidence tracking, and system acceptance. | Removing failures by weakening thresholds or changing fixtures. |

Appendix B proposes a possible sibling `ftqc` package for higher-level fault-tolerant protocol semantics. It is conditional and does not replace these established owners.

### 4.2 Exactness and Provenance Categories

Distinguish exact algebra subject to floating-point error, truncation/decomposition approximations, finite-shot statistics, declared-DEM sampling, executed-circuit sampling, and experimental measurement records. Their errors, randomness, and provenance cannot be collapsed into one “error rate.” Reuse existing result and evidence fields; propose any necessary new public fields before implementation.

General non-Pauli or coherent noise must not silently become an exact Pauli DEM. Approximation, precision downgrade, backend substitution, and host staging require prior authorization and traceable reporting. Reject requests that cannot be satisfied under the accepted policy.

## 5. Key Technology Research Plan

### 5.1 Clifford Execution and Large-Scale Sampling

Reuse the stabilizer engine and its replacement boundary, private fault propagation, and compiled sampling. Measure construction, propagation, sampling, and decoding bottlenecks before optimizing bit packing, batching, caching, repeated-block compression, or I/O. Avoiding repeated compilation is a useful optimization, but it is not evidence of a new numerical algorithm.

Reference-frame methods and stabilizer representations provide candidate techniques without requiring reimplementation of mature dependencies. Establish their validity for the target dynamic and correlated-noise domain before adoption. The [Stim paper](https://arxiv.org/abs/2103.02202) and [official implementation description](https://github.com/quantumlib/Stim) support method selection, not performance claims for this repository.

### 5.2 Mixed Clifford / Non-Clifford Simulation

Run an algorithm-selection experiment before committing to a general mixed engine. Compare candidates on the same tasks:

| Candidate | Intended favorable domain | Boundaries to measure |
| --- | --- | --- |
| Bounded Clifford+T decomposition / stabilizer superposition | Encoded circuits dominated by Clifford operations with limited non-Clifford resources. | Term count, T count/depth, cancellation error, memory, conditional states, and sampling normalization. |
| Existing MPS / TN or a bounded active subspace | Circuits with manageable entanglement or contraction structure. | Bond dimension / contraction width, truncation error, observable evaluation, and sampling overhead. |
| Small statevector / density-matrix execution | Independent correctness references and local non-Clifford subtasks. | Capacity and precision limits; the same implementation cannot be its sole independent reference. |

FTS-10 selects one primary optimized implementation and the necessary reference path. The default candidate for FTS-12 is bounded exact Clifford+T decomposition. If the selection evidence rejects it, revise the plan and contract before replacing that package. Do not build three unverified frameworks in parallel.

Cost may still grow rapidly with non-Clifford resources. Do not promise scalability at arbitrary T density. Record exact/approximate classification, resource limits, error budgets, and refusals. See the [original low-rank stabilizer-decomposition research](https://quantum-journal.org/papers/q-2019-09-02-181/) for technical background.

Under dynamic measurement, coherent decomposition terms must not be sampled independently as though they were a classical mixture. Verify Born probabilities, interference, conditional normalization, and continuation. A prototype without correct sampling semantics may report bounded expectation-value research results, but cannot pass product sampling acceptance.

### 5.3 Dynamic Measurement, Feedback, and Logical Frames

Reuse stable `DynamicCircuit`, `CircuitIR`, and `fq.plan` / `fq.run`. Do not modify or merge #731. First prove that a small QEC schedule preserves measurement ordering, reset, classical def-use, conditional operations, and logical-observable mapping; then integrate decoder feedback.

Provide explicit comparable paths for physical correction and Pauli/Clifford frame updates. Convert or resolve frames correctly at non-Clifford and measurement boundaries; do not assume arbitrary corrections can be postponed indefinitely. Define the earliest feedback application point, decode failure/timeout behavior, and finite-buffer policy.

Accept software feedback simulation, offline replay, and real hardware real-time control separately. Real control additionally requires device clocks, interfaces, deadlines, loss handling, and recovery contracts. Software tail latency is not a substitute for a physical control loop when experimental resources are unavailable.

### 5.4 Codes, Logical Operations, and Noise

Add code families through complete workflows, not matrix or class registration alone. The target includes repetition, rotated Surface, another independent CSS family, one subsystem workflow, and one Floquet periodic-measurement workflow. The latter two first require approved gauge and spacetime-logical semantics.

Logical tasks progress through preparation/storage/readout, encoded Clifford operations, a logical two-qubit operation, lattice surgery, magic-state injection, and conditional correction. Small-system single-fault enumeration or appropriate independent arguments establish scoped fault-tolerance properties. Ideal agreement with a target gate is insufficient. Full magic-state-factory throughput is a later extension, not an implied capability of a small injection example.

The noise matrix covers independent Pauli, exclusive/correlated Pauli, readout/reset error, coherent over-rotation, and amplitude damping/dephasing. Reuse existing channel and noisy-execution routes for non-Pauli noise. A stabilizer-route refusal does not establish that the whole system lacks support; conversely, do not silently twirl a channel to satisfy that route. Leakage, crosstalk, and temporal drift require further model review when needed by the selected experiment.

### 5.5 Domestic Backend Adaptation and Performance

Reuse Compute, Kernels, FlagOS, Torch-FL, and numerical-precision plans. Run the shortest CPU-verified workflow on a named domestic device, identify the devices and operators actually doing the work, and then optimize measured hotspots. A domestic CPU may accelerate classical sampling/decoding while an accelerator handles mixed/tensor workloads; each claim needs evidence. Not every stage must be forced onto an unsuitable device.

Acceptance on at least one selected domestic device is mandatory. A second device is a separately tracked portability extension rather than a requirement inferred from the original task. Freeze device, driver, software stack, and two suitable workload classes in FTS-27. Development may proceed without hardware, but physical qualification remains blocked.

For devices without FP64/complex128 support, dtype names do not prove accuracy. Native, split-real/imaginary, mixed-precision, and extended-precision paths require independent qualification under the existing numerical-trust plan. Do not gain speed by relaxing an unapproved error budget.

## 6. Standard Workloads and Proposed Quantitative Targets

The following are **proposed project-entry targets, not measured achievements or approved commitments**. FTS-02 must freeze primary metrics, hardware budgets, baselines, and statistical protocols before examining optimization results. If resources are insufficient, revise and approve targets explicitly rather than selecting an easier metric after seeing results.

| Workload | Proposed initial matrix | Core acceptance |
| --- | --- | --- |
| W1 Small exact reference | 2–12 physical qubits; tractable branch/fault enumeration; noisy and noiseless. | Probabilities, conditional states, logical channels, and measurement records against independent high-precision references. |
| W2 Clifford memory | Start with Surface d=3/5/7, rounds=d and 3d; repetition and another CSS family. | X/Z readout, correlated errors, boundaries, statistical consistency, and a precise logical-failure definition. |
| W3 Capacity and long history | d=9/17/33 with up to 100 rounds; d=65 or longer histories as exploratory points. | Executable domain, peak memory, runtime, timeouts, and OOM. Do not require the entire Cartesian product. |
| W4 Mixed circuits | Sparse selections from n=8/16/32/64 and T=0/4/8/16/32. | Scan non-Clifford resources and entanglement structure. Larger points need inherited small-reference evidence and invariants, not an infeasible full-state reference. |
| W5 Adaptive fault tolerance | Injection, measurement, conditional correction, a logical two-qubit operation, and multiple correction cycles. | Branch probabilities, conditional logical results, frame equivalence, and feedback overhead. |
| W6 Sampling pipeline | 10^3–10^6 shots; streaming output with a fixed batch size. | End-to-end throughput, memory not accumulating linearly with total shots, random streams, and statistical consistency. |
| W7 Domestic hardware | At least two device-appropriate workload classes selected from W2/W4/W6. | Equivalent semantics/precision, physical-device evidence, and complete phase timings. |
| W8 Experimental validation | One preselected, legally usable dataset with separate calibration and held-out partitions. | Detector means/correlations, logical metrics, residuals, and uncertainty. |

Proposed initial acceptance targets:

1. **Correctness:** an absolute probability/expectation error of at most 1e-10 is suggested for small complex128 references. Complex noisy or deep workloads need separately justified tolerances. Discrete identifiers, logical mappings, registration, and identity must match exactly. This suggested tolerance neither applies universally nor replaces existing stricter thresholds.
2. **Statistics:** existing Wilson intervals may be used for fixed sample sizes. Predeclare family-wise error control for multiple comparisons. Adaptive stopping needs a method valid under its stopping rule; repeatedly checking ordinary fixed-sample intervals until a result is significant is invalid. Report zero-failure upper bounds and inconclusive outcomes when precision is insufficient.
3. **Performance:** target at least 1.5x end-to-end speedup on at least two frozen workload classes, with repeated-measurement uncertainty supporting a benefit; alternatively, one of those classes may demonstrate at least 2x expansion of its primary executable scale dimension under the same resources and semantics/precision. Overall acceptance still requires both throughput improvement and capacity-boundary evidence, rather than choosing only one.
4. **Sampling:** target a million-shot chunked W6 experiment within the selected hardware budget. With fixed batch size and record width, increasing total shots permits only explained bounded growth in peak memory. Freeze cache limits before setting the numerical memory budget.
5. **Feedback:** include window wait, conversion, decoding, and feedback in throughput and p50/p95/p99 software latency. Sustain the frozen arrival rate without an ever-growing queue. The experimental partner specifies physical real-time deadlines; this plan invents no microsecond requirement.
6. **Domestic performance:** measure optimization against an available portable baseline on the same device and report CPU/other-device context. Absolute differences between different hardware must not be attributed entirely to software optimization.

Every scale report records physical/logical qubits, gates, depth, correction rounds, shots, detectors, T count/depth, decomposition terms/bond/contraction width, dynamic branches, precision, noise, device memory, and time budget. One large-qubit success does not establish coverage of the whole matrix.

## 7. Phased PR and Work-Package Roadmap

FTS-01–36 identifies 36 reviewable work packages, not a guarantee of 36 additional PRs. Reuse completed work first. Scope each new PR to one concrete behavior or necessary contract, splitting complex packages where appropriate. Preserve original QEC-FUNCTIONAL identifiers and delivered history as subordinate-plan tracking.

All proposed titles follow `type(scope): description`. **Every PR, including documentation, tests, performance, and internal refactoring, must contain runnable user-facing example code, prerequisites, an execution command, and expected results.** A description of a future use case, pytest command, or hypothetical API is not a substitute. Appendix C defines the mandatory author/reviewer checklist. Contract changes precede implementation and require explicit approval.

### P0: Mission Alignment, Semantics, and Measurement

| ID | Proposed PR title and deliverable | Owner role | Dependencies | User scenario and completion gate |
| --- | --- | --- | --- | --- |
| 01 | `docs(ftqc): define mission traceability and delivery gates`: commit the master plan, connect subplans, and assign requirement/evidence ownership. | Integration | Review of this plan. | Existing QEC/dynamic example; all R01–R15 map to work; no new API. |
| 02 | `test(ftqc): establish reference workloads and performance budgets`: W1–W8, independent references, metrics, baselines, and resource budgets. | Benchmarks / Simulation | 01. | A runnable benchmark reports correctness, phase timings, memory, and refusals; freeze the pre-optimization baseline. |
| 03 | `docs(ftqc): specify execution and accuracy contract extensions`: only semantics that existing contracts cannot express; split by domain if needed. | Integration | 01–02. | Each change shows the current workflow, alternatives, and a failure example; create no type without need. |
| 04 | `test(ftqc): add sampling and feedback conformance cases`: independent measurement/frame/branch and sampling references. | Simulation / QEC | 02 and approval of relevant 03 contracts. | Two implementations of a small circuit agree on conditional distributions; invalid inputs fail early. |

### P1: Complete the Existing QEC Workflow and Establish a Baseline

| ID | Proposed PR title and deliverable | Owner role | Dependencies | User scenario and completion gate |
| --- | --- | --- | --- | --- |
| 05 | `feat(qec): add independent-shot decoding across supported decoders`: reuse the complete-result contract awaiting approval in #794. | QEC | Explicit approval and merge of #794. | Batch results equal complete per-shot records; plugin/stateful-stream boundaries remain intact; a loop wrapper alone is not a speed claim. |
| 06 | `feat(qec): compose reproducible memory experiments`: reuse #784/#788/#792 and implement original Decision C. | QEC | 05; prior approval of C. | X/Z memory reports shots, failures, intervals, and sampling source; failed decodes cannot disappear from the denominator. |
| 07 | `perf(qec): stream sampling and decoding with bounded memory`: compilation reuse, chunks, and accumulated statistics; use native logical prediction and full records for their respective purposes. | QEC / Runtime | 06. | Run fixed-batch W6 without ordering, RNG, or statistical regressions; review public lifecycle changes separately. |
| 08 | `feat(qec): add distance sweeps with uncertainty-aware analysis`: controlled distance/noise sweeps, stopping rules, and finite-size fitting. | QEC | 06 and statistics contract approval. | Export raw counts, intervals, and fit diagnostics; insufficient evidence cannot support a threshold claim. |
| 09 | `test(ftqc): measure Clifford capacity and sampling throughput`: complete CPU baseline, scale boundaries, and phase breakdown. | Benchmarks | 07–08. | Reproduce W2/W3/W6 and retain every timeout/OOM as later optimization context. |

### P2: Mixed-Circuit Research and Bounded Delivery

| ID | Proposed PR title and deliverable | Owner role | Dependencies | User scenario and completion gate |
| --- | --- | --- | --- | --- |
| 10 | `docs(simulation): select a bounded Clifford non-Clifford strategy`: compare decomposition and MPS/TN candidates and record selection. | Simulation | 02, 09. | Compare error/time/memory on the same W4 tasks and freeze the primary algorithm/support domain. |
| 11 | `test(simulation): add independent mixed-circuit reference cases`: small oracles for interference, gate order, measurement, and noise. | Simulation | 04, 10. | Independent dense evolution or analytic answers, rather than implementation self-consistency alone. |
| 12 | `feat(simulation): execute bounded Clifford T decompositions`: bounded exact path if the default candidate survives selection, reusing numerical boundaries. | Simulation | 10–11; approval of required 03 contracts. | Exact small W4 comparisons and resource limits at larger points; no fourth cross-domain numerical core. |
| 13 | `feat(simulation): enforce controlled approximation budgets`: reuse the chosen truncation/decomposition route only if needed and separately approved. | Simulation | 12 and approximation-contract approval. | Convergence and separate truncation/decomposition/statistical errors; otherwise keep this package conditional. |
| 14 | `feat(runtime): execute bounded adaptive mixed circuits`: connect existing dynamic control while preserving conditional probabilities. | Runtime / Simulation | 04, 12; dynamic-mixed contract approval. | W5 measurement and conditional operations produce the correct joint distribution; budget exhaustion fails explicitly. |
| 15 | `perf(simulation): optimize verified Clifford dominated workloads`: profile-driven block/cache/contraction/batch optimization. | Simulation / Kernels | 12, 14; also 13 if approximation is adopted. | Equal-precision end-to-end benefits against 09/10; report regressions outside the optimized domain. |

### P3: QEC Feedback, Logical Operations, and Code Families

| ID | Proposed PR title and deliverable | Owner role | Dependencies | User scenario and completion gate |
| --- | --- | --- | --- | --- |
| 16 | `docs(qec): specify dynamic schedules and logical frame semantics`: connect QEC schedules to existing dynamic execution; review semantics first. | QEC / Integration | 04; reuse stable dynamic contracts. | Static/dynamic interpretations and feedback timing for one syndrome schedule; no modification of #731. |
| 17 | `feat(qec): lower syndrome schedules to dynamic execution`: records, reset, unrolling/bounded loops, and logical mapping. | QEC / Compiler | Approval of 16. | Dynamic W2 extraction agrees with independent references; invalid dependencies are diagnosable. |
| 18 | `feat(qec): apply decoder feedback and logical frame updates`: equivalent physical and frame-correction paths. | QEC / Runtime | 05, 17; feedback-contract approval. | Correct fault-injected logical outcomes, explicit late/failure behavior, and non-Clifford frame-boundary tests. |
| 19 | `perf(runtime): batch adaptive error correction trajectories`: bounded dynamic batches, active-branch grouping, and queues. | Runtime | 14, 18. | Unbiased W5/W6 joint distributions, full p99 latency, and sustained throughput; kernel-only timing is insufficient. |
| 20 | `feat(qec): verify encoded Clifford operations`: preparation/readout, H/S, and logical CNOT on selected codes. | QEC | 17–18; logical-protocol approval. | Small-code logical channels/Pauli action match independent references with explicit fault coverage. |
| 21 | `feat(qec): execute a verified lattice surgery protocol`: one complete merge/split/logical-measurement protocol. | QEC | 20; schedule/boundary contract approval. | W5 two-logical-qubit action, detectors, and frames agree for ideal and fault-injected cases. |
| 22 | `feat(qec): execute magic state injection with feedforward`: one named injection protocol and conditional correction. | QEC / Simulation | 14, 18, 20. | Encoded T-type task includes a genuine non-Clifford input/operation; independent success probability, postselection denominator, and logical-action checks. |
| 23 | `feat(qec): execute a subsystem memory workflow`: original 010/011 using existing code records. | QEC | 06, 16; gauge/spacetime contract approval. | Complete gauge measurement -> detectors -> decoding -> protected logical outcome; distinguish bare and dressed distances. |
| 24 | `feat(qec): execute a Floquet memory workflow`: periodic schedule from original 010/012. | QEC | Approval of relevant contracts shared by 16/23. | Independent inter-cycle detectors, logical tracking, and spacetime fault-boundary checks. |

### P4: Noise Fidelity and Domestic Computing

| ID | Proposed PR title and deliverable | Owner role | Dependencies | User scenario and completion gate |
| --- | --- | --- | --- | --- |
| 25 | `feat(qec): preserve noise semantics across execution routes`: faithful routing of existing channels into FT workflows and a refusal matrix. | QEC / Simulation | 03, 06, 14. | Compare Pauli/coherent/damping variants of one small task against references; no silent DEM substitution. |
| 26 | `test(ftqc): validate calibrated noise and model discrepancy`: parameter-identification references, residuals, readout, and correlation diagnostics. | Benchmarks / QEC | 25; review the 33 data contract early. | Recover known synthetic parameters and predict held-out data; successful fitting alone is not experimental validation. |
| 27 | `docs(compute): define domestic device qualification for FTQC`: select the primary device, stack, workloads, precision, and budget. | Compute / Integration | 02; reuse existing domestic plans. | Identify execution location without another user API; establish resources and a device owner. |
| 28 | `feat(compute): qualify FTQC execution on a domestic device`: reuse providers and fill operator/numerical qualification gaps. | Compute / Kernels | Approval of 27; physical hardware available. | Execute W7 on the real device without implicit CPU fallback and pass equivalent-precision references. |
| 29 | `perf(kernels): optimize domestic FTQC execution hotspots`: optimize measured hotspots rather than rewriting algorithms. | Kernels | 15 or 19, plus 28. | End-to-end gain for at least one complete W7 task including transfer/conversion, without relaxed error budgets. |
| 30 | `perf(runtime): distribute FTQC shot batches reproducibly`: shot parallelism, independent random streams, count reduction, and failure behavior. | Runtime | 07/19, 28; distributed-contract approval. | Partition one statistical experiment across devices; identify throughput parallelism rather than single-state capacity expansion. |
| 31 | `perf(simulation): shard an approved FTQC numerical workload`: reuse SV/TN sharding only when capacity analysis establishes a need. | Simulation / Runtime | 15, 28; actual communication resources and approved sharding contract. | One logical numerical task is genuinely sharded; report communication, ownership, memory, and strong/weak scaling. Remain conditional without evidence. |
| 32 | `test(compute): qualify a second domestic execution profile`: extend portability without copying quantum algorithms. | Compute | 28–29; second device available. | Same-workflow/contract replacement tests and per-device operator/precision limits; separately tracked extension. |

### P5: Experiments, Architecture Studies, and System Acceptance

| ID | Proposed PR title and deliverable | Owner role | Dependencies | User scenario and completion gate |
| --- | --- | --- | --- | --- |
| 33 | `docs(ftqc): specify experimental data and validation protocol`: provenance, license, fields, measurement mapping, and calibration/held-out splits. | Experimental partner / Integration | 01; start early. | An accessible real/public dataset with clear semantics; missing inputs remain blockers, not replaced by synthetic data. |
| 34 | `feat(ftqc): cross validate simulated and experimental outcomes`: connect offline experimental records to the same logical-statistics workflow. | QEC / Ecosystem | 26; approval of 33 and available data. | Held-out detector means/correlations and logical metrics with residuals and uncertainty; disclose model failures. |
| 35 | `feat(ftqc): compare fault tolerant architecture tradeoffs`: study periods, latency, distance, and resources using existing workflows. | QEC / Benchmarks | 08, 19–22, 29; 34 may supply calibration input. | Three user-study examples, Pareto/resource curves, and assumptions; distinguish simulator time from physical time. |
| 36 | `docs(ftqc): record integrated scientific and performance acceptance`: final support matrix, reproduction bundle, and R01–R15 closure evidence. | Integration | All mandatory gates. | Full CI, installations, user examples, independent numerics, performance, domestic hardware, and experiments; unfinished extensions are not counted as delivered. |

If Appendix B's package boundary is approved, revise affected future titles and owners, such as logical-protocol scope from `qec` to `ftqc`, before opening those PRs. Keep existing PRs and historical identifiers unchanged.

## 8. Dependencies, Milestones, and Execution Order

The 36 packages are not one serial queue. Establish shared contracts first, then work only within boundaries that do not share an unproven contract. This document does not itself authorize parallel teams or subagents.

Critical sequence: 01–04 -> 05/06/09 baseline -> 10–15 mixed execution -> 16–22 adaptive logical workflows -> 28/29 domestic hardware -> 34 experiments -> 36 mission acceptance. Begin contract review in 16, hardware preparation in 27, and data preparation in 33 early enough to expose external blockers.

| Milestone | Meaning | Exit gate | Claims it does not establish |
| --- | --- | --- | --- |
| M0 Scope and evidence freeze | Mission, baseline, workloads, metrics, and external conditions are explicit. | 01–04, with named owners for 27/33 resource/data decisions. | Solved mixed algorithms or domestic performance. |
| M1 QEC experiment workflow | Existing Clifford memory/statistics become a scalable measured baseline. | 05–09; current results and complete compatibility acceptance. | Completed general FT computing. |
| M2 Mixed and dynamic workflow | Selected mixed route passes references and integrates feedback. | Required packages within 10–19; 13 only if approximation is adopted; explicit domain and precision. | Efficient simulation at arbitrary non-Clifford density. |
| M3 Logical tasks and multiple codes | Memory, gates, injection, and different measurement structures are executable. | 20–25 and a per-combination matrix. | All codes supported because their records exist. |
| M4 Domestic performance delivery | One named domestic device runs complete verified optimized workflows. | 27–30; 31 mandatory only if required by the approved capacity target; 32 a separate extension. | A simulated device or CUDA alias is domestic certification. |
| M5 Experimental and system delivery | Three research tools, held-out experimental validation, and mission evidence. | 26, 33–36 and all mandatory checklist items closed. | Documenting unresolved external blockers completes them. |

Set a schedule after M0 with separate estimates for algorithm research, domain implementation, hardware measurement, data collaboration, and review latency. Thirty-six packages do not mean thirty-six days. Do not give an overall deadline without staffing/device evidence. Reestimate remaining effort and risks after each milestone.

## 9. Reconciliation with Existing Plans and Concurrent Work

| Existing task/work | New-plan mapping | Treatment |
| --- | --- | --- |
| QEC-FUNCTIONAL-001/002 | Baseline and FTS-01/03. | Preserve delivery history; unapproved B/C remain explicitly pending. |
| 003/#779 and 004/#789 | Existing foundation for FTS-04/09/25. | Do not reimplement; extend combinations and resource evidence. |
| 005/#794 | FTS-05. | Obtain approval of the specific contract and retain complete-result/type semantics. |
| 006 native PyMatching | FTS-07. | Reuse its fast path; do not create another PR claiming the same batch API as new. |
| 007 and #784/#788/#792 | FTS-06/07. | Recheck merge state and fill only missing contracts/composition/semantics. |
| 008/009 | FTS-08/09 plus system performance protocol. | Accept threshold and performance claims separately. |
| 010–012 | FTS-16/23/24. | Review gauge, Floquet, and dynamic semantics before implementation. |
| #786 / AI plan | Independent algorithm work may feed FTS-35. | Reuse assessment data/protocols; AI benefit is not a prerequisite for basic simulator usability. |
| Existing stable dynamic execution | FTS-14/16–19. | Use existing entries; do not modify or merge #731. |
| Domestic, Kernels, and Noisy Simulation plans | FTS-25/27–32. | This master plan owns FTQC use cases/acceptance; domain documents retain implementation authority. |

This document is the mission-tracking entry. Domain documents remain authoritative for interfaces, architecture, and numerical semantics. Resolve conflicts through reviewable amendments; a master plan does not override existing approval requirements.

## 10. Performance and Capacity Verification Protocol

### 10.1 Three Different Scaling Claims

1. **Single-circuit capacity:** size supported by one quantum state, decomposition, tensor network, or complete dynamic task; report representation structure and memory.
2. **Multi-shot throughput:** independent shots of one experiment are batched or partitioned across devices. This can improve throughput without increasing single-circuit capacity.
3. **Genuine numerical-task sharding:** state, decomposition, or contraction work for one logical task is partitioned across ranks. Make this claim only with evidence satisfying the established sharding contract.

### 10.2 Required Contents of Each Performance Record

- Exact commit, dependency/compiler versions, device model, topology, threads, precision, memory, drivers, noise, workload, and input-generator version.
- Separate cold and warm measurements. Three warmups and ten independent timing repetitions are suggested starting points; preapprove alternatives for very long workloads. Record synchronization, timing boundaries, and raw observations.
- Phase timings, end-to-end wall time, shots/s, time per round/logical task, peak RSS/device memory, compilation caches, transfers, and I/O. Real-time studies also include p50/p95/p99 and queueing.
- Predeclared primary metrics and statistical/repetition methods. Retain failures, timeouts, and OOM; do not plot only successful points.
- Prefer equal-precision/semantics/device baselines. Explain algorithm, approximation, parallelism, and thread differences for cross-framework comparisons. Installation or compatibility is not performance evidence.
- Distributed records include `world_size`, `local_world_size`, `node_count`, rank ownership, communication, `distribution_semantics`, and blockers. Define work in strong/weak scaling explicitly.
- Published figures link raw observations and executable producers. Failed audits, unexecuted hardware paths, and empty result directories are not passing evidence.

## 11. Experimental Cross-Validation Protocol

### 11.1 Minimum Useful Experimental Scope

Prefer one repetition/Surface memory dataset with raw measurements, circuits/schedules, readout mapping, calibration, and rounds. Public experimental data can support the first stage. If on-device real-time control becomes mandatory, add a separately accepted physical-control milestone; offline reproduction cannot replace it.

The experimental partner confirms license, anonymization, version/checksum, device/acquisition time, qubit/classical/logical mapping, initialization/postselection, and the meaning of faults/missing data. Missing metadata narrows the interpretable scope or blocks acceptance; do not fabricate records.

### 11.2 Procedure

1. Import raw records with provenance; reconstruct detectors/logical observables and check basic counts against the publication or partner.
2. Predeclare calibration, validation, and final held-out partitions by batch/time/distance. Do not fit and claim predictive validation on the same data.
3. Estimate readout/gate-noise parameters and uncertainties on calibration data; report identifiability/model selection. First validate inference against known synthetic truth.
4. Freeze model/decoder configuration, then predict held-out detector means, spacetime correlations, per-round failures, joint logical failures, and per-observable counts.
5. Separate sampling, numerical/approximation, calibration uncertainty, and model discrepancy. Use predeclared intervals/tests/multiple-comparison procedures; an aggregate mean must not hide correlation errors.
6. Negative findings can be scientific deliverables. Close R15 only when the agreed procedure is complete and discrepancy explanations/support boundaries are reviewable; arbitrary disagreement does not automatically validate the model.

The Google Quantum AI [Surface-code experiment paper](https://arxiv.org/abs/2207.06431) is a candidate methodological reference. This plan has not confirmed that its raw data meet this project's ingestion requirements, nor treated published figures as already integrated experimental records.

## 12. System Acceptance Checklist and Evidence Template

### 12.1 Scientific and Functional Acceptance

- [ ] Every R01–R15 has a responsible owner and closure evidence; no completed item points only to a plan.
- [ ] Publish support/refusal by code family x logical task x noise x representation x dynamic capability x backend. The entire Cartesian product need not be supported, but gaps must be visible.
- [ ] Complete small independent references, target-scale invariants/convergence, and statistical testing at their respective layers.
- [ ] Validate mixed execution, dynamics, logical operations, and injection together in one end-to-end scenario, rather than only as isolated features.
- [ ] Preserve noise exclusivity/position, logical observables, and postselection denominators; make non-Pauli treatment and approximation transparent.
- [ ] Supply distinct sampling, throughput, and capacity evidence; complete domestic-device and held-out experimental verification.
- [ ] Provide executable user workflows for QEC algorithm research, logical-operation verification, and architecture analysis.

### 12.2 Engineering, Compatibility, and Delivery

- [ ] Preserve the original ordered 108 exports and separately approved mainline additions; existing class/pickle identities, one `RotatedSurfaceCode`, numerical behavior, and registration policy.
- [ ] Do not restore retired modules or alter the original 15 frozen fixtures. Cover approved new semantics with separate contracts and separately named evidence.
- [ ] No unapproved public interfaces, fields, defaults, exceptions, or serialization; no private parallel contracts to bypass review.
- [ ] Each PR directly targets main, uses `type(scope): description` and a non-`codex/` branch, and includes actual runnable user code and expected results under Appendix C.
- [ ] Full CI, required single-/two-GPU, installation, and domain checks pass at each reviewed head. Report pre-existing conditional non-selections separately; no threshold reduction or new skip to pass.
- [ ] After merging, revalidate main CI, fresh sdist/wheel installation, and examples across the supported Python/optional-dependency matrix.
- [ ] Run controlled performance and real-hardware acceptance separately; ordinary green CI cannot substitute for them.
- [ ] Include evidence, producers, dependencies, licenses, data permissions, and limitations; expose no credentials, personal sensitive information, or restricted data.
- [ ] Do not modify or merge #731, merge new PRs without authorization, or publish versions automatically. Apply existing authorizations only to their exact scope.

### 12.3 Per-Item Evidence Record

```text
requirement_id / work_package_id:
owner / reviewer:
status: planned | blocked | measured | accepted | rejected
source_commit / PR / reviewed_head:
contract_version / approval_record:
user_workflow / command / expected_result:
actual_result / raw_artifact / checksum:
hardware / software / dataset_version:
workload / algorithm / representation / noise / precision:
sampling_source / approximation / fallback:
independent_reference / numerical_tolerance / statistical_protocol:
performance_baseline / timing_boundary / repetitions / uncertainty:
failure_points / limits / unresolved_blockers:
acceptance_decision / approver / date:
```

This is a documentation ledger template, not a new public Python type or serialization protocol. Map it onto the existing evidence system instead of introducing a second registry.

## 13. Risks, Resources, and Decision Gates

| Risk / missing condition | Impact | Control and owner role | Decision point |
| --- | --- | --- | --- |
| Unmanageable non-Clifford cost at target scale. | R04/R07/R09 cannot meet targets. | Simulation compares decomposition and existing TN/MPS on identical tasks, with explicit refusal domains and no hidden approximation. | FTS-10, before M2. |
| Branch or decoder-queue explosion. | Throughput loss and unbounded latency. | Runtime budgets, batching, branch limits, and failure policies; measure the full workflow first. | FTS-14/19. |
| Domestic hardware or critical operators unavailable. | R11/R12 blocked. | Compute secures devices, stack, and permissions early; no physical certification without execution. | FTS-27, before M0 exit. |
| Low precision changes logical statistics. | Apparently plausible numerical output yields incorrect conclusions. | Simulation/Compute separate numerical and statistical budgets; approve precision policy first. | FTS-28/29. |
| Missing experimental mapping or permission. | R15 cannot close. | Experimental partner reviews the dictionary, license, and minimum useful data early. | FTS-33, before M0 exit. |
| A noise model fits but does not predict. | Invalid experimental validation. | Separate calibration/holdout and inspect correlations/residuals, not LER alone. | FTS-26/34. |
| Work stops at APIs and wrappers. | No demonstrated key-technology contribution. | Freeze algorithmic bottlenecks/baselines in 09/10; require numerical/performance evidence for M2/M4. | Every milestone. |
| Duplicate cross-domain work or concurrent-PR conflicts. | Maintenance burden and unreliable status. | Integration checks main/open PRs before each package and reuses authoritative boundaries. | Every PR. |
| Promise all combinations and devices at once. | Unbounded scope. | Finite mandatory matrix, separate extensions, reviewed scope changes. | M0 and milestone reviews. |

Minimum roles cover project/Integration, QEC, Simulation, Runtime/Compiler, Compute/Kernels, scientific benchmarking, and experimental collaboration. One person may hold multiple roles, but independent references and conclusions need independent review. Finalize staffing, device allocations, data ownership, and test budgets at M0. Before those are established, approve executable local stages rather than an unsupported total completion date.

## 14. Immediate Next Steps and Decisions Required

Review requirement mapping, mandatory scope, and proposed metrics, then prepare FTS-01/02 documentation/baseline PRs. Start FTS-27 device selection, FTS-33 data-availability review, and FTS-10 experiment design early to expose mission-level blockers.

The #794 acceptance/batch-contract document has merged without implementation. This master roadmap is a separate FTS-01 change. Implementation of its concrete batch API still requires explicit contract approval. The user separately authorized ongoing in-scope delivery and conditional merges after all exact-head CI/review/conflict gates pass; neither a green run nor a documentation merge grants approval of a pending interface or approximation policy.

Project-lead decisions:

- [ ] Accept R01–R15 as the mission closure map and M0–M5 as staged acceptance.
- [ ] Confirm mandatory code families, logical protocols, and complexity boundaries; decide whether second-device portability and physical QPU real-time control become mandatory.
- [ ] Approve scale, precision, performance, and statistical targets after the FTS-02 freeze; do not treat this draft's suggested numbers as commitments.
- [ ] Name the first domestic device, resource owner, experimental partner/dataset, and resource budgets.
- [ ] Separately approve necessary mixed-representation, approximation, QEC dynamic-bridge, and public-experiment contracts. Approval of this plan does not approve interfaces.
- [ ] Review Appendix B's sibling-package admission conditions before creating `flagquantum.ftqc`.

Subsequent reports should state verified capability gains, closed requirements, and remaining blockers by R01–R15/milestone, with PR links. A remaining PR count is not a measure of mission completion.

### Delivery Tracking

| Item | State | Evidence / next gate |
| --- | --- | --- |
| FTS-01 | Merged in [#799](https://github.com/flagos-ai/FlagQuantum/pull/799); post-merge acceptance complete | [Exact main, CI, source, fresh artifacts/install and examples](../development/FTQC_001_MAINLINE_ACCEPTANCE.md). |
| FTS-02 | #805 merged; baseline-only collection explicitly authorized; first CPU observations recorded; remaining scope/review pending | [FTS-02A decision](../development/FTQC_REFERENCE_WORKLOADS_AND_BUDGETS.md#baseline-only-decision) and [FTS-02B raw observations](../development/FTQC_002_CPU_BASELINE.md). [W6 phase observer](../development/FTQC_SAMPLING_PHASE_BASELINE.md) fills the W6 measurement gap through existing compiled sampling/prediction; its mainline acceptance and the [W1-A exact phase baseline](../development/FTQC_EXACT_REFERENCE_BASELINE.md) record the next bounded observation. [Post-820 main acceptance](../development/FTQC_EXACT_REFERENCE_BASELINE.md#post-820-mainline-acceptance) is complete. The [bounded scale pilot proposal](../development/FTQC_REFERENCE_WORKLOADS_AND_BUDGETS.md#fts-02e-bounded-scale-campaign-proposal) received explicit Stage A approval; the [supervisor admission and pilot records](../development/FTQC_REFERENCE_WORKLOADS_AND_BUDGETS.md#supervisor-admission-and-stage-a-pilot-observations) retain all twelve bounded process observations. Proposed 1.5x/2x targets remain disabled; no capability promotion or full FTS-02 closure. |
| FTS-03 / preparatory FTS-04/16 | Ownership proposal #833 merged and main accepted; finite matrix approved on 2026-10-25, delivered in #837 and main accepted | [QEC/FTQC protocol boundary](../architecture/QEC_FTQC_PROTOCOL_BOUNDARY.md), [post-833 main acceptance](../development/FTQC_REFERENCE_WORKLOADS_AND_BUDGETS.md#post-833-mainline-acceptance) and [finite schedule/feedback matrix](../development/QEC_SCHEDULE_FEEDBACK_CONFORMANCE_PROPOSAL.md). Reuse existing repetition feedback; no new API, frame, package or complete FTS-02/system closure. |
| FTS-10A | Finite study approved; #848 observer/main accepted; one campaign stopped on monitor failure, 41/48 points complete; final method selection open | [Existing route inventory, finite comparison design and admission](../development/FTQC_MIXED_STRATEGY_STUDY_PROPOSAL.md). [Partial raw evidence, monitor failure and post-848 main acceptance](../development/FTQC_MIXED_CPU_PARTIAL_BASELINE.md): 164 completed processes, one failed host warm-up and 27 unattempted slots, no retry. No new engine, approximation or W4 capacity/performance claim; [Seven-point continuation admission](../development/FTQC_MIXED_CPU_CONTINUATION_PROPOSAL.md) is proposed, not approved; study and FTS-02/09 dependencies stay open. |
| FTS-05 / QEC-FUNCTIONAL-005 | Contract document merged in #794; implementation approval pending | Do not activate the new cross-decoder operation until its concrete contract is explicitly approved. |
| FTS-27 / FTS-33 | Resource and data decisions pending | Name the device, owner, dataset, permissions, and budgets before hardware/experimental certification. |

## 15. References and Reused Documents

Repository links below are pinned to the Section 2 snapshots and can be updated after re-audit. Papers inform technical choices; they do not prove implementation or measurements in this project.

- [QEC functional subplan](https://github.com/flagos-ai/FlagQuantum/blob/4be2375eff47c112ce8d8e24f8206ed5592130d8/docs/roadmap/QEC_FUNCTIONAL_DEVELOPMENT_PLAN.md)
- [Stable dynamic execution contract and delivery record](https://github.com/flagos-ai/FlagQuantum/blob/4be2375eff47c112ce8d8e24f8206ed5592130d8/docs/development/API_CHANGE_PROPOSAL_084_DYNAMIC_EXECUTION_UNIFIED_RUN.md)
- [Clifford stabilizer simulation and replacement boundary](https://github.com/flagos-ai/FlagQuantum/blob/4be2375eff47c112ce8d8e24f8206ed5592130d8/flagquantum/simulation/stabilizer/README.md)
- [Domestic accelerators and numerical trust](https://github.com/flagos-ai/FlagQuantum/blob/4be2375eff47c112ce8d8e24f8206ed5592130d8/docs/roadmap/DOMESTIC_ACCELERATOR_AND_NUMERICAL_TRUST_PLAN.md)
- [Quantum kernel architecture](https://github.com/flagos-ai/FlagQuantum/blob/4be2375eff47c112ce8d8e24f8206ed5592130d8/docs/roadmap/QUANTUM_KERNEL_ARCHITECTURE_PLAN.md)
- [Noisy simulation roadmap](https://github.com/flagos-ai/FlagQuantum/blob/4be2375eff47c112ce8d8e24f8206ed5592130d8/docs/roadmap/NOISY_SIMULATION_ROADMAP.md)
- [Latest M3 scenario acceptance and remaining boundaries](https://github.com/flagos-ai/FlagQuantum/blob/ba4ee53d21993a5ef2c802ad30a6576ebfffe6cb/docs/development/M3_SCENARIO_ACCEPTANCE.md)
- [#794 batch-contract proposal](https://github.com/flagos-ai/FlagQuantum/pull/794), not implementation approval.
- [Stim: a fast stabilizer circuit simulator](https://arxiv.org/abs/2103.02202)
- [Simulation of quantum circuits by low-rank stabilizer decompositions](https://quantum-journal.org/papers/q-2019-09-02-181/)
- [Suppressing quantum errors by scaling a surface code logical qubit](https://arxiv.org/abs/2207.06431)

## Appendix A. Runnable Current-API User Baselines

These examples use existing public APIs, not proposed imports. They were executed successfully against the accepted mainline wheel in the Python 3.12 optional-dependency installation profile while preparing the source plan. Revalidate them at the actual reviewed revision of a new PR.

### A.1 Complete Single-Shot Records from Multiple Decoders

```python
from flagquantum.qec import DetectorErrorModel, get_decoder

model = DetectorErrorModel.from_stim_text(
    "error(0.125) D0 L0\ndetector D0\nlogical_observable L0\n"
)
for name in (
    "minimum_weight_matching", "belief_propagation", "belief_propagation_osd"
):
    decoder = get_decoder(name, model)
    results = tuple(decoder.decode(shot) for shot in ((), (0,)))
    print(name, tuple(result.observables for result in results))
```

Expected stdout:

```text
minimum_weight_matching ((), (0,))
belief_propagation ((), (0,))
belief_propagation_osd ((), (0,))
```

This proves the current single-shot workflow is runnable. It does not imply the proposed cross-decoder batch API exists or improves performance.

### A.2 Dynamic Measurement and Software Classical Feedback

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)
result = fq.run(circuit, shots=128)
assert isinstance(result, fq.ExecutionResult)
assert result.samples is not None
assert result.samples.shape == (128, 2)
assert bool((result.samples[:, 0] == result.samples[:, 1]).all())
print("Dynamic samples:", tuple(result.samples.shape))
print("Feedback parity:", bool((result.samples[:, 0] == result.samples[:, 1]).all()))
```

Expected stdout:

```text
Dynamic samples: (128, 2)
Feedback parity: True
```

This establishes the existing local control path, not completed QEC feedback, large-scale mixed execution, domestic hardware acceptance, or QPU real-time control.

## Appendix B. Conditional Sibling `ftqc` Package Proposal

### B.1 Recommendation and Admission Conditions

A sibling `flagquantum/ftqc/` is reasonable when executable logical protocols require domain semantics beyond error correction. Do not create it merely because the mission includes “FTQC,” or move all QEC functionality into it.

Consider admission when at least two concrete workflows, such as lattice surgery and magic-state injection, need shared encoded-block composition, protocol-level logical dependencies, or frame transitions that cannot be expressed cleanly by the existing QEC code/schedule and Runtime contracts. Demonstrate the duplication or misplaced responsibility with actual workflows. If they only compose stable calls without new domain semantics, use the existing Services boundary instead.

Record an architecture/API proposal under Integration first. It must explain why `qec`, `algorithms`, and `services` cannot already own the work, and prove reuse of existing IR, Runtime, Simulation, noise, and device boundaries. A new package is not permission for another general manager, protocol registry, execution engine, or public result hierarchy.

### B.2 Ownership and Dependency Direction

| Responsibility | Recommended owner |
| --- | --- |
| Code records, check matrices, code families, code-level logical operator certification. | Existing `qec.codes`, shared QEC algebra, and `qec.logical`. |
| Syndrome extraction, detectors, DEM, decoders, memory experiments, and correction-round primitives. | Existing `qec.circuits`, `dem`, `decoders`, and `execution`. |
| Encoded-block composition, cross-block logical protocols, lattice surgery, injection/feedforward, and protocol-level frame transitions. | Proposed `ftqc`, after boundary approval. |
| Classical def-use, control-flow IR, compilation, execution plans, shots/branches, and lifecycle. | Existing Core / Compiler / Runtime. FTQC lowers into them. |
| Clifford/non-Clifford evolution, representations, trajectories, and numerical precision. | Existing Simulation / Kernels. No `ftqc` numerical engine. |
| Device discovery, kernels, synchronization, and communication. | Existing Compute / Kernels / Runtime. |
| Gate tallies and existing logical resource estimation. | Reuse `compiler.resource_estimation` and `algorithms.logical_resources`; distinguish declared cost models from execution measurements. |
| Transport-facing reusable orchestration without FT domain semantics. | Existing Services; CLI/MCP/notebooks remain thin adapters. |

Dependency direction: FTQC protocols may consume public QEC primitives and lower through established compilation/execution boundaries. QEC must not import FTQC. Simulation and Compute must remain independent of FTQC. Consumers must not reach into private numerical or decoder internals to bypass contracts. New logical-scheduling semantics are domain inputs to the existing compiler/runtime, not a second program IR.

### B.3 Candidate Layout, Created Incrementally

The target has sibling `qec` and `ftqc` packages sharing the existing compilation, execution, numerical, and device infrastructure. The following is an eventual responsibility map, **not a request to create empty directories now**. It is not a literal inventory of current files. Existing module locations remain unchanged unless a separately approved migration preserves their contracts and identities.

```text
flagquantum/
├── qec/                    # error-correction primitives and memory workflows
│   ├── codes/              # code definitions, check matrices, families, registration
│   │   ├── surface/
│   │   ├── subsystem/      # grow only with concrete executable workflows
│   │   └── floquet/        # periodic-code definitions; schedules belong to circuits
│   ├── circuits/           # syndrome extraction, schedules, detector declarations
│   ├── dem/                # error-model construction, text, chunks, and graphs
│   ├── decoders/           # matching, BP, BPOSD, windows, and approved batching
│   ├── execution/          # correction rounds, memory experiments, sampling, statistics
│   ├── adapters/           # optional external decoder-library adaptation
│   ├── logical.py          # code-level logical-operator certification and readout
│   ├── pauli.py            # shared Pauli algebra
│   └── gf2.py              # shared GF(2) operations
├── ftqc/                   # conditional; create only after approved admission
│   ├── README.md           # ownership, dependencies, supported golden path
│   ├── __init__.py         # minimal approved public surface
│   ├── logical.py          # encoded-block/protocol composition, if truly needed
│   ├── frames.py           # protocol-level frame transitions; reuse QEC algebra
│   ├── protocols/
│   │   ├── encoded_clifford.py
│   │   ├── lattice_surgery.py
│   │   └── magic_state_injection.py
│   └── analysis.py         # protocol studies using existing resource estimators
├── core/                   # shared IR, contracts, and foundational types
├── compiler/               # shared IR validation/lowering/optimization
├── runtime/                # shared execution planning and orchestration
├── simulation/             # numerical algorithms; no code or protocol registry
│   ├── stabilizer/         # Clifford evolution, sampling, and fault propagation
│   └── ...                 # existing SV/MPS/TN/noisy paths and approved mixed methods
├── kernels/                # reusable numerical operators
├── compute/                # directly controlled device infrastructure
├── noise/                  # shared channels and noise models
├── algorithms/             # existing algorithm tools and logical-resource estimates
├── ecosystem/              # external formats and data conversion
├── remote/                 # external platform and device-job adaptation
└── services/               # reusable application workflows and preflight
```

Start with the smallest approved file set for one complete protocol. Add `logical.py`, `frames.py`, or a subpackage only when its distinct responsibility is demonstrated; retain flat modules if that is clearer. Do not add generic `ftqc.backends`, `ftqc.decoders`, `ftqc.simulator`, or `ftqc.runtime` copies.

The mixed-simulation method belongs inside the existing Simulation architecture after FTS-10 selection. The `...` above deliberately avoids naming or freezing another engine package before that decision. Likewise, the subsystem/Floquet layout is a candidate organization rather than authorization to move existing code records.

#### User Examples, Tests, Benchmarks, and Documentation

```text
examples/
├── qec/                    # memory, sampling, decoding, logical-error experiments
└── ftqc/                   # encoded gates, surgery, injection, and feedback

tests/
├── qec/                    # existing correction semantics and compatibility
├── ftqc/                   # protocol correctness, frame transitions, fault injection
└── integration/            # complete workflows crossing established domain boundaries

benchmarks/                 # capacity, throughput, latency, and domestic-device evidence
docs/                       # contracts, guides, support matrices, and acceptance records
```

Existing Simulation, Runtime, Compiler, Compute, and kernel tests remain in their domain-owned locations. The `tests/ftqc/` directory tests protocol semantics; it does not absorb numerical-engine or device-conformance suites. Benchmark producers and evidence remain in the existing benchmark infrastructure rather than a parallel FTQC-only harness. Every example uses actual public APIs at its PR head and includes prerequisites, a run command, and expected results under Appendix C.

#### Responsibility Questions

| User or implementation question | Authoritative home |
| --- | --- |
| What logical operators does a code protect, and how is its syndrome extracted? | QEC. |
| How do encoded blocks compose into surgery or injection protocols? | FTQC, if admitted. |
| How does a measured bit control a later branch? | Existing Compiler / Runtime control semantics. |
| How is a Clifford/non-Clifford quantum state evolved efficiently? | Simulation, using reusable Kernels. |
| How is work executed and optimized on a domestic device? | Compute / Kernels, orchestrated by Runtime. |
| How are gates and logical resources counted? | Reuse Compiler / Algorithms estimates; FTQC adds protocol-level analysis. |

Prefer adding a complete new protocol at its approved home while retaining existing QEC classes in place. Directory organization must not create a second simulator, Runtime, noise implementation, or hardware-adapter stack.

### B.4 PR Sequence and Compatibility

1. **Boundary proposal:** extend FTS-03/16 review with `docs(architecture): define QEC and FTQC protocol ownership`. Include runnable current QEC/dynamic examples and concrete protocol requirements. This is not package/API activation.
2. **Approved contract and conformance:** only if needed, propose minimal protocol semantics, logical mapping, frame/failure behavior, and permitted dependencies. Review names and serialization before implementation. Add an architecture/dependency check and a replacement/conformance scenario using the existing test infrastructure, not a new registry.
3. **First complete protocol:** adapt FTS-20/21 scope to `feat(ftqc): execute a verified encoded protocol`. Reuse existing QEC records and execution. Include actual public-user code with ideal/fault-injected expected results.
4. **Second concrete consumer:** FTS-22 adds injection/feedforward and proves that shared protocol/frame semantics serve both workflows without consumer-specific engine changes.
5. **Analysis integration:** FTS-35 consumes protocol outcomes and existing resource estimates; no duplicated gate counting, assumed hardware error rate, or invented factory cost.

Prefer adding new protocols at their approved home over moving existing classes. Any proposed move requires a separate class identity, module/pickle path, public-import, plugin, and serialization analysis. Preserve existing QEC identities and frozen fixtures; the earlier retirement of 23 modules does not authorize further breakage. Do not re-export FTQC symbols from the historical QEC root or append stable exports without explicit approval.

This sequence is a possible decomposition of existing FTS packages, not an automatically authorized extra project or fixed additional PR count. Report the concrete boundary proposal before creating the package.

## Appendix C. Mandatory User Examples for Every PR

This applies to **every** PR: documentation, contract, implementation, internal refactor, test, benchmark, backend adaptation, and final acceptance.

- [ ] State the user task, required package extras, supported Python/device/data prerequisites, and whether credentials or physical hardware are needed.
- [ ] Include copy-paste runnable code in the PR description, or a concise runnable public-API snippet plus an exact repository example invocation. A test command or benchmark-internal helper alone is insufficient.
- [ ] Use public APIs available at the reviewed head. Label proposed API illustrations separately; they cannot replace a runnable current-API example in a proposal PR.
- [ ] Provide the exact execution command and working-directory/install assumptions. Documentation-only PRs may reuse an existing workflow and explain the user impact.
- [ ] Provide expected stdout, shapes, invariants, or scientifically justified statistical bounds. Do not pin cross-engine random bit patterns or claim predetermined timing numbers.
- [ ] For behavior changes, make the new result, newly accepted task, or explicit refusal visible to the user. For internal changes, show preserved observable behavior and, when relevant, separately measured performance.
- [ ] For unsupported or error-prone workflows, include a relevant public refusal/failure example with the expected exception or diagnostic, without inventing future exception behavior.
- [ ] Keep examples short enough for a contributor's ten-minute golden path. Do not require users to manipulate internal provenance IDs, private registries, or scheduler objects.
- [ ] Reexecute the example at the final reviewed head; retain actual output and environment evidence. Validate installed-package behavior outside the checkout where applicable.
- [ ] Hardware PRs provide both a reproducible core-only sanity example and the real device-qualified workflow. The CPU example cannot replace required hardware execution.
- [ ] Performance PRs include an executable user workflow as well as the benchmark command, timing boundary, baseline, and uncertainty. Never describe green CI as a speedup.
- [ ] Final review blocks handoff if example code, prerequisites, expected results, or execution evidence are missing. Do not weaken tests or snapshots to make an example pass.

Appendix A supplies two already executed public examples for immediate documentation/contract work. Future feature PRs must add their own actual affected workflow after its contract is approved; this plan deliberately does not fabricate runnable code for APIs that do not yet exist.

### FTS-10A continuation controller review

The human explicitly approved the seven-point continuation in
[the bounded proposal](../development/FTQC_MIXED_CPU_CONTINUATION_PROPOSAL.md):
28 new workload processes, cumulative 193 attempts, unchanged RSS/query/gap
policy, controller review first. The separate private controller retains primary
supervision failures before producer validation and has fake boundary tests.
Implementation review and mainline admission completed before the separately
reported one-time probe and continuation below. Proposal main8e40 passed full CI/GPU/fresh-install/public
examples. FTS-10, FTS-02/09 and R01-R15 remain partial; no method selection,
performance target, new API or hardware admission follows from this work.

### FTS-10A admitted pilot observation matrix completed

The [separate seven-point continuation](../development/FTQC_MIXED_CPU_COMPLETED_PILOT.md)
completed once after #863 controller review and main admission: one non-study
probe, 28 new workloads (seven warm-ups/21 measurements), no retry. Original
663-file inventory is unchanged. Combined pilot evidence contains 48 complete
points/192 successful processes plus one original monitor failure, 193 attempts.
This closes only the specifically admitted finite observation collection. FTS-10
method selection, FTS-02/09 and R01-R15 remain partial; no performance target,
capacity, API, approximation, hardware or further recovery admission follows.

### FTS-10B / FTS-11A finite coherent-reference conformance delivered

Pilot-evidence main7720 passed complete CI/GPU, real installation evidence,
fresh core/public examples and exact data/inventory checks. The
[finite coherent-branch proposal](../development/FTQC_COHERENT_BRANCH_CONFORMANCE_PROPOSAL.md)
received specific human approval on 2026-10-25 for n=2/3, t=0/1/2/4 private
reference conformance, two analytic anchors, phase/mixture/cancellation negatives
and complex128 atol=1e-12/rtol=0. Proposal main0e7a then completed all admission
gates, including 54 CI jobs, required GPUs, fresh installation and installed user
examples. The independent test-only implementation is delivered: 50 private
scalar branch vectors and ten existing public unexpanded CPU calls per module
verification, with prior thread settings restored. No production phase contract,
public API, new engine, benchmark collection or performance target follows.
This contributes to R04 reference correctness; FTS-10 selection and
FTS-02/09/R01-R15 system acceptance remain partial.

Conformance main1f3916 passed all 54 CI jobs, required GPUs, real QEC installation
checks, fresh 834-member core installation and installed runnable user examples.
Its focused source check passed 68 tests without skips, including 23 new finite
reference checks and original retirement contracts. No pilot campaign was rerun.

### FTS-11B finite dynamic coherent-reference conformance delivered

The [three-case finite proposal](../development/FTQC_DYNAMIC_COHERENT_REFERENCE_PROPOSAL.md)
received specific human approval on 2026-10-26 for private Born weights,
conditional normalization, record-preserving reset and coherent measurement
continuation. Proposal mainebc2 completed all admission gates: 54 CI jobs,
required GPUs, real installation checks, fresh core and installed user examples.
The private test implementation is delivered: 20 four-amplitude projected
Clifford terms and three existing public density calls per module verification,
with the separately approved complex128 atol=1e-12/rtol=0. It retains the static
qualification and independently checks direct-T matrices and analytic answers.
It does not qualify trajectory sampling, create a branch API/engine, or activate
a collection budget.
R04/R05, FTS-10/11 and the system-level milestones remain partial.

Conformance mainc95e passed 54 CI jobs, required GPUs, real QEC installation,
fresh 836-member core and installed user examples. Its focused source check
passed 95 tests without skips, including the 17 new E-H checks, existing static
and dynamic references, and original retirement contracts. Both original study
campaigns remain unchanged; no new collection was run.

### FTS-11C finite trajectory Born-conformance proposal

The [Runtime/reference boundary proposal](../development/FTQC_TRAJECTORY_BORN_CONFORMANCE_PROPOSAL.md)
requested separate I-L approval for three current dynamic programs, real
multinomial inputs, controlled positive-branch coverage and stable result
projection. The approved cap is 19 existing local calls and 82 trajectories per
module verification, with atol=1e-12/rtol=0 and no empirical-frequency threshold.
It distinguishes natural draws from controlled coverage and does not certify
RNG quality, a production decomposition or efficient large-scale sampling.
The user separately approved this boundary; E-H and pilot budgets alone did not
authorize it. The finite implementation below does not complete R04/R05,
FTS-10/11 or FTS-02/09/R01-R15.


### FTS-11C finite trajectory Born-conformance implementation

The user separately approved #884 I-L on 2026-10-26; #884 actual main fee4ad and
entry-pool prerequisite #894 actual main a6ec8e completed full CI, required
single/two-GPU, fresh installation and executable user-example acceptance.
The private finite conformance module now observes existing E0/E1/E2 trajectories
and reuses E-H references: six natural, ten controlled-positive and three stable
calls, totaling nineteen calls/eighty-two trajectories with seed 53 and one CPU
thread restored. Raw float64 Born vectors and complex128 conditional density/norm
use the specifically approved atol=1e-12, rtol=0. Controlled branches are not
random observations; no frequency/RNG or general sampling certificate is claimed.
Implementation PR #898 and its actual-main admission are complete, as recorded
in the FTS-10C entry below. R04/R05, FTS-10/11 selection/product sampling,
FTS-02/09 and R01-R15 remain partial; no public API, engine, approximate policy
or additional research budget is activated.

### FTS-10C phase/overlap feasibility decision

I-L PR #898 completed actual-main admission at
`44fb9c6a6779a746780b5b96fcaaa2cb99ce7f2d`: full CI all 54 jobs, required
source/single/two GPU, main smoke, real QEC optional installation/evidence,
fresh isolated 838-member package/core and installed public examples passed.
Source-focused acceptance passed 118 tests with zero skips, using developer
Torch 2.13 and an existing SDK overlay; fresh core used Torch 2.14.1 without
that overlay. This closes only finite I-L correctness under the prior approval.

The next [phase/overlap feasibility proposal](../development/FTQC_PHASE_OVERLAP_FEASIBILITY_PROPOSAL.md)
reviews the missing compact complex-phase representation and pre-selection
evidence. Its M-P request is limited to a private Simulation research prototype,
64 candidate states, 850 complex-overlap queries and ten existing public exact
executions per module verification, with proposed complex128/float64
atol=1e-12/rtol=0. The human separately approved this finite M-P scope; #922 delivered it after
#911 actual-main admission. Prior A-D/E-H/I-L approval, document
merge and green CI did not admit it. No collection, production engine,
public phase guarantee or method selection follows. FTS-02/09/10/11 and R01-R15
remain partial. Independent QEC #904 and remote normalization work are reused
under their own contracts, not claimed as this proposal's implementation.

The separately approved M-P implementation now supplies one private CH-mask
prototype and finite conformance: 64 retained states, 850 captured overlaps,
ten existing public calls, 41 new checks. Combined local contracts passed 159
checks with zero skips (developer Torch 2.13/existing SDK overlay). The compact
state and inverse-word overlap replay have no production route, export,
dependency, dense fallback, sample or collection. Actual-main admission at
`82143b09dea21bf565c51a2e7f3f3c1496ec9dbf` passed all 54 CI jobs, required
source/single/two GPU, smoke, real QEC optional installation/evidence, fresh
845-member isolated package/core and installed public examples. Actual-main
focused contracts passed 159 checks with zero skips on the developer profile;
fresh core used Python 3.12.14/Torch 2.14.1 without an SDK overlay. This is finite
R04 correctness evidence; FTS-10 selection and production integration remain open.

### FTS-10D bounded phase/overlap cost decision

The [cost-study proposal](../development/FTQC_PHASE_OVERLAP_COST_STUDY_PROPOSAL.md)
specifies a separately approved observation boundary after M-P correctness admission.
It preserves the ten existing one-to-three-qubit descriptions and compares the
private candidate with statevector/MPS/tensor-network full-probability outputs.
Proposed maxima are 40 points, 160 workload starts plus one new monitor probe,
416 state constructions, 3400 overlaps and 120 public executions, with serial
one-thread CPU work and explicit phase/outer-wall/RSS/failure-stop provenance.
The finite matrix, numerical values, observer boundary and resource protocol
were specifically approved on 2026-10-26. Proposal #931 actual-main admission
passed full CI, required GPUs, smoke, real optional installations/evidence,
fresh isolated 847-member package/core and installed examples. A private worker
and serial controller are under separate review; no probe or observation has
run. The old 193 attempts and their failure remain immutable. Observer/controller
review and actual-main admission precede the one admitted collection. Production
selection and a larger domain remain unapproved. R01-R15, FTS-02/09/10/11 and
M2 remain partial.
