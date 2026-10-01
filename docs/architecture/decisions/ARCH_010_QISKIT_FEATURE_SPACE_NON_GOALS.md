# ARCH-010: Deliberate non-goals against the Qiskit feature space

Status: Proposed
Date: 2026-09-30
Scope: planning and boundary decisions only; no Stable Core, implementation, or capability-level change

## Context

FlagQuantum is frequently compared with Qiskit, and the comparison is load-bearing: it sets what
contributors propose, what reviewers accept, and what users expect. Without a recorded floor, every
Qiskit class that FlagQuantum does not implement reads as unfinished work. That framing is wrong in
both directions. It understates surfaces that exist and will never be extended, and it turns product
choices into backlogs.

The repository already contradicts itself in small ways, because the boundary has been decided
repeatedly and locally rather than once and in writing:

- [`docs/roadmap/QUANTUM_KERNEL_ARCHITECTURE_PLAN.md`](../../roadmap/QUANTUM_KERNEL_ARCHITECTURE_PLAN.md)
  has a `## 4. Non-Goals` section;
- [`docs/roadmap/PYTHON_HYBRID_COMPILATION_EXECUTION_PLAN.md`](../../roadmap/PYTHON_HYBRID_COMPILATION_EXECUTION_PLAN.md)
  has `## 3. Explicit non-goals`, which states "QIR, pulse, timing, or controller firmware generation"
  is excluded;
- [`docs/roadmap/ECOSYSTEM_DEVELOPMENT.md`](../../roadmap/ECOSYSTEM_DEVELOPMENT.md) has
  `## Deliberate Non-Goals For The Current Stage`;
- [`docs/roadmap/DOMESTIC_ACCELERATOR_AND_NUMERICAL_TRUST_PLAN.md`](../../roadmap/DOMESTIC_ACCELERATOR_AND_NUMERICAL_TRUST_PLAN.md)
  has `## 3. Goals and Non-Goals`;
- [`contracts/interop-capability-gap-matrix.toml`](../../../contracts/interop-capability-gap-matrix.toml)
  already encodes the outcome for ten interop capabilities per framework with an `out_of_scope` status
  and a mandatory `reason`.

None of those is the boundary document. A reader asking "will FlagQuantum ever have pulses?" has to
find four different sections and a TOML contract to assemble an answer, and the answer they assemble
may not be the one the project intends.

This ADR is that answer. It is a decision, not a status report: it does not measure the current gap and
it does not schedule work. It records what is refused, why, and what would have to change for a refusal
to be revisited.

## Decision

### What a non-goal is here

An item is a non-goal only if all three of these hold. Anything failing one of them is a gap, and gaps
are scheduled elsewhere. This distinction is the whole point of the record, so it is defined before the
list rather than inferred from it.

1. **No commitment exists and none is planned at any maturity level.** Not "L3, so deferred" — absent
   from the roadmap, not merely unprioritised.
2. **A stated reason survives the work being cheap.** Cost alone never makes something a non-goal, and
   a non-goal is not revisited just because it became easier. A version that says "we lack the
   bandwidth" is a deferral and must be recorded as a gap instead.
3. **Adding it later would be a new decision with a named approver**, not the completion of an existing
   one.

Under those tests the following are non-goals.

| Qiskit surface | Decision | Reason | Where this is already encoded |
| --- | --- | --- | --- |
| Rust core of `qiskit/` | Never reimplemented in Rust | The differentiator is PyTorch-native differentiability and distributed training, not single-thread speed. A second core in another language would have to be crossed by every autograd path, which is the surface that must stay cheap. | `docs/roadmap/QUANTUM_KERNEL_ARCHITECTURE_PLAN.md` non-goals ("merge the FlagQuantum circuit compiler with the FlagTree kernel compiler"); [ARCH-009](ARCH_009_NATIVE_CPU_OPERATOR_BOUNDARY.md) chose a dispatcher-registered PyTorch extension inside Simulation instead of a parallel core |
| `pulse/`, `scheduler/`, pulse visualization, and the pulse/stretch methods on `QuantumCircuit` (`add_stretch`, `add_capture`, `get_stretch`, `has_stretch`, `iter_stretches`, `use_stretch`, `qubit_duration`, `qubit_start_time`, `qubit_stop_time`, `estimate_duration`, `estimate_fidelity`) | Never added | Pulse formats are vendor-specific and change with hardware revisions, so supporting the layer would trade a portable IR for a per-vendor ABI. FlagQuantum's route is target legalisation plus text emission onto an explicit backend snapshot, and its scheduling is logical layers. A calibration- and timing-aware compiler is a different product. | `flagquantum/compiler/IMPLEMENTATION.md`: scheduling "is not target timing or pulse scheduling"; `docs/roadmap/PYTHON_HYBRID_COMPILATION_EXECUTION_PLAN.md` excludes "QIR, pulse, timing, or controller firmware generation"; `flagquantum/compiler/noise.py` owns the only noise transformation, and it is model-based rather than calibrated |
| `qiskit-ibm-runtime` and the retired IBM Quantum backend names | Not connected | An active product choice, not missing capability. It is listed because it must be revisitable without redesign, and the provider abstraction exists for that reason: deciding against one provider is only meaningful when the provider boundary is real. | [`contracts/quafu-provider-contract-v1-candidate.json`](../../../contracts/quafu-provider-contract-v1-candidate.json) records the admitted provider; ARCH-001 and [ARCH-006](ARCH_006_PROVIDER_ADMISSION_MODEL.md) define admission by control boundary rather than by vendor |
| Equivalents of `qiskit-nature`, `qiskit-finance`, and `qiskit-optimization` | Never added as in-repository packages | These are application domains, not framework capabilities. [`docs/roadmap/FLAGQUANTUM_VISION.md`](../../roadmap/FLAGQUANTUM_VISION.md) positions the project as "foundational infrastructure for quantum AI" rather than as an application suite, and the analogue that is tracked is machine learning. A domain package added to this repository would bring its own APIs, dependencies, and release cadence under the framework's compatibility promise. | `flagquantum/algorithms/` and [`docs/guides/ALGORITHMS.md`](../../guides/ALGORITHMS.md) are titled "demonstration scale" on purpose |
| The v1 primitive compatibility surface (`base_primitive_v1.py` and the v1 pub, binding, and result semantics) | Never added | Qiskit deprecated v1 in favour of v2, so implementing it would add a compatibility obligation inherited from a decision its own owner reversed. FlagQuantum has no installed base holding v1 objects, which is the only reason a compatibility surface exists at all. The v2 object model is a gap and is tracked separately. | `flagquantum/observables/` (`OutputRequest`) and `fq.run(outputs=...)` already cover the execution request model without a v1 shape; no `Estimator` or `Sampler` identifier exists anywhere in the package |
| `Qobj`, `QIR`, and `Quil` program import | Never added | `Qobj` is retired and the other two have negligible adoption next to OpenQASM, so the formats would be carried without a user. Emission formats the project does support are owned deliberately and one file each. | `flagquantum/compiler/openqasm.py` and `flagquantum/compiler/qcis.py` own their target formats; `contracts/legacy-root-api-test-debt.json` tracks the legacy surface that is being retired rather than extended |
| The 75-file template library behind template matching | Not added as a library | The templates serve one pass, and the maintenance cost is a file per template against a benefit that no tracked user goal requires. Template matching as a pass remains a gap; shipping the corpus is not the same decision. | `flagquantum/compiler/` has no template corpus; the pass framework it would need is recorded as a gap elsewhere |
| The complete QEC stack | Not added beyond the current minimal set | Quantum error correction is a research programme with its own abstractions, decoders, and hardware assumptions. A minimal, self-contained set is supported so that error-correction results are reproducible inside the framework; growing it into the ecosystem-scale stack of a dedicated project is not planned here. | `flagquantum/qec/` (codes, decoders, detectors, Pauli, repetition) and `flagquantum/lindblad/` are the minimal set; `capability-maturity.toml` states each one's scope |

The repository-language rule (English in-repository) is not a Qiskit feature decision. It is owned by
`AGENTS.md` and enforced by `tools/check_repository_language.py`, and this ADR neither restates nor
extends it.

### How a non-goal is revisited

A non-goal is reversed only by a superseding ADR that states the new reason, not by a pull request that
implements the surface, and not by a maturity-level change. The three tests above are the review
question: if the new proposal cannot say why an old reason stopped holding, the answer is that the
proposal is a gap that was mislabelled.

Two of the entries above deserve an explicit warning against erosion.

**Pulse is the most likely to be re-proposed**, because a target-aware compiler invites the next step.
The boundary is the artefact: FlagQuantum emits a program for a target and stops. It does not own
timing, calibration, or waveforms, and the first pulse-shaped pull request is the one that should be
refused on this ADR rather than on taste.

**The provider choice is the most likely to be misread as a gap.** "FlagQuantum cannot run on IBM
hardware" and "FlagQuantum does not connect to IBM Quantum" are different statements, and only the
second is true. Wording in issues, guides, and capability entries must not blur them.

## Prohibited practices

- Describing a non-goal as "not yet" or "not currently" supported. Either it is scheduled as a gap or it
  is refused here; an unqualified "not yet" is the ambiguity this ADR exists to remove.
- Listing a surface in a roadmap, backlog, or issue as a candidate after it appears in the table above
  without a superseding ADR.
- Extending a non-goal by adjacent reasoning, such as adding one pulse method because the compiler
  already computes layers, or adding a domain package because `flagquantum/algorithms/` already ships
  demonstrations.
- Treating the `out_of_scope` value in `contracts/interop-capability-gap-matrix.toml` as equivalent to
  the decisions here. That contract governs one adapter boundary; this ADR governs the project. A row
  may change status without changing this record, and vice versa.
- Citing this ADR to refuse a gap. The refusals are enumerated; nothing outside the table is covered.

## Compatibility

This ADR changes no code, no public API, no capability level, and no contract. Every cited artefact
remains authoritative for its own scope, and where a contract and this table disagree the contract wins
for that boundary while the disagreement is reported.

Because nothing here is implemented, the decision is reversible in the only way that matters: a
superseding ADR costs one document. What is not cheaply reversible is a partially implemented
non-goal, which is why the list is written before the work rather than after it.

## Acceptance tests

This ADR is satisfied by review, not by a machine check. The properties a reviewer verifies are:

1. Every entry in the table fails all three non-goal tests, and no entry can be satisfied by a
   deferral reading. An entry whose reason would evaporate if a person-month became free does not
   belong in the table.
2. Every entry names an artefact that already exists in the repository or a superseding ADR. An entry
   with only a prose reason is an assertion, not a record.
3. No entry in the table is also scheduled in the compiler, circuit, operator, or visualisation
   workstreams. An item in both places is a defect in one of them.
4. Each cited path resolves.

Existing checks that remain relevant and unchanged: `tools/check_capability_maturity.py` and
`tools/docs_source_of_truth.py --check` for the maturity matrix,
`tools/check_interop_capability_gap_matrix.py` for the per-adapter `out_of_scope` reasons, and
`tools/check_architecture.py` for layer boundaries. None of them is extended by this decision.

## Open Questions

1. **Should the non-goal list be machine-readable?** The three tests are prose, and test 3 — that no
   entry is also scheduled — is exactly the kind of property a check could hold. It is not proposed
   here because the schedule lives outside the repository, so a check would have nothing in-repository
   to compare against. If a roadmap format is added later, this ADR should be revisited to point at it.
2. **Who approves a superseding ADR?** The table mixes a product choice (IBM Quantum), an architecture
   choice (the Rust core), and a scope choice (application domains). Whether these need one approver or
   three is unresolved, and the answer determines whether a reversal is a maintainer decision or a
   roadmap decision.
3. **Does the minimal QEC boundary need a number?** "The complete stack" is refused and "the current
   minimal set" is supported, but the line between them is a judgement each time. A quantitative
   boundary, such as a named list of codes and decoders, would make the refusal checkable.
4. **Should the pulse refusal name the replacement path?** A user who needs pulse-level control has no
   route inside FlagQuantum. Whether the answer is an external tool, a documented hand-off format, or
   nothing at all is a product question this ADR leaves open.
5. **Is the Rust refusal time-bound?** Reason 2 forbids revisiting a non-goal because it became cheap,
   but a hardware or toolchain change could make the differentiation argument itself stop holding. The
   tests cover new reasons, not changed premises.
