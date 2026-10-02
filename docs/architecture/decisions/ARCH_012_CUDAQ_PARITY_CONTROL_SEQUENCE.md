# ARCH-012: CUDA-Q parity control sequence and NVIDIA-proprietary backend replacement

Status: Approved
Date: 2026-10-06
Scope: program-level control sequence and dependency policy. No Stable Core change.
Supersedes in part: the binding control sequence in `AGENTS.md` § Current Strategic Priority.

## Context

FlagQuantum's product requirement now includes functional parity with CUDA-Q as a
floor, not a ceiling: every capability CUDA-Q exposes must exist in FlagQuantum,
and FlagQuantum must additionally hold capabilities CUDA-Q structurally lacks. The
requirement also carries an ownership constraint: the parity path must not depend
on NVIDIA-proprietary components.

Three facts make this requirement harder to schedule than to state.

**First, it conflicts with the current binding control sequence.** `AGENTS.md`
§ Current Strategic Priority states that the immediate vNext sequence is freeze
new horizontal abstractions, finish the Compiler boundary, deliver the smallest
complete CPU vertical path, move code into authoritative domains, and delete or
freeze legacy code, and that additional horizontal architecture tracks must not be
opened while the CPU path is incomplete. A parity program spanning statevector,
tensor-network, and dynamics numerical cores, richer internal IR levels,
compilation infrastructure, error correction, fault-tolerant layers, operator
algebra, and realtime control is, read literally, several horizontal tracks at
once. Without an explicit decision the program cannot proceed, and proceeding
silently would make `AGENTS.md` unreliable as a governing document.

**Second, most of the required architecture is already designed.** Every internal
capability this program needs is already specified in
[`MULTI_LEVEL_IR_ARCHITECTURE.md`](../MULTI_LEVEL_IR_ARCHITECTURE.md): four IR
levels with `CircuitIR` held public and frozen at schema `1.0`, a pass
infrastructure with declared preservation and analysis invalidation, an
`ExecutableArtifact`/`RuntimeAdapter` boundary, versioned `TargetCapabilities`,
and a phased programme whose Phase 4 already covers functions, scopes, blocks,
classical control, measurement def-use, reset, conditionals, bounded loops,
partial evaluation, `ProgramIR` lowering, and dynamic migration, and whose Phase 5
already covers QIR Base/Adaptive, timing, scheduling, gradient and distributed
lowering, and evidence-based native evaluation. ADR `IR-001` through `IR-006` are
approved. The design also already fixes where the internal levels live:
§ 19.13 places them under `flagquantum/_compiler/{ir,analyses,passes,pipelines,
targets,emitters,diagnostics}/`, which `team-ownership.toml` already assigns to
the compiler team, and § 19.13 marks `ProgramModule`, `QuantumModule`, and
`TargetModule` as internal with no planned root exports. The parity program
therefore does not need a new architecture; it needs the existing one scheduled,
authorized, and implemented.

**Third, the obstacle is narrower than "replace CUDA".** Splitting CUDA-Q's
dependencies into three classes shows that only one class is a CUDA replacement
obligation.

| Class | Components | Obligation |
| --- | --- | --- |
| Proprietary | `cuStateVec`, `cuTensorNet`, `cuDensityMat`, `cuTensor`, CUDA runtime, `cuRAND`, NVRTC, `cuBLAS`, `cuSolver`, NCCL/NVSHMEM, NVQLink / Holoscan Sensor Bridge / ConnectX | Must be replaced |
| Vendor-neutral | LLVM/MLIR, QIR specification, Q++, Eigen, OpenMP, nlohmann/json, Ensmallen, NLopt, GMP/MPFR, Stim | May be used |
| FlagOS-covered | FlagTree, libtriton_jit, Torch-FL, FlagGems, FlagBLAS, FlagFFT, FlagCX | Already replaced |

CUDA-Q's framework layer does not itself require CUDA: it ships pure-CPU
`qpp-cpu`, `density-matrix-cpu`, and `stim` targets, and its simulator plugin
mechanism is an open C++ extension point. The hardware-independent parts of the
framework are reachable without any NVIDIA component.

## Decision

### 1. Replace the control sequence with a parity-capable one

The binding control sequence in `AGENTS.md` is replaced by the following. The
original sequence's intent — small rounds, vertical slices before breadth,
aggressive subtraction — is preserved; what changes is that a second concurrent
track is permitted under stated conditions instead of forbidden outright.

1. Every round extends a proven vertical path through input, validation,
   planning, execution, result, failure, and evidence. Breadth is earned by
   completing such a path, not by declaring an abstraction.
2. A new horizontal abstraction is admitted only when it **replaces** an existing
   implementation behind an existing boundary. Admission requires a replacement
   test in which at least one implementation is swapped without modifying its
   consumers, per engineering decision principle 10.
3. Two tracks may develop concurrently only when they do not share an unproven
   contract. Where they would, the contract lands first on the integration
   branch together with a contract fake and a conformance test, and each track
   then synchronizes that baseline.
4. Legacy and transitional code is deleted or explicitly frozen in every round.
   This clause is unchanged from the original sequence.
5. No parity claim is made without a generated parity-matrix entry and the
   evidence its maturity level requires. A capability present in CUDA-Q and
   absent here must appear as an explicit, owned gap rather than as silence.

Clauses 2 and 3 are what make the programme schedulable: the three numerical
cores below are admitted as replacements, and the internal IR levels are
admitted as an already-designed contract rather than as new scaffolding.

### 2. Deliver parity by sequencing the existing designed work

Parity is delivered by activating the already-specified internal levels and
phases, not by inventing a parallel architecture:

- **Internal IR levels.** `ProgramIR`, `QuantumIR`, and `TargetIR` are
  implemented as internal levels behind the frozen public `CircuitIR` schema
  `1.0`, in the layout the approved design already fixes rather than in a new
  location. `IR_VERSION` is not changed and no new public IR schema is
  introduced. Phase 4 scope is ratified by its own IR ADR; this ADR only requires
  that the levels are implemented before the waves that depend on them.
- **Compilation infrastructure.** The pass contracts, pipeline, analysis
  invalidation, and `ExecutableArtifact`/`RuntimeAdapter` boundaries of
  `MULTI_LEVEL_IR_ARCHITECTURE.md` § 6 and § 7 are the pass and artifact
  infrastructure. No second pass manager is created, and the existing
  `CompilerPassExtension` in the extension SDK is wired to it rather than
  duplicated.
- **Native evaluation.** Native or MLIR-based evaluation remains the
  evidence-gated Phase 5 question it already is. This ADR does not select a
  compiler implementation.

### 3. Own exactly three numerical cores

Three capabilities have no acceptable vendor-neutral substitute and are therefore
owned by FlagQuantum. Each is admitted as a **replacement** of an existing
backend implementation under clause 2, not as a new product surface:

- **Statevector core**, replacing `cuStateVec`-class device statevector
  execution, including multi-rank sharding.
- **Tensor-network core**, replacing `cuTensorNet`-class contraction and
  matrix-product-state execution.
- **Dynamics core**, replacing `cuDensityMat`-class time evolution. This one is a
  maturation of the existing `flagquantum/simulation/lindblad.py` engine rather
  than a new implementation; its current CPU-only, dense, static-Hamiltonian
  limits are recorded in `docs/reference/KNOWN_LIMITATIONS.md` and are lifted
  under their own evidence.

All three reach hardware through the Compute layer and FlagOS. Numerical
implementation stays in Simulation; device and operator selection stays in
Compute; planning and one execution attempt stay in Runtime.

### 4. Vendor-neutral dependencies are permitted under existing policy

Using LLVM/MLIR, the QIR specification, Stim, or comparable vendor-neutral
projects does not violate the ownership constraint, because none of them is an
NVIDIA asset. Each adoption remains subject to engineering decision principle 6:
documented need, ownership boundary, license and supply-chain review, replacement
interface, and exit plan. Evaluation algorithms and third-party numerical
libraries are referenced where they are authoritative rather than reimplemented.

Proprietary simulation cores are not adopted, not wrapped, and not shipped. CUDA-Q
interoperability remains limited to the one-way export already documented in
`docs/development/CUDAQ_EXPORT_CONTRACT.md` and is not a parity mechanism.

### 5. Parity is an evidence artifact, not a claim

A generated parity matrix becomes the single source of truth for "what is still
missing". Each row names the CUDA-Q capability, the FlagQuantum counterpart or an
explicit absence, the owning team, the maturity level, and the evidence artifact.
Generation and drift checking are machine gates. A capability may not be moved to
a higher maturity level to make the matrix look closer to parity.

## Consequences

The CPU vertical path continues to be delivered and keeps first-class status;
this ADR does not authorize slowing local users down or deprioritizing CPU
correctness. What changes is that a second track may run beside it under clause 3,
with the parity matrix making the resulting breadth visible and accountable.

Concurrent development increases integration frequency and therefore merge and
conformance cost. The contract-first rule in clause 3 is the mitigation, and it
is a precondition rather than a preference.

The programme is long. Reaching parity across the simulator matrix, compilation
infrastructure, error correction, and fault-tolerant layers is a multi-quarter
effort measured in over a hundred engineer-months. That cost is a property of the
requirement, not of this decision, and stating it is part of keeping the sequence
honest.

Two risks are accepted explicitly. First, native compilation infrastructure
remains the largest uncertainty; the design already defers the native-versus-MLIR
choice to an evidence gate, and this ADR adds no commitment ahead of that gate.
Second, realtime control requires a hardware partner and cannot be delivered by
software alone; the software-side messaging and device-call abstraction can be
owned here, and the remainder must be scoped with that partner.

## Rejected alternatives

- **Keep the current sequence and defer parity until the CPU path is complete.**
  Preserves the sequence's letter but leaves the product requirement unowned and
  unscheduled for at least a quarter. Rejected because the requirement is real and
  the sequence is a means, not an end.
- **Proceed without amending the sequence.** Rejected: it would make `AGENTS.md`
  inaccurate and would let the same question be re-litigated every round.
- **Abandon the sequence entirely and run parity as the sole programme.** Rejected:
  it discards the subtraction discipline that keeps the repository coherent, and
  it would allow new abstractions to accumulate without replacement tests.
- **Depend on CUDA-Q as an execution backend to close gaps.** Rejected as a parity
  mechanism: it makes CUDA-Q capabilities rented rather than owned, and it
  contradicts the ownership constraint. It remains permitted only as the existing
  documented one-way export for interoperability evidence.
- **Write a new IR from scratch for parity.** Rejected: the internal levels are
  already designed and partially approved, and a new IR would be exactly the
  parallel scaffolding the original sequence prohibited.

## Compatibility and rollback

This ADR changes governance text and dependency policy only. It changes no public
API, no serialized schema, and no Stable Core surface. `IR_VERSION` remains `1.0`.

Rollback is textual: reverting this ADR and the corresponding `AGENTS.md` section
restores the previous control sequence. Work admitted under clause 2 remains
governed by its own replacement tests and evidence, and no capability claim
depends on this ADR's continued force.

## Acceptance

An unchecked item is a condition this ADR requires but does not itself complete;
each one names the separate decision it is waiting on.

- [x] `AGENTS.md` § Current Strategic Priority states the replacement sequence.
      The section now states the five clauses of Decision 1 verbatim in substance,
      names this ADR as their justification, and states that the parity path must
      not depend on NVIDIA-proprietary components.
- [x] This ADR is listed in `docs/architecture/decisions/README.md` with its status.
- [ ] Phase 4 scope is ratified by an IR ADR following the `IR-006` precedent.
      `IR_007_PHASE_4_HYBRID_DYNAMIC_SCOPE.md` is drafted and registered, and stands
      at **Proposed**. Ratifying it is a separate owner decision; this ADR does not
      grant it.
- [x] Each of the three owned numerical cores has a named owner, a replacement
      test, and a maturity entry in `capability-maturity.toml`.
      - Statevector: `sharded_statevector_training`, owner
        `PyTorch distributed maintainers`, level `production_supported`. Replacement
        test: `tests/unit/test_layout_bmm_device_axis.py`, where
        `test_a_declared_device_and_precision_select_the_fused_route` and
        `test_the_same_call_without_the_declaration_stays_on_the_eager_route` issue
        the same call and reach two different implementations because a catalog
        record changed and the consumer did not.
      - Tensor network: `tensor_network_training`, owner
        `tensor-network maintainers`, level `experimental`. Replacement test:
        `tests/test_tensor_network.py::test_cotengra_slicing_plan_executes_external_path_with_autograd`,
        where the native planner is swapped for the external cotengra planner behind
        the unchanged `plan.contract_slicing_plan` consumer.
      - Dynamics: `continuous_time_lindblad`, owner `simulation maintainers`, level
        `production_supported`. Replacement test:
        `tests/unit/test_lindblad_integrators.py`, where
        `test_the_default_integrator_remains_the_explicit_scheme` and
        `test_the_reported_method_is_the_method_that_ran` reach two different
        integration schemes through the unchanged `evolve_density_matrix` consumer.
- [x] The parity matrix is generated by a machine gate and its drift is checked in
      CI. `tools/parity_matrix.py` owns generation and `--check`, and
      `.github/workflows/ci.yml` runs `python tools/parity_matrix.py --check` in the
      quality job.
- [x] No Stable Core API, serialized public schema, or `IR_VERSION` changes as a
      result of this ADR. `IR_VERSION` reads `1.0` and `docs/public_api_v1.json` is
      unchanged against the revision this ADR landed on.
- [x] Repository owner approves the amended control sequence.

## Approval record

Approved by the repository owner, who authorized amending the control sequence in
Decision 1 and the dependency policy in Decisions 3 and 4, and authorized replacing
the `AGENTS.md` § Current Strategic Priority text this ADR supersedes in part.

Approval does not by itself authorize any Stable Core change, promote any capability
maturity level, or complete any Multi-Level IR phase gate. The acceptance item that
remains unchecked — Phase 4 scope ratification — is one of those separate decisions
and is not granted here.
