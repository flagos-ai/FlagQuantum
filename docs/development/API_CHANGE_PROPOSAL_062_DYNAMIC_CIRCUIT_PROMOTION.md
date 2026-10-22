# API Change Proposal 062: Dynamic circuit promotion criteria

## Status

**Readiness audited on 2026-10-22. Criteria 1–7 are satisfied; criterion 8 and
final API-owner authorization remain open. `DynamicCircuit` is not promoted by
this audit.**

This proposal defines the criteria for promoting the candidate-stable
`DynamicCircuit` to stable and for leaving `run_dynamic` experimental. It
supersedes the promotion expectations left open by
[`API_CHANGE_PROPOSAL_010`](API_CHANGE_PROPOSAL_010_DYNAMIC_CIRCUIT.md).

No public API is added, removed, or renamed here. The root API is unchanged.

## Problem

`API_CHANGE_PROPOSAL_010` froze a deliberately small candidate-stable surface:
the namespace `flagquantum.dynamic`, the single name `DynamicCircuit`, a
machine-readable contract at `contracts/dynamic-circuit-v1-candidate.json`, and
nothing else. `run_dynamic`, providers, dialects, and native results were left
experimental. That decision was correct at the time and it recorded no criteria
for what would have to become true for the candidate to graduate.

The absence of criteria has two costs. Users cannot tell how much weight the
candidate-stable label carries, and the engineering work needed to graduate the
surface is not scheduled anywhere. In practice the dynamic path is also the place
where the missing internal IR levels bite hardest:
`docs/development/CUDAQ_EXPORT_CONTRACT.md` records that CUDA-Q kernels perform
classical computation and control flow, and that converting such kernels "would
either lose semantics or require FlagQuantum IR features that do not exist". The
same structural gap is what keeps `DynamicCircuit` from expressing
measurement-dependent control with verified def-use.

This proposal is therefore the public-surface counterpart of Phase 4. Phase 4
scope is ratified by `IR-007`; this proposal governs only the label the public
namespace carries and the conditions under which it may change.

## Decision

### 1. Promotion is criteria-gated, not schedule-gated

`DynamicCircuit` moves from candidate-stable to stable only when every criterion
in the table below holds with recorded evidence. Passing a milestone date, or
implementing a related internal feature, does not by itself promote the surface.

### 2. The criteria

| # | Criterion | What must be demonstrated |
| --- | --- | --- |
| 1 | Semantics | Mid-circuit measurement, reset, and conditional execution have specified and tested semantics, including the behaviour of repeated measurement and reset on the same wire. |
| 2 | Verifier | Measurement-to-condition def-use is explicit and machine-checked. A condition on a measurement that has not occurred is rejected, not defaulted. |
| 3 | Determinism | A fixed dynamic circuit with a fixed seed replays to identical results, across repeated runs in one process and across processes. |
| 4 | Round trip | The dynamic structure survives a round trip through the internal program level with conditions, ordering, and wire identity preserved, or it is rejected with diagnostics. Silent structural loss disqualifies the criterion. |
| 5 | Distribution | Sharded execution either preserves feedback semantics, or fails closed with a reported blocker naming the unsupported case. Replicated per-rank execution must not be described as distributed scalability. |
| 6 | Degradation | A target without conditional execution is rejected at planning time with an actionable blocker. No silent flattening to a static circuit, and no substitution of another backend. |
| 7 | Non-regression | Static batched execution shows no regression against the recorded baseline. |
| 8 | Contract | `contracts/dynamic-circuit-v1-candidate.json` is superseded by a stable contract with a recorded migration path, and the candidate contract is retired rather than left to coexist indefinitely. |

Criterion 7 is satisfied for the declared Apple-arm64 CPU profile at source
revision `4972fd0ce9b03a809cca78dc298810bb45a3c5c0`. The profile-aware gate passed
both maintained 18-qubit, batch-32, complex128 workloads with exact-statevector
correctness and stable timing. Current/baseline time was `0.823x` for local
brickwork and `0.909x` for Random Clifford; current/baseline peak RSS was
`0.488x` and `0.509x`, respectively. The policy permits at most `1.20x` time
and `1.10x` peak-RSS growth and requires at least five timing samples and three
fresh-process memory probes.

The machine-readable
[`gate verdict`](../../benchmarks/results/comparison/batched_statevector_dynamic_promotion_gate_cpu_arm64_20261022.json)
and its
[`raw observations`](../../benchmarks/results/comparison/batched_statevector_dynamic_promotion_current_cpu_arm64_20261022.json)
are comparison-only evidence: they make no release, scalability, or dynamic
execution performance claim. Reproduce them with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
flagquantum-benchmark run batched_statevector_memory \
  --workloads random_clifford_statevector local_brickwork_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch flagquantum_native_layout_retention \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output candidate-batched-memory.json

flagquantum-benchmark run cpu_performance_gate \
  benchmarks/results/comparison/batched_statevector_layout_lifetime_cpu_arm64_20261001.json \
  candidate-batched-memory.json \
  --max-slowdown 1.20 --max-memory-growth 1.10 \
  --minimum-samples 5 --minimum-memory-probes 3 \
  --json-output candidate-batched-memory-gate.json
```

### 2.1 Readiness audit

The audit is deliberately narrower than promotion. It establishes that the
implementation and evidence satisfy criteria 1–7, but it does not replace the
candidate contract, change an API label, or authorize stable publication.

| Criterion | Verdict | Executable evidence |
| --- | --- | --- |
| 1. Semantics | Satisfied | `tests/test_dynamic_circuit.py::test_mid_circuit_measurement_collapses_and_controls_a_later_gate`, `::test_reset_returns_measured_qubit_to_zero`, and `::test_repeated_measurement_and_reset_on_one_qubit_have_defined_semantics`; `tests/test_dynamic_conformance.py::test_local_dynamic_conformance_vectors_pass` |
| 2. Verifier | Satisfied | `tests/unit/test_classical_control_contract.py::test_classical_control_def_use_is_validated_at_build_time` and `tests/test_dynamic_circuit.py::test_dynamic_execution_rejects_unmeasured_classical_reads_and_gradients` prove rejection both while building and when an externally restored IR bypasses the builder. |
| 3. Determinism | Satisfied | `tests/test_dynamic_circuit.py::test_seeded_dynamic_execution_is_reproducible_in_one_process` repeats a seeded local run directly, and `::test_seeded_dynamic_execution_is_reproducible_across_processes` compares two fresh processes. Neither proof depends on an optional framework. |
| 4. Round trip | Satisfied | `tests/unit/test_multi_level_ir_contract.py::test_dynamic_circuit_lowers_and_round_trips_exactly`, `::test_remeasuring_a_classical_slot_rebinds_subsequent_conditions`, and `::test_a_live_but_incorrect_dynamic_condition_fails_exact_round_trip` cover exact Program IR lowering, overwrite def-use, and fail-closed graph/payload mismatch. |
| 5. Distribution | Satisfied by explicit refusal | `tests/unit/test_classical_control_contract.py::test_distributed_execution_refuses_every_classical_control_key` proves refusal occurs before compilation and reports `dynamic_conditional_execution_unsupported`; no replicated execution is presented as scalability. |
| 6. Degradation | Satisfied | `tests/test_dynamic_circuit.py::test_dynamic_backend_capability_negotiation_fails_closed` verifies actionable blockers, while `::test_dynamic_deployment_preflight_blocks_provider_submission` proves an incompatible program never reaches `provider.run`. |
| 7. Non-regression | Satisfied on the declared profile | The raw observations, gate verdict, policy, source revision, and reproduction commands are recorded immediately above. |
| 8. Contract | Open | `contracts/dynamic-circuit-v1-candidate.json` is still the active frozen contract and `tests/unit/test_dynamic_circuit_candidate.py` still enforces candidate status. Promotion requires one separately approved change that installs the stable contract and retires this candidate rather than retaining both. |

Run the functional portion of the audit with:

```bash
python -m pytest -q \
  tests/test_dynamic_circuit.py \
  tests/test_dynamic_conformance.py \
  tests/unit/test_classical_control_contract.py \
  tests/unit/test_dynamic_circuit_candidate.py \
  tests/unit/test_multi_level_ir_contract.py
```

The audit decision is therefore **ready to enter criterion 8**, not “already
stable”. If any cited test is removed, weakened, or stops passing, readiness is
withdrawn until an equivalent recorded proof replaces it.

### 3. `run_dynamic` and the rest stay experimental

`run_dynamic`, providers, dialects, and native results remain experimental
regardless of the outcome for `DynamicCircuit`, and this proposal does not
schedule their promotion. A mixed label is intentional: the construction contract
is the part users build against, and the execution surfaces still move.

### 4. The candidate contract is retired, not stacked

Criterion 8 exists to prevent permanent compatibility debt, which engineering
decision principle 1 prohibits. When the stable contract lands, the candidate
contract has a named owner, a documented scope, a removal condition, and a target
version. Two coexisting dynamic-circuit contracts is not an acceptable end state.

## Public API

No change is requested by this proposal. The surface under assessment is:

```python
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(3)
circuit.h(0)
circuit.measure(0)
circuit.conditional("x", 1, classical_bit=0)
```

The namespace already exists and is candidate-stable. Promotion changes the
label and the contract that backs it; it does not add names, change signatures,
change defaults, or alter documented exception behaviour.

If a later step concludes that a promotion requires a public surface change — a
new method, a condition expression, or a result field — that change is proposed
separately and is not covered by this proposal.

One of the three examples above has since happened, and it did not come from a
promotion step. `DynamicCircuit.conditional` gained an `any_of` keyword on
2026-10-20 so that a disjunction of classical condition clauses has a public
spelling, which is a condition expression in the sense of the sentence above. It
was taken as the separate change that sentence asks for, under explicit API-owner
approval, and the additive, defaulted, keyword-only scope is recorded in
`contracts/dynamic-circuit-v1-candidate.json` under
`approval.approved_extensions`. Nothing else here moves: the namespace label, the
promotion criteria below, and the experimental status of `run_dynamic` are
unchanged, and this proposal remains **Proposed**.

## Compatibility

Before promotion, the surface is candidate-stable and `contracts/dynamic-circuit-v1-candidate.json`
is the machine contract. After promotion, the stable contract replaces it under
criterion 8.

Because nothing changes in this proposal, rollback is not applicable until a
promotion is actually made. Once made, a reverse move would be a public API
change requiring the standard deprecation process, which is why the criteria are
gated on recorded evidence rather than on intent.

`IR_VERSION`, `CircuitIR` schema `1.0`, `docs/public_api_v1.json`, and the root
export set are unaffected either way.

## Acceptance

- [ ] Every criterion in Decision 2 holds, each with a recorded evidence artifact
      and a command that reproduces it.
- [x] Dynamic, unknown, and lossy inputs fail closed with typed diagnostics.
- [x] Planning-time rejection is verified for a target lacking conditional
      execution, including that no fallback path executes.
- [x] Sharded dynamic execution is tested for both outcomes: preserved feedback
      semantics, or an explicit blocker.
- [ ] `python tools/ci_tier.py pr-runtime` and the relevant `distributed_cpu` tier
      pass.
- [x] Static batched execution shows no regression against the recorded baseline
      on the declared Apple-arm64 CPU profile; the evidence and its limits are
      recorded above.
- [ ] The stable contract supersedes the candidate contract, and the candidate
      contract has a named owner, removal condition, and target version.
- [x] The root export set, `IR_VERSION`, `CircuitIR` schema, and
      `docs/public_api_v1.json` are unchanged.
- [x] Any criterion that cannot be met is recorded as an explicit, owned gap with a
      reason, and promotion is deferred rather than granted conditionally.
- [ ] API owner authorizes the promotion.

## Non-goals

- Promoting `run_dynamic`, providers, dialects, or native results.
- Adding public syntax for conditions, classical registers, or loops beyond what
  `DynamicCircuit` already exposes.
- Implementing the internal IR levels. That is `IR-007` and the Phase 4 contract
  work, and this proposal only consumes their result.
- Any distributed scalability, performance, or parity claim. The distribution
  criterion is a fail-closed correctness requirement, not a capacity claim.
