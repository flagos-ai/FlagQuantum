# API Change Proposal 062: Dynamic circuit promotion criteria

## Status

**Proposed. Promotion assessment requested; no change is made by this proposal
until every criterion below holds.**

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
import flagquantum as fq

circuit = fq.DynamicCircuit(3)
circuit.h(0)
circuit.measure(0)
circuit.cx(0, 1)          # executed only when the measurement result is 1
```

The namespace already exists and is candidate-stable. Promotion changes the
label and the contract that backs it; it does not add names, change signatures,
change defaults, or alter documented exception behaviour.

If a later step concludes that a promotion requires a public surface change — a
new method, a condition expression, or a result field — that change is proposed
separately and is not covered by this proposal.

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
- [ ] Dynamic, unknown, and lossy inputs fail closed with typed diagnostics.
- [ ] Planning-time rejection is verified for a target lacking conditional
      execution, including that no fallback path executes.
- [ ] Sharded dynamic execution is tested for both outcomes: preserved feedback
      semantics, or an explicit blocker.
- [ ] `python tools/ci_tier.py pr-runtime` and the relevant `distributed_cpu` tier
      pass.
- [ ] Static batched execution shows no regression against the recorded baseline.
- [ ] The stable contract supersedes the candidate contract, and the candidate
      contract has a named owner, removal condition, and target version.
- [ ] The root export set, `IR_VERSION`, `CircuitIR` schema, and
      `docs/public_api_v1.json` are unchanged.
- [ ] Any criterion that cannot be met is recorded as an explicit, owned gap with a
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
