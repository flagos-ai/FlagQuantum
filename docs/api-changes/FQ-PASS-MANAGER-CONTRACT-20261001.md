# Pass manager contract

## Decision and authorization

Status: **proposed, not approved.** This document records what a compiler pass
manager would have to express, what the repository already provides, and what the
contract should be. It proposes **not** publishing a Qiskit-shaped pass manager in
M1, and it proposes instead the smallest surface that satisfies the requirements
that are on record. No code change is included, no approval is claimed, and no
implementation is authorized.

This is the document the parity backlog names W6-08. `ARCH-011` records it as the
prerequisite that decides whether a general instruction graph is needed; its Open
Question 3 asks what the pass-manager contract must actually express and defers
the answer here.

The change is proposed under `docs/development/PUBLIC_API_PROTECTION.md` and
`docs/development/MULTI_TEAM_DEVELOPMENT.md`. `fq.compile` is one of the 34 Stable
Core exports (`docs/public_api_v1.json`), so any change to its signature is a
protected change. A new public `PassManager` name would be a new protected name.
Neither may land before this document is approved.

## Problem and affected user journey

### What the backlog asks for

The backlog budgets twelve PR for W6: a private instruction DAG, five ports of
existing passes onto it, a deletion PR, this proposal, then `PassManager`,
`PassManagerConfig`, `AnalysisPass`/`PropertySet`, `StagedPassManager`/
`ConditionalController`, and a pass-manager drawer. The backlog's stated
justification is that Qiskit's 136 transpiler passes need a mounting point
(the original is in Chinese; this is the translation).

`ARCH-011` already measured that the mounting-point premise does not survive
inspection for the three canonicalization passes. This document addresses the
remaining part of the premise: **whether there should be a pass manager at all,
and if so, what it must express.**

### What a user can do today, and cannot

Today a user compiles through one of three surfaces:

| Surface | Reach | Selection available |
| --- | --- | --- |
| `fq.compile(circuit, coupling_map=..., optimize=..., routing_strategy=...)` | Stable Core; 11 call sites across 5 modules inside the package | four keyword arguments |
| `fq.plan(...)` / `run(...)` | Stable Core | `ExecutionOptions`; calls `compile` internally |
| `compiler.target_legalization.legalize_circuit_for_target(...)` | deep module path | explicit capability snapshot; returns evidence |

`fq.compile`'s full signature is four optional keywords
(`flagquantum/compiler/pipeline.py:245-252`): `coupling_map`, `routing_strategy`,
`optimize`, `config`. There is no way to select an optimization level, no way to
insert a pass, and no way to ask which passes ran. The fixed sequence is
hard-coded in `_optimize_to_fixed_point` (`pipeline.py:225-236`) and re-run after
routing (`pipeline.py:278-279`).

That is a real gap. The question is which artifact closes it.

## Evidence

### The pass shape a Qiskit-shaped manager assumes does not match this compiler

Qiskit's `PassManager(passes=(), max_iteration=1000)` runs tasks whose unit of
work is a circuit in, a circuit out, with `PropertySet` carrying shared analysis
between them. Measured against that shape, FlagQuantum's compiler is not a pass
pipeline. Of the 28 public top-level functions outside `_hybrid/`:

| Return type | Count | Examples |
| --- | --- | --- |
| Typed result envelope or artifact | 15 | `TargetLegalizationResult`, `NativeGateLegalizationResult`, `TopologyLegalizationResult`, `CircuitSchedule`, `RoutingCostEstimate`, `ProgramArtifactV2` |
| `CircuitIR` from a loose input type | 5 | `compile`, `optimize`, `lower_noise_model`, `route_to_topology`, `route_to_directed_topology` |
| **Exactly `CircuitIR -> CircuitIR`** | **4** | the three canonicalization passes, `record_post_routing_optimization` |
| `str` | 2 | `emit_openqasm`, `emit_qcis` |
| `list[list[Instruction]]` | 1 | `schedule_layers` |
| `None` (a verifier) | 1 | `verify_compilation_evidence_bundle` |

**Four of twenty-eight are shaped like a pass.** The other twenty-four are not,
and the fifteen that return an envelope are not incidental: they carry the
evidence the repository requires. `TopologyLegalizationResult` has 20 fields,
including `target_snapshot_id`, `source_content_hash`, `legalization_identity`,
`inserted_swap_count`, `initial_logical_to_physical`, and
`final_logical_to_physical`. A manager whose contract is
`run(circuit) -> circuit` has no place to put any of them.

This is not a stylistic preference, it is a rule collision. Engineering Decision
Principle 9 requires that "approximation, precision downgrade, backend
substitution, and CPU fallback ... must be recorded in the plan, result, and
evidence". A pass manager that returns only the transformed circuit drops exactly
that record. Adopting the Qiskit shape therefore requires either publishing the
envelopes as a parallel channel beside the manager, or weakening the evidence
contract. Neither is a reason to add a manager.

### The composition authority already exists, in two places

| Existing authority | Type it composes | Evidence it preserves |
| --- | --- | --- |
| `compiler/target_legalization.py::legalize_circuit_for_target` → `TargetLegalizationResult` | `CircuitIR` under an explicit capability snapshot | Composes `topology_legalization`, `native_gate_legalization`, `direction_legalization`, `capability_match`, and `schedule` into one recorded result |
| `compiler/_hybrid/passes.py::run_pass_pipeline` → `HybridOptimizationResult` | `HybridProgram` | `PassRecord` per pass with input/output identity, operation counts, pre-expanded iterations, statistics, and remarks; a 32-pass limit; verification between steps |

`_hybrid/passes.py` is the closer analogue, and it is worth stating what it already
provides: a `HybridProgramPass` Protocol with a `name` property and a
`run(program, analysis)` method, a `PassRecord` audit type, a frozen result object,
a default pass tuple, and a bounded driver that re-analyses between passes. That is
a pass manager with an audit trail, written for the private structured-program IR
and documented as such: `flagquantum/compiler/IMPLEMENTATION.md:30-33` states that
`_hybrid/` "is not a second circuit compiler ... and it is intentionally absent from
public exports", and
`tests/unit/test_hybrid_compilation_private_contract.py:892` pins
`"pipeline_and_audit": "flagquantum.compiler._hybrid.passes"` as that authority.

So publishing `PassManager` for `CircuitIR` would create a **third** composition
authority in the same package, over the same kind of object, with a weaker audit
record than the private one it would sit beside. AGENTS.md Engineering Decision
Principle 6 forbids that: "Do not create a second source of truth." Principle 11 is
blunter: a new manager "requires a distinct current responsibility or a second
concrete use, not a hypothetical future need."

### The orchestration boundary is Runtime's, and it is already populated

Qiskit's `transpile()` is the orchestrator that a `PassManager` serves. FlagQuantum's
equivalent is not in `compiler/`:

`flagquantum/runtime/planner/__init__.py:526-534` calls `compile_program(...)`, then
`lower_noise_model(...)`, then `analyze(...)`, then selects an execution mode. That
is the compile-flow orchestration, and it lives in Runtime because Runtime owns
execution organization while Compiler owns program transformation
(`flagquantum/compiler/AGENTS.md`, and AGENTS.md Principle 4).

Placing a staged, conditional pass manager in `compiler/` would put a second
orchestrator on the far side of that boundary. The architectural cost is not the
class; it is that "which transformations run, in what order, under what condition"
would then have two answers.

### No requirement on record asks for pass-level control

Three specific things that would create the requirement are all absent:

| Candidate requirement | Status |
| --- | --- |
| A third party installs a pass without editing the repository | No plugin entry point exists in `compiler/`, and none is requested. `grep -rn "entry_points\|register_pass" flagquantum/compiler/` returns nothing |
| A user needs to insert a pass at a named position | No issue, example, or test asks for it. The two documented compiler examples are `examples/compiler_optimize.py` and `examples/target_aware_compilation.py` (`flagquantum/compiler/IMPLEMENTATION.md:40-45`), and neither inserts a pass |
| A user needs to know which passes ran | Already answerable differently: `compile` records `runtime_config` in IR metadata (`pipeline.py:257-259`) and `record_post_routing_optimization` records routing facts |

### What is genuinely missing is a level, not a manager

Qiskit's user-facing surface is not the pass manager. Reading its published API,
`generate_preset_pass_manager(optimization_level=2, backend=None, target=None,
basis_gates=None, coupling_map=None, initial_layout=None, layout_method=None,
routing_method=None, translation_method=None, scheduling_method=None, ...)`
returns a `StagedPassManager`, and the stage choices are **strings**, not pass
objects. `PassManager` is the advanced and plugin surface; `optimization_level=0..3`
is what ordinary callers use.

FlagQuantum has no equivalent of the ordinary surface. `grep -rn
"optimization_level" flagquantum/compiler/` exits 1 with no output. Across the
whole repository the string appears in exactly four Python lines, all in
`ecosystem/qiskit/aer.py:357,384` and `benchmarking/simulator_compare.py:120,368`,
and every one of them passes the argument **to Qiskit's own `transpile()`**. That
is the clearest possible statement that the gap is real and is currently delegated
away.

An `optimization_level` parameter is an `int` on an existing function. It
introduces no type, no protocol, and no authority. It is the smallest artifact
that closes the measured gap.

### The two open questions ARCH-011 defers here

`ARCH_011_COMPILER_RESPONSIBILITY_BOUNDARY.md` Open Question 3 asks "what must the
pass-manager contract actually express", and Open Question 4 asks "should the four
optimization passes become individually addressable". Proposed answers:

**OQ3 — it must express the pass order and the evidence, not the pass objects.** The
evidence requirement is already satisfied without a manager: `TargetLegalizationResult`
and `PassRecord` both carry it, and `compile` records its configuration in IR
metadata. What is not expressed today is the order, and an ordered level table
expresses it without exposing a pass type.

**OQ4 — no.** `tests/unit/test_api_namespace_convergence.py:74-76` currently asserts
that `remove_identity_gates`, `merge_self_inverse`, and `merge_adjacent_rotations`
are **absent** from `flagquantum.compiler`, and
`IMPLEMENTATION.md:47-48` states that they "are pipeline implementation details, not
expert-facing entry points". Making them addressable would reverse an existing,
tested decision. The three names are also not a stable decomposition: `optimize`
applies `remove_identity_gates` twice per round around the other two
(`pipeline.py:230-233`), so "the four passes" is already a sequence, not a set.
If a future trigger requires addressability, it should be a `Pass` protocol over
the private sequence, not three new public functions.

## Alternatives considered

**Alternative 1 — Publish the Qiskit-shaped manager (backlog W6-09 to W6-12).**
`PassManager`, `PassManagerConfig`, `AnalysisPass`, `PropertySet`,
`StagedPassManager`, `ConditionalController`, `pass_manager_drawer`, plugin
registry. Rejected. It creates a third composition authority beside
`target_legalization` and `_hybrid/passes.py`; its `circuit -> circuit` pass shape
cannot carry the fifteen result envelopes without a parallel evidence channel; and
it places flow orchestration in Compiler while Runtime already owns it. Adopting it
before `optimization_level` exists would mean publishing the advanced surface
before the ordinary one.

**Alternative 2 — Publish `optimization_level` only, keep composition private.**
**Recommended.** One parameter on the existing Stable Core `compile`, plus private
level tables in `pipeline.py`. Every level is an ordered pass sequence, which is
what `_optimize_to_fixed_point` already is. No new public name. Delivers the
requirement that is actually on record and defers the one that is not.

**Alternative 3 — Publish a narrow `Pass` protocol and a `run_passes` driver, at
`experimental` maturity.** Rejected for now, retained as the likely next step. It
is the right shape if a second concrete consumer appears, because it generalizes
`_hybrid/passes.py`'s proven protocol rather than inventing a new one. It is not
justified yet: there is one consumer, it is private, and it operates on
`HybridProgram` rather than `CircuitIR`, so sharing the protocol would couple two
program types for no present benefit.

**Alternative 4 — Publish nothing and change nothing.** Rejected. The absent level
surface is used by the ecosystem adapters and by users who currently reach into
`optimize`/`compile` argument combinations that do not compose into a documented
level. Leaving it absent keeps Qiskit's `optimization_level` as the only name in
the repository for this concept, which is the inverse of the product identity rule.

**Alternative 5 — Move `transpile`-style orchestration into Compiler and let
Runtime call it.** Rejected. This is W11's question, not W6's, and it inverts an
approved boundary to solve a naming problem.

## Old and proposed behavior

**Old.** `fq.compile(circuit, *, coupling_map=None, routing_strategy="restore_after_each_gate", optimize=True, config=None) -> CircuitIR`.
Optimization is on or off. The passes and their order are undocumented
implementation detail, and `IMPLEMENTATION.md:47-48` states the current policy
explicitly: "Individual canonicalization functions are pipeline implementation
details, not expert-facing entry points. Change or compose them through `optimize`."

**Proposed for W11-01, under this document's approval.** Add a keyword-only
`optimization_level: int = 1`. Level 0 applies no canonical optimization and
performs no post-routing optimization. Level 1 is today's behavior at
`optimize=True`, and is the default so that no existing caller changes meaning.
Levels 2 and 3 are defined by later W11 PR as ordered pass sequences over the same
private passes. An out-of-range level fails closed with `CompilationError`, in the
manner Rule 9 requires, and not by clamping.

**Proposed for this document's decision.** No `PassManager`, no `AnalysisPass`, no
`PropertySet`, no `StagedPassManager`, no `ConditionalController`, no
pass-manager drawer, and no plugin registry, until one of the triggers below fires.
Backlog W6-09 to W6-12 are withdrawn from M1 and become contingent.

**Triggers that would reopen the pass-protocol question.** Any one of:

1. A pass must be installed without editing the repository — a plugin, an
   out-of-tree backend, or an ecosystem adapter that needs its own transformation.
2. `optimization_level` cannot express a recorded requirement, because a caller
   must insert a transformation at a named position rather than choose a level.
3. A second concrete consumer of pass composition appears over `CircuitIR`, so that
   the `_hybrid/passes.py` protocol has a proven second use.

When a trigger fires, the proposal should be the narrow Alternative 3, and it must
state how pass evidence survives composition.

## Source and behavioral compatibility impact

Nothing in this document changes behavior. The withdrawal of W6-09 to W6-12 removes
planned work and adds none.

If W11-01 is later approved, `optimization_level=1` must reproduce `optimize=True`
exactly. The verification is a differential test over the existing corpus: for
every circuit in `tests/`, `compile(ir, optimization_level=1)` must equal
`compile(ir, optimize=True)`, and `optimize` must remain a working name because it
is exported by `flagquantum/compiler/__init__.py` even though it is not Stable Core.

`optimize: bool` and `optimization_level: int` cannot both be added without a
decision on precedence. The recommended reading, to be recorded in the W11-01
proposal, is that `optimize` is retained as a deprecated alias for
`optimization_level` and that specifying both with conflicting values raises rather
than silently preferring one.

## Migration example

```python
import flagquantum as fq

# Today: optimization is on or off, and the pass order is implicit.
compiled = fq.compile(circuit, optimize=True)

# After W11-01: the level names the pass order, and level 1 is today's behavior.
compiled = fq.compile(circuit, optimization_level=1)

# Level 0 performs no canonical optimization, for callers who need the
# instructions they wrote to survive to emission.
unoptimized = fq.compile(circuit, optimization_level=0)
```

No pass object appears in either form, which is the point: the ordinary surface
must not require a user to understand the compiler's internal composition.

## Versioning

`optimization_level` is additive to a Stable Core signature and does not change the
meaning of any existing call, so it is a minor-version addition under the
compatibility rules. `optimize` remains accepted for at least one release, with a
named owner, documented scope, and a removal condition, as Principle 1 requires.

The withdrawal in this document is not a deprecation, because no pass-manager name
was ever published: `grep -rni "passmanager\|pass_manager" flagquantum/` returns 0
hits.

`IR_VERSION` is unaffected. No serialized schema, result field, or enum value
changes.

## Documentation and tooling impact

| Artifact | Change |
| --- | --- |
| `flagquantum/compiler/IMPLEMENTATION.md` | Record that `optimization_level` is the level selector and that pass composition stays private. Its current sentence at `:47-48` remains the policy and should be pointed at this document |
| `docs/generated/CAPABILITIES.md` | No change until W11-01 lands |
| `docs/architecture/decisions/ARCH_011_...md` | Its Open Question 3 is answered by this document; the answer removes the graph step's last stated justification, which the ADR already anticipates |
| No new tool | The four-of-twenty-eight measurement above is a one-off; a gate that pinned "no pass manager is published" would be a rule about a name that does not exist. If W11-01 lands, its differential test is the gate that matters |

`python tools/check_team_scope.py --team integration --files docs/api-changes/FQ-PASS-MANAGER-CONTRACT-20261001.md`
passes: `docs/api-changes/**` is a shared path.

## Owner and approvals

- Owning domain: `integration`, with `compiler` as the affected domain.
- Required approvals before implementation: **integration owner** (the withdrawal
  from M1 and the W11 sequencing) and **API owner** (`fq.compile` is Stable Core).
  The **compiler domain owner** must confirm that the level tables belong in
  `pipeline.py` and that no composition change is needed.
- Sequencing: this document, then `ARCH-011`'s decision on the boundary
  disagreement, then W11-01 if approved. W6-09 to W6-12 stay unstarted until a
  trigger in "Old and proposed behavior" fires.
- Open question for the approvers, and the one place this document could be wrong:
  if the intent of the parity effort is to accept a Qiskit-compatible *plugin*
  ecosystem, then trigger 1 is already satisfied by that intent rather than by a
  request, and Alternative 3 should be proposed now instead of later. This document
  reads the backlog's plugin PR (W11-09, W11-10) as L2 and therefore as not yet
  required, but that reading is a judgement about priority and not a measurement.
