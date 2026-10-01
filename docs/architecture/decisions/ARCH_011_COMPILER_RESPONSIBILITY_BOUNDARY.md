# ARCH-011: Compiler responsibility boundary and pass infrastructure

Status: Proposed

Date: 2026-10-01

Scope: Which responsibilities `flagquantum/compiler/` owns, which move to another
domain, and what pass infrastructure may be introduced before the compiler's
public contract is extended. Internal architecture only. No Stable Core change,
no capability level change, and no implementation authorization.

## Context

Two machine-readable files disagree about what Compiler may depend on, and no
check compares them. `contracts/long-horizon-architecture-v1.json` declares domain
`compiler` with `may_depend_on = ["core"]`.
`architecture.toml` gives Compiler a seven-entry deny list,
`compiler_forbidden = ["_compiler", "compilation", "runtime", "deployment",
"simulation", "compute", "remote"]`. The two rules agree on six entries and differ
on everything else: the contract permits only `core`, while the policy permits
every other package in the repository. `noise` is the concrete case. It is absent
from the deny list, and `flagquantum/compiler/noise.py:10` imports from it — the
single edge that turns a disagreement into a live dependency.

`docs/api-changes/FQ-COMPILER-BOUNDARY-SPLIT-20260930.md` records the full
measurement: six distinct responsibilities inside one package, a 10,380-line
package with a 970-line `physical_plan.py` against a 1250-line ceiling, and
`CouplingMap` acting as a cross-domain contract type at five import sites while
being absent from `docs/public_api_v1.json`. This ADR decides the split. It does
not re-derive that document's evidence.

### The pass infrastructure premise does not survive inspection

The parity backlog proposes `refactor(compiler): introduce the internal directed
acyclic graph representation` as W6-02, described as the mounting point for the
136 Qiskit transpiler passes. Three measurements say that premise is wrong as
stated.

**First, the optimization path has three private passes and one public function.**
`flagquantum/compiler/pipeline.py` — 242 lines — contains
`remove_identity_gates` (`:72`), `merge_self_inverse` (`:97`),
`merge_adjacent_rotations` (`:119`), `schedule_layers` (`:154`), and the fixed-point
driver `_optimize_to_fixed_point` (`:173`). Only `schedule_layers` is exported
(`flagquantum/compiler/__init__.py:6-12`), and
`tests/unit/test_api_namespace_convergence.py:74-76` asserts the other three are
**not** reachable from `flagquantum.compiler`. So W6-03, W6-04, and W6-05 port
three functions that no consumer outside one module can name.

**Second, the cost they impose is quadratic in one helper, not architectural.**
`_last_touching_instruction` (`:86-94`) scans the output list backwards for the
last instruction sharing a wire. Both merging passes call it once per instruction.
Measured on this checkout, `merge_adjacent_rotations` on distinct-wire gates:

| Gates | One pass, seconds | Ratio vs half |
| --- | --- | --- |
| 500 | 0.0034 | |
| 1000 | 0.0136 | 4.0x |
| 2000 | 0.0533 | 3.9x |
| 4000 | 0.2296 | 4.3x |

Four times the time for twice the gates. `_optimize_to_fixed_point` (`:173-184`)
calls four passes per round for up to `len(ir) + 1` rounds. The defect is a
backwards list scan, and a per-wire position index removes it. It is not a reason
to introduce a graph object.

**Third, an instruction dependency representation already exists.** Core owns
`ScheduledInstruction` with sorted unique `predecessors` and matching
`dependency_kinds` (`flagquantum/core/_compilation_evidence.py:263-294`), and
`compiler/schedule_legalization.py` builds a `CircuitSchedule` whose instructions
carry those predecessors and a computed layer. That is a dependency graph over
instructions, with barrier, classical, and wire edges, already implemented,
already validated, and already consumed by the evidence format. A second
instruction-graph representation in `compiler/` would duplicate an authoritative
type, which `AGENTS.md` engineering principle 6 prohibits.

What the three optimization passes need is narrower than any of these: the most
recent live position per wire. That is one index, not a representation.

### The strategic priority freezes the abstraction the backlog proposes

`AGENTS.md` binds the current sequence: "freeze new horizontal abstractions;
finish and simplify the current Compiler boundary; deliver the smallest complete
CPU vertical path". It then permits a new abstraction "only when the current
vertical path cannot be completed with an existing authoritative type".

The CPU vertical path can be completed with the existing types. A pass manager
with the 136-pass Qiskit surface is not on that path, and W6-08 — the API change
proposal for a public pass-manager contract — has not been written. Introducing
the mounting point before its contract is proposed inverts the sequence the
repository already committed to.

## Decision Candidates

**Candidate 1 — Resolve the dependency disagreement in favor of Core-only, and
replace W6-02 with a per-wire index.** Set `architecture.toml`'s
`compiler_forbidden` to include `noise`, matching
`contracts/long-horizon-architecture-v1.json`, and move `lower_noise_model` to the
`noise` domain. Replace the "introduce the internal DAG" step with an internal
per-wire position index inside `pipeline.py`: no new module, no new public type,
no second representation, and the list scan deleted in the same change.

**Candidate 2 — Adopt the wider policy and amend the contract.** Keep
`compiler_forbidden` as it is and add the permitted domains to
`contracts/long-horizon-architecture-v1.json`. Consequence: Compiler keeps a
dependency on `noise`, and the contract stops meaning "Compiler depends on Core
semantics", which is the domain definition the repository states elsewhere.

**Candidate 3 — Introduce the DAG as the backlog proposes, and staff it.**
Consequence: it arrives with three private consumers, one of which
(`remove_identity_gates`) is already linear, and it duplicates
`ScheduledInstruction`. It must be justified as a second authority over the same
fact.

**Candidate 4 — Defer all compiler work until W6-08 exists.** Write the pass-manager
API proposal first, and let its required analysis surface decide whether a graph
is needed and what it must express. Consequence: nothing improves in the meantime,
including the measured quadratic cost, which is independent of the pass manager.

**Candidate 5 — Change nothing.** Consequence: two files keep disagreeing, and the
next contributor who reads `architecture.toml` concludes Compiler may import
`services` and `twin`.

## Prohibited Practices

1. **Do not introduce a general instruction-DAG class in `compiler/` before an
   approved pass-manager contract requires it.** `ScheduledInstruction` already
   carries predecessors, dependency kinds, and a layer.
2. **Do not port a pass without deleting the implementation it replaces in the
   same change.** A coexistence window for three private functions with one caller
   each is not a migration; it is two code paths.
3. **Do not treat `architecture.toml` and
   `contracts/long-horizon-architecture-v1.json` as independent.** Any change to
   one boundary rule must state its effect on the other, because no tool compares
   them.
4. **Do not extend the compiler's public contract through W6-02, W6-06, or
   W6-07.** `flagquantum/compiler/__init__.py`'s six-name
   `__all__` is the published surface, and
   `tests/unit/test_api_namespace_convergence.py:74-76` pins the private status of
   the three optimization passes. An intermediate representation is not a reason
   to publish a name.
5. **Do not describe the quadratic helper as an architecture finding.** It is a
   local defect in one function with a measured fix, and it must not be cited as
   evidence for a framework.

## Compatibility

- **Moving `lower_noise_model` is a public contract change.** It is one of the six
  names in `flagquantum/compiler/__init__.py:15-22` and is absent from
  `docs/public_api_v1.json`, so it is neither Stable Core nor an intentional
  compatibility surface. It must keep working from its current path for at least
  one release with an alias, or the move is a breaking change to an expert
  interface.
- **Tightening `compiler_forbidden` can fail an existing import.** The only
  Compiler-to-`noise` edge today is `flagquantum/compiler/noise.py:10`. The
  boundary rule must therefore be changed together with that file, not before it.
- **`noise` gains Compiler-independent ownership of noise-model lowering.** Its own
  boundary (`architecture.toml`, `noise_forbidden`) already forbids `circuit`,
  `compilation`, `deployment`, `runtime`, and `simulation`; adding the inverse edge
  makes the two rules symmetric rather than adding a new one.
- **The per-wire index changes no observable behavior.**
  `_last_touching_instruction` returns the maximum live position among a wire set;
  a per-wire stack returns the same maximum, because a popped instruction always
  shares exactly the current instruction's wire set. Equivalence is a test
  obligation, not an assumption; see the acceptance tests.
- **`schedule_layers` is unaffected.** It is public, has five importers
  (`flagquantum/circuit.py:644`, `runtime/planner/__init__.py:12`,
  `runtime/executors/statevector/planning.py:12`,
  `runtime/execution_plan_contract.py:90`, `compiler/__init__.py:12`), and is
  already linear.
- **`physical_plan.py` is not touched by this ADR.** Its 970-line occupancy is
  recorded in the W0-06 proposal and belongs to the split decision, not to the pass
  infrastructure decision.
- **`docs/architecture/decisions/**` is an integration-protected path.**
  `tools/check_team_scope.py` requires `--team integration` for it.

## Acceptance Tests

1. **The two boundary declarations agree.** A check compares
   `contracts/long-horizon-architecture-v1.json`'s per-domain `may_depend_on`
   against `architecture.toml`'s boundary lists and fails on any entry where
   neither is implied by the other. Without it, this ADR's own premise can recur.
2. **The dependency edge is gone, not waived.**
   `grep -rn "from ..noise import" flagquantum/compiler/` returns nothing, and
   `noise` is in `compiler_forbidden`.
3. **The optimization passes are behavior-equivalent.**
   A differential test runs every sequence of `remove_identity_gates`,
   `merge_self_inverse`, and `merge_adjacent_rotations` over randomized circuits
   — including repeated wires, disjoint wires, interleaved operations, zero and
   non-zero rotations, parametrized and matrix instructions, and parameter tensors
   with `requires_grad` — against the list-based implementations, and asserts
   identical instruction sequences. The reference implementation is retained in
   the test file, not in the package.
4. **The public surface is unchanged.**
   `tests/unit/test_api_namespace_convergence.py:74-76` and
   `tools/public_api_snapshot.py` still pass, and
   `flagquantum/compiler/__init__.py` still exports exactly six names.
5. **The quadratic cost is gone.** A timing test on 4000 distinct-wire gates
   asserts a bound that the list implementation cannot meet, so a future
   regression to a backwards scan fails rather than slowing the suite silently.
6. **The pass manager is not published.** No name matching a pass-manager type
   appears in `flagquantum/compiler/__init__.py` before W6-08 is approved.
7. **The Compiler boundary is machine-checked where it is claimed.**
   `tools/check_architecture.py` enforces boundary lists for seven prefixes but has
   none for `observables`; this ADR does not add one, and it does not claim the
   checker is complete.

## Open Questions

1. **Does `noise` or Compiler own `lower_noise_model`?** The function lowers a
   noise model onto a program, which is a transformation, but the type it lowers is
   the `noise` domain's. Candidate 1 puts it in `noise`; the reverse reading is
   defensible if the output is a program.
2. **Is `_hybrid/` — 4,361 lines, 42% of the package — Compiler's?** The W0-06
   proposal does not resolve it and this ADR does not either. It is the largest
   unexplained mass in the package.
3. **What must the pass-manager contract actually express?** Until W6-08 exists,
   whether a graph is needed at all is unknown. Analysis and preservation metadata
   requirements (W6-10) are the likely deciding input.
4. **Should the four optimization passes become individually addressable?** A pass
   manager implies a pass object per optimization. That is a public contract
   question and belongs to W6-08, not to this ADR.
5. **Does the quadratic fix need a benchmark, or is a bound test enough?** The
   repository separates correctness tests from benchmark evidence; a timing
   assertion is neither, and its placement should be decided once.

## Approval record

Not approved. This ADR requires the **integration owner** for the boundary change
and the **compiler domain owner** for the responsibility split. Candidate 1 also
requires the **noise domain owner**, because it moves a public name into that
domain. No approval is claimed and no implementation is authorized by this
document.

Sequencing if Candidate 1 is approved: this ADR, then the boundary change with
`compiler/noise.py` and the two declarations in one integration change, then the
per-wire index as a compiler-owned change with its differential test, then
W6-08. The general-purpose graph step is withdrawn from the plan until W6-08
declares a requirement for it.
