# Coherent integer construction profile

Compiler owns this private profile of the existing `_hybrid.model` records.
It verifies coherent integer expression definitions and physical binding
preconditions. Binding validation checks the complete definition graph before
checking physical ports; an unchecked function cannot bypass profile verification.
Arithmetic lowering emits existing Core gates through a shared pure arithmetic
kernel. `frontend_python.py` captures a restricted Python integer expression
without running its body. This package does not run circuits or export a public
`synth` API. Its private CS06-B selector checks the two existing schedules.

The same `IRType`, `ValueId`, `Value`, `Operation`, `Block`, `Region`, source
location and `HybridProgram` are reused. `quantum.uint(width)` represents a
logical basis-encoded value, not an allocated register or a consumed effect
token. Function inputs are immutable basis borrows. Repeated reads of a value
are legal; overlapping physical input/output/workspace ports are not.

Supported records are `arith.constant` with classical INDEX type and a checked
nonnegative integer, `qint.add`, `qint.multiply`, `qint.call`, and `qint.return`.
Functions return one quantum integer expression. Definitions form an acyclic
call graph. This is a closed verifier profile, not an extension to the existing
classical hybrid verifier or its lowering pipeline.

Unknown operations and attributes fail before effect-specific validation. Role
and trust flags cannot silently change the borrowed-input contract. General
qfunc/qperm effects, fresh assignment, truncating in-place XOR, wrapping addition,
bind, free/drop and within-apply need separate semantics; their names are not
aliases for the full-width arithmetic evaluator.

The lowering contract requires `z ^= f(inputs)` for every target bitstring
with zero workspace, preserving phases and cleaning workspace. The verifier
checks definitions and bindings; arithmetic lowering implements this permutation
for constants, addition, multiplication, calls and returns. `fresh_zero` is an allocator obligation; `caller_asserted_zero` is a
caller precondition. Neither is inferred from an external state.

For a small change, edit the corresponding verifier, add an accepted/rejected
scenario, and run:

```bash
python -m pytest tests/team/compiler/test_construction_profile.py -q
python -m pytest tests/team/compiler/test_construction_semantics.py -q
python -m pytest tests/team/compiler/test_construction_dependencies.py -q
python -m pytest tests/integration/test_construction_execution_boundary.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

The CS-01 boundary probes use existing Circuit/IR/runtime capabilities. They
establish destination semantics independently of arithmetic lowering. Three-bit states
pin the physical bit-order boundary. A Bell subsystem probe shows why omitting
an output does not reclaim its entangled partner; it does not implement drop.
Arithmetic lowering requires explicit zero-workspace maps and operation budgets.
The default `retain` schedule retains results and reuses only locally cleaned
multiplier scratch. Addition borrows the first operand, which its kernel restores.
It returns existing CircuitIR for Circuit composition and
execution, including nonzero XOR targets. See the
[CS-03 delivery record](../../../docs/roadmap/CS03_MULTIPLICATION_DELIVERY.md).

See the [CS-01 contract](../../../docs/api-changes/FQ-SYNTH-CS01-20261029.md)
and [Classiq review](../../../docs/roadmap/CLASSIQ_SEMANTICS_REVIEW_20261029.md).

For Python capture, run `tests/team/compiler/test_construction_python.py`.
The [CS-04 workflow](../../../docs/roadmap/CS04_PYTHON_TRIAL_DELIVERY.md)
composes this frontend from Services; Compiler still has no Circuit or Runtime
dependency. Both ordinary functions and optional `qfunc` declarations lower to
the existing verified construction profile, with source diagnostics.

`lifetimes.py` records direct use sites and inverse dependencies after inlining
calls into existing SSA. `lowering.py` uses one emitter for `retain` and
`recompute`. The latter can clean operands between uses, restore missing
dependencies before inverse computation and rebind regenerated kernels to their
current qubits. Only inverse-cleaned allocations return to the zero pool.
This is a deterministic heuristic, not a general free/drop operation or optimal
pebbling. The private `schedule_arithmetic` diagnostic includes peak workspace
and compute/cleanup events; `lower_arithmetic` still returns CircuitIR.

Run `python -m examples.synthesis_workspace`, then
`tests/team/compiler/test_construction_scheduling.py` and
`tests/integration/test_construction_scheduling_execution.py` for the shortest
CS-05 contributor path. The [delivery record](../../../docs/roadmap/CS05_WORKSPACE_SCHEDULING_DELIVERY.md)
states measured costs and separates large structural checks from small real CPU
phase tests. The [private boundary](../../../docs/api-changes/FQ-SYNTH-CS05-20261029.md)
keeps automatic candidate selection and broader effects outside this change.

CS06-A adds `constraints.CircuitLimits`, the pure shared authority for final
whole-program width, operation and depth ceilings. Its violations reuse existing
diagnostics and ResourceEstimate; it does not select candidates, plan execution
or infer workspace ownership from CoreIR. Services consumes this record when
checking a sealed Runtime plan. See the
[CS06-A contract](../../../docs/api-changes/FQ-SYNTH-CS06-A-20261029.md).

CS06-B `selection.py` enumerates the existing retain/recompute schedules, keeps
their actual allocation histories, and ranks fully checked candidates. Final
CircuitLimits and workspace ceilings are distinct from ConstructionBudget.
`target.py` reuses basis conversion, compilation, routing, Layout and ASAP
scheduling; it predicts normal Runtime instructions without importing Runtime.
Each target stage retains its existing CoreIR content identity, and `validate()`
rejects later mutations without compiling. Runtime confirmation still goes
through CS06-A; C owns the single normal facade and named reading integration.

Run `python -m examples.synthesis_selection` for the real constraint-driven
phase/cleanup workflow, or `python -m examples.synthesis_selection_costs` for the
complete static table and current manual comparison. Relevant tests are
`tests/team/compiler/test_construction_selection.py` and
`tests/integration/test_construction_selection_execution.py`. See the
[contract](../../../docs/api-changes/FQ-SYNTH-CS06-B-20261029.md) and
[delivery record](../../../docs/roadmap/CS06_CANDIDATE_SELECTION_DELIVERY.md).
