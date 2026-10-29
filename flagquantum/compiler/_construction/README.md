# Coherent integer construction profile

Compiler owns this private profile of the existing `_hybrid.model` records.
It verifies coherent integer expression definitions and physical binding
preconditions. Binding validation checks the complete definition graph before
checking physical ports; an unchecked function cannot bypass profile verification.
Arithmetic lowering emits existing Core gates through a shared pure arithmetic
kernel. It does not capture Python, select candidates, run circuits or export a
public `synth` API.

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
Conservative arithmetic lowering requires explicit zero-workspace maps
and operation budgets. Results are retained; only locally cleaned multiplier
scratch is reused. Addition borrows the first operand, which its kernel restores.
It returns existing CircuitIR for Circuit composition and
execution, including nonzero XOR targets. See the
[CS-03 delivery record](../../../docs/roadmap/CS03_MULTIPLICATION_DELIVERY.md).

See the [CS-01 contract](../../../docs/api-changes/FQ-SYNTH-CS01-20261029.md)
and [Classiq review](../../../docs/roadmap/CLASSIQ_SEMANTICS_REVIEW_20261029.md).
