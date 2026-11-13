# Controlled register-XOR resource connections

The bits domain owns `controlled_xor_width.py`. It binds the existing
`ArithmeticCall("controlled_xor", ...)` to recursive logical widths and Drawer.
The existing `controlled_register_xor.py` emitter supplies each atomic Toffoli
and its operand order. The generic through-wire owner supplies connections,
while `single_zero_control_width.py` supplies the control-zero wrapper.

## Interface and ownership

The owner accepts the existing arithmetic identity and child-width callback.
The control and two equal-width big-endian words remain caller-owned. Positive
control splits both words, applies one Toffoli to each corresponding source and
target bit, and joins both words. Each Toffoli preserves the control and source.
The source sequence is self-adjoint. A native root can retain its explicit
direction, while a nested positive child uses the source-canonical direction.
Zero control applies X, the positive child and X, with both words live as
bystanders during the flips. Extra quantum work and classical slots are zero.

Python positive widths have connected decompositions. Sympy integers and symbols
retain the signature/callee fallback and its Toffoli cache visitation. Both Python zero and Sympy zero widths fail in the source. Native
admission raises ValueError beside its RuntimeError. A failed zero-control query
retains the already-queried X entry and allows a later valid query to reuse the
cache. No failed root is cached.

Bookkeeping keys retain the existing dtype-to-width normalization. Arbitrary
dtype-specific seeded caches, dirty/array allocation and general register/control
inference remain open. Symbolic fallback retains possible simultaneous-workspace
underestimates. This batch's bound body uses no extra workspace.

## Executable example

```python
from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_call_graph,
)
from flagquantum.drawer.call_graph import export_resource_call_graph_dot

for control_value in (0, 1):
    for adjoint in (False, True):
        call = ArithmeticCall(
            "controlled_xor", (3,), control_value=control_value,
            adjoint=adjoint,
        )
        assert query_resource_logical_width(call) == 7
        assert '<TD>Qubits</TD><TD>7</TD>' in export_resource_call_graph_dot(
            call, declared_calls=arithmetic_call_graph(call)
        )
```

Users keep `circuit.append(op)` with the existing operation implementation.
Execution tests preserve complete complex control/source/target/spectator
amplitudes and restore the input on a second append. Existing grouped-XOR tests
also compare source operands and feedback, all small forced measurement
histories and clean predicate work for multi-control compositions.

## Separate emitted and modeled resources

For three-bit words, either direction emits three atomic CCX gates. Control
zero adds two X gates. Quantum work and measurement slots are zero. The source
category model separately returns Toffoli3, Clifford0 for positive control or
Clifford2 for zero control, measurement0, logical peak7 and T equivalent12.
Atomic gates and a T-equivalent estimate are recorded separately. The width
binding adds zero gates, work or measurements to the existing physical stream.

## Acceptance scope

Reference Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389` remains unchanged. Eight frozen blobs
cover bitwise decomposition, both control wrappers, split/join, Toffoli and
QubitCount/category budgets. Full normalized caches and insertion order match for both polarities
and directions, three encodings, Python widths one through five/eight,
Sympy literals and positive/nonnegative symbols. Seeded peaks cover Toffoli,
control X, views and the positive child, including unused entries under symbolic
fallback. There are 240 full-DOT comparisons across five aggregations and three
depths. Gate declaration overrides retain connected ownership. Zero failures,
partial caches and retry are checked against the frozen source.

The initial child-peak fixture constructed symbolic Split/Join objects while
seeding unrelated Toffoli/X cases. Those source constructors reject symbolic
dtypes. The fixture now uses concrete width-three view seeds for symbolic
roots, preserving their unused cache entries. A dispatch edit's indentation
was corrected during inspection before tests ran.

The resource-report script initially called a reference-only method on the
native category record. It now uses the shared budget owner and asserts the
result against the reference default of four T equivalents per Toffoli.

Local acceptance passed 1,060 focused tests in 4.49 seconds, 8,083 concentrated
regression tests in 99.60 seconds and 87 real-kernel/CPU integration tests in
11.99 seconds, without skips. The regression reports 1,112 upstream deprecation
warnings, and integration two existing warnings. Strict typing covers all three
changed Python files. Black, Ruff, team scope, architecture, internal IR,
protected API, nine Drawer doctests and the executable Markdown example pass.
The new owner has 30 statements and ten branches, all covered. The resource-width
dispatcher also has full statement/branch coverage. Seventeen measured modules
have 99 percent combined coverage. Q-0734 remains open for further bindings.
