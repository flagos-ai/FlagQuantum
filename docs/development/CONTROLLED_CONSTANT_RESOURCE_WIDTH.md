# Controlled constant resource width and caller control ownership

Controlled XorK belongs to `registers/bits/controlled_constant_width.py`.
Controlled AddK extends the existing constant-add lifetime owner in
`registers/arithmetic/constant_addition_width.py`. Both use the shared
`registers/control/single_zero_control_width.py` connection model for the
single control-on-zero wrapper. No new resource identities or instruction
representation are introduced.

Both interfaces accept the existing arithmetic call and recursive child-width
query. Control and target are caller-owned through registers. Words are
big-endian. Positive controlled XorK reuses the existing controlled XOR emitter
and signed range admission, with split/join boundaries and each CNOT's operands.
It uses no work or measurement slots. AddK allocates a constant word and controls
only XorK loading and unloading. Always(Add) keeps its original carry ownership.
The caller control stays live as a bystander across Add and allocation/free.
The exact adjoint loads the same constant and reverses Add direction.

Control-on-zero uses the locked ControlledViaAnd sequence of control X,
positive controlled child and control X. Its aggregate word connections support
symbolic widths, preserving cache visitation and idle-word peaks. Positive
connected decompositions require Python integer width and constant. Other
literals retain the source signature/callee fallback, which can underestimate
workspace. Source self-adjoint XorK child identity is canonical, while a native
root can retain its requested direction field.

Existing dtype-to-width normalization, clean scalar allocation admission and
limited scalar cast support are retained. General control shapes, multi-control
width bindings, dtype-specific seeded caches, dirty/array allocation and general
register-program width bindings remain open. Source zero/range failures are
checked beside native admission errors and partial-cache retry behavior.

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
        root = ArithmeticCall(
            "controlled_constant_add", (3,), constant=2,
            control_value=control_value, adjoint=adjoint,
        )
        assert query_resource_logical_width(root) == 9
        assert '<TD>Qubits</TD><TD>9</TD>' in export_resource_call_graph_dot(
            root, declared_calls=arithmetic_call_graph(root)
        )
```

Users continue to use the existing operation with `circuit.append(op)`.
The positive controlled-XOR emitter and existing control-X instructions are
reused by execution tests for both coherent polarities. This width binding
emits no physical instructions and allocates no extra workspace.

## Separate physical counts and source model

For controlled AddK(3,2), either direction emits CX23, H6, T4, Tdg4, S2,
measurement2, conditional CZ2 and reset2. Physical work is five qubits with
two classical slots. Control-on-zero adds X2. The separate source model returns
AND2, Clifford13, measurement2 and width9 for positive control. Control zero
raises modeled Clifford to15. T equivalent remains eight.
The QEC category model and actual decomposed instruction counts remain separate,
including the inherited one-bit negative Clifford formula. Controlled XorK(3,2)
emits one CNOT, plus two X gates for control zero, with no work or measurements.

## Acceptance scope

The unchanged reference is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Eight frozen Git blobs cover the
arithmetic, allocation, Always, QubitCount and both control wrappers. Tests
compare full normalized cache values and insertion order for both operations,
three encodings, both control polarities, both directions, negative constants,
widths one through five/eight and Python/Sympy/symbolic dispatch. Seeded local
peaks cover views, load, allocation/free, Add, Always, AND/erase and Cliffords.
An unused X cache entry retains its value without inflating positive-control
width. There are 480 full-DOT comparisons over five aggregations and three
depths. Zero/range failures retain matching partial caches and allow retry.
For control zero, a failed positive child leaves the outer X cache entry.

Execution tests compare source primitive/feedback order and resource counts.
Both coherent XorK polarities preserve full complex amplitudes with entangled
spectators, and repeated append restores the input. Existing controlled AddK
regression covers exact inverse gates, all small measurement histories, complete
complex amplitudes, clean work and append. Source dtype-specific seeded caches,
clean scalar allocation limitations, symbolic fallback underestimation and
other control shapes remain explicit open boundaries.


Local acceptance passed 1,342 focused tests in 4.04 seconds, 6,951 concentrated
regression tests in 106.26 seconds and 87 integration tests in 14.69 seconds,
without skips. The concentrated run reports 1,112 upstream deprecation warnings
and integration reports two existing warnings. Strict typing covers seven
changed Python files. Black, Ruff, team scope, architecture, internal IR,
protected API, nine Drawer doctests and the Markdown example pass. Controlled
XorK has 30 statements and 10 branches, the shared zero-control wrapper has
13 statements and six branches, and the extended constant-add owner has
31 statements and 16 branches. All three owners and the resource width
dispatcher have full statement/branch coverage. Thirteen measured modules have
99 percent combined coverage. Resource encodings preserve source identity and
do not expand the existing executable dtype interface.

An initial seeded test incorrectly required an unused X cache entry to inflate
positive-control width. The unchanged source retains that entry without using
it in the root peak, and the test now verifies that behavior. Strict typing also
required importing the encoding fixture from its defining test module.
