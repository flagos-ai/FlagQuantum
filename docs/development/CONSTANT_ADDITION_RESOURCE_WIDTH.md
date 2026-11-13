# Constant Add resource width and allocated-word ownership

`registers/arithmetic/constant_addition_width.py` owns the connected AddK width
binding. The interface takes the existing constant-add resource identity and a
recursive child-width query. The target is a caller-owned big-endian through
word. A clean constant word is allocated, loaded by the existing XorK identity,
passed to Always(Add), unloaded by the same self-adjoint XorK and released.
Add owns the internal carries. The adjoint retains that lifetime chain and
loads the same constant while querying the exact Add adjoint.

Concrete positive Python integer widths and constants use the source connected
sequence. Other symbolic literals use the unchanged signature/callee fallback.
The shared arithmetic declaration owner supplies the normalized constant and
Always identity, including unsigned negative-constant normalization and signed
range admission. Existing width steps, connections and the greedy scheduler
account for the target as a bystander during allocation/load/free and for the
constant word across Add. No physical instructions are emitted by this binding.

Clean scalar allocation/free bookkeeping uses the existing width-only layout.
This follows the existing split/join normalization and does not distinguish
source register dtypes or dirty borrowing. Arbitrary dtype-specific seeded
cache identity and dirty/array allocation bindings remain open. Unsupported
zero-width AddK fails at admission. Source errors are compared separately from
the native ValueError contract. Symbolic fallback can underestimate workspace.

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

for adjoint in (False, True):
    root = ArithmeticCall("constant_add", (3,), constant=2, adjoint=adjoint)
    assert query_resource_logical_width(root) == 8
    assert '<TD>Qubits</TD><TD>8</TD>' in export_resource_call_graph_dot(
        root, declared_calls=arithmetic_call_graph(root)
    )
```

Users continue to append the existing constant-add operation with
`circuit.append(op)`. Physical gate emission and allocation of scratch remain
owned by the existing component implementation.

## Physical counts and source model

AddK(3,2), in either direction, emits X2, CX21, H6, T4, Tdg4, S2,
measurement2, conditional CZ2 and reset2. Physical work is five qubits,
comprising three constant bits and two carries, with two classical slots.
The separate source model returns AND2, Clifford13, measurement2 and width8,
with T equivalent eight. One-bit AddK(1,0) emits CX1 while the source category
model declares Clifford -3. The connected width query adds zero physical
gates, work or measurements. Existing
source-model boundaries, including the one-bit negative Clifford formula, remain
explicit. The source stays frozen at Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

## Acceptance scope

Six frozen Git blobs cover QubitCount, AddK/Add, XorK, Always, Allocate and Free.
Full normalized cache values and insertion order cover widths one through five,
an eight-bit word, three encodings, both directions, negative constants,
Sympy and symbolic widths/constants. Seeded allocation/free, load, Add, Always,
AND/erase and Clifford peaks verify bystanders and carry ownership. There are
120 full-DOT comparisons across five aggregations and three depths. Separate
tests verify zero admission, invalid constants with partial caches and retry,
clean scalar allocation atoms, unsupported layouts and gate-override
independence. Concentrated regression reuses the frozen gate and measurement
feedback comparisons, complex amplitudes, entangled spectators, all small
measurement branches, clean work and append workflows. Controlled constant-add
and general register-program automatic bindings remain open under Q-0734.

Local acceptance passed 378 focused tests in 2.00 seconds, 5,642 concentrated
regression tests in 83.82 seconds and 87 integration tests in 11.62 seconds,
without skips. Concentrated regression reports 1,112 upstream deprecation
warnings and integration reports two existing warnings. Strict typing covers
four changed Python files. Black, Ruff, team scope, architecture, internal IR,
protected API, nine Drawer doctests and the Markdown example pass. The new
owner has 24 statements and 12 branches with full coverage. The resource width
dispatcher also has full statement and branch coverage. Combined coverage over
eleven measured modules is 99 percent.
