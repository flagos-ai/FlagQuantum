# Add resource width and carry ownership

`registers/arithmetic/addition_width.py` owns the automatic width binding for
existing Add arithmetic identities. It consumes `addition_components`, the
logical sequence already shared by the gate emitter and factorized Add program.
The Compiler query retains recursive cache ownership. The domain owner maps
that sequence into existing WidthStep/WidthConnection values and invokes the
existing greedy scheduler and connection-width estimator.

## Interface and lifetimes

`addition_width(call, child)` accepts an Add resource identity and a recursive
child-width query. Both operands are caller-owned, big-endian through words.
The left word is no wider than the right. Concrete positive Python integers
use the literal decomposition. A one-bit word remains an atomic view. Wider
words have the original split/join boundaries. The one-bit Add body is CNOT.

Each logical AND consumes two controls, returns those controls and creates one
carry. Its matching AND adjoint consumes that carry and returns the controls.
The width graph has no independent carry allocation/free nodes, matching the
source Cirq import. Idle words and live earlier carries count as bystanders
while a child's local peak is queried. The emitter's caller-provided physical
scratch still starts and ends in zero. This binding emits no instructions.

The adjoint uses the existing reversed logical sequence with exchanged AND
and AND-adjoint calls. Its split/join word order follows the source adjoint.
The existing Always identity queries its underlying Add as a distinct child,
including symbolic fallback. Source cache insertion order is retained.

Sympy integer objects and symbolic widths use the unchanged source fallback
maximum over signature and declared children, including zero-multiplicity
children. Thus Python Add(1,1) has width two, while the Sympy-integer version
falls back to width three. Symbolic fallback can underestimate workspace.
Concrete zero input widths fail in the original decomposition. Native admission
raises ValueError where source QubitCount wraps the error in RuntimeError.
Reversed concrete operand widths are rejected before child queries.

## Example

```python
from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_call_graph,
)
from flagquantum.drawer.call_graph import export_resource_call_graph_dot

root = ArithmeticCall("add", (2, 3))
assert query_resource_logical_width(root) == 7
assert '<TD>Qubits</TD><TD>7</TD>' in export_resource_call_graph_dot(
    root, declared_calls=arithmetic_call_graph(root)
)
```

Gate costs retain the existing arithmetic declaration map. Available connected
widths retain their own ownership when gate declarations are overridden.
Native split/join keys retain width-only identity. Arithmetic roots retain
word encodings, direction and Always. Mixed encodings at different widths are
covered. Equal-width mixed-dtype bookkeeping cache identity remains open.
Width binding does not extend the executable dtype interface.

## Separate physical counts and source models

For Add(2,3), either direction emits CX20, H6, T4, Tdg4, S2, measurement2,
conditional CZ2 and reset2. Physical work is two qubits, with two classical
slots. The source QEC model separately returns AND2, Clifford11 and
measurement2, equivalent to eight T in its model. The connected width is seven.
This binding adds zero gates, physical work and measurements.

The unchanged source model has known boundaries. One-bit Add declares
Clifford -3 while its real decomposition emits one CNOT. Add(2,3) and Add(3,3)
share the same model while emitting CX20 and CX21 respectively. These source
formula behaviors are retained and recorded independently from gate parity.
The source is never modified to make a cost formula agree with its stream.

## Acceptance scope

The locked source is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Tests verify three immutable Git
blobs for Add, Always and QubitCount. Full cache values and insertion order
cover all positive operand pairs through width five, a three/eight-bit pair,
Sympy-integer/mixed/symbolic dimensions, three encodings, both directions and
Always wrappers. Nine distinct encoding pairs at widths two/three are checked
in both directions. Seeded split/join, AND/erase and CNOT peaks verify idle
bystanders. There are 150 full-DOT comparisons over five aggregations and
three display depths. Zero-input failure/retry, reversed widths and gate
explicit-override independence are checked separately.

Concentrated execution regression reuses the original source gate and feedback
stream comparisons, factorized matrix, full complex amplitudes with an
entangled spectator, all small measurement branches, clean work and append
workflows. Controlled arithmetic, constant-add connected lifetimes and general
register-program automatic width bindings remain open under Q-0734.

Local acceptance passed 467 focused tests in 2.11 seconds, 3,731 concentrated
regression tests in 56.42 seconds and 87 integration tests in 11.61 seconds,
without skips. The concentrated run reports 1,112 upstream deprecation warnings.
The integration run reports two existing deprecation warnings. Strict typing
covers four changed Python files. Black, Ruff, team scope, architecture,
internal IR, protected API, nine Drawer doctests and the Markdown example pass.
The new owner has 53 statements and 30 branches, all covered. The resource width
dispatcher also has full statement and branch coverage. Combined measured
coverage over nine modules is 99 percent.
