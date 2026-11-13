# Bitwise arithmetic resource width bindings

`registers/bits/bitwise_width.py` owns connection-width decomposition for the
existing XOR, constant XOR and bitwise NOT arithmetic identities. The shared
Compiler resource query delegates these operations to that owner. The caller
keeps ownership of every input/output word. Registers are big-endian through
words. These operations allocate no quantum work, classical slots or logical
outputs. XOR retains both words. Constant XOR and NOT retain one word.

The XOR and constant-XOR paths reuse `xor_ir` and `xor_constant_ir`. The
connection projection adds the original split/join views around those native
streams. Signed constants reuse existing declaration range checks and convert
to the same unsigned bit pattern used by the emitter. NOT retains its existing
OnEach child boundary. These self-adjoint operations keep the native root's
explicit direction field and the source's unchanged child identities.

Concrete Python widths and constants decompose. Sympy integer objects and
symbolic expressions retain source fallback and sparse-cache semantics. A
concrete zero XOR constant never visits X during decomposition. A symbolic zero
object follows the fallback and visits X. Both Python and Sympy zero widths
fail before querying children. Native admission raises ValueError where the
source QubitCount wraps its decomposition error in RuntimeError. Out-of-range
concrete constants follow the same error-type adaptation. A one-bit signed
constant +1 is outside the source dtype's range. It is tested as a rejection.
The generic fallback may underestimate simultaneous workspace.

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

root = ArithmeticCall("xor", (3,))
assert query_resource_logical_width(root) == 6
assert '<TD>Qubits</TD><TD>6</TD>' in export_resource_call_graph_dot(
    root, declared_calls=arithmetic_call_graph(root)
)
```

Width decomposition is independent from category declarations. The resource
Drawer uses the existing explicit arithmetic declaration map for gate costs.
An explicit gate declaration never replaces an available connection-width
model. The normal executable path remains `circuit.append(op)`.

## Acceptance scope

The reference is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Tests verify the immutable Git blobs
for bitwise.py, on_each.py and QubitCount. Complete width caches and insertion
order cover unsigned, two's-complement and Montgomery word identities, both
native directions, Python integers, Sympy integers and symbolic widths and
constants. Signed negative patterns, zero flips, seeded larger child peaks,
invalid constants and zero-width failure/retry have independent checks.
Full DOT comparisons use source-computed width rows across all five gate
aggregations and three visibility depths.

This width binding adds zero gates, quantum work and measurements. Actual
three-bit XOR emits CX3, constant XOR with pattern 3 emits X2, and NOT emits X3.
Their QEC Clifford counts are respectively 3, 2 and 3, with zero T and
measurements. Emitted primitive streams, complex amplitudes with an entangled
spectator, involution and append are covered separately by the execution tests.
Uncontrolled integer addition, controlled arithmetic and general register
program bindings remain open under Q-0734.

Local acceptance passed 307 focused tests in 1.98 seconds, 3,114 concentrated
regression tests in 50.65 seconds and 87 integration tests in 12.26 seconds,
without skips. The concentrated run reports 1,112 upstream deprecation warnings.
The integration run reports two existing deprecation warnings. Strict typing
covers all three changed Python files. Black, Ruff, team scope, architecture,
internal IR, protected API, nine Drawer doctests and the Markdown example pass.
The new owner has 39 statements and 18 branches, all covered. The shared width
dispatcher has full statement and branch coverage. The eight-module measured
coverage scope is 99 percent overall. The matrix includes 135 full-DOT
comparisons and complete cache insertion-order checks.

Existing bookkeeping keys represent width-only split/join identities. The
arithmetic root retains dtype encoding and explicit direction. Source identities
are translated at the test boundary, one dtype per source query. Distinct mixed
dtype bookkeeping cache identities are outside this acceptance scope.
