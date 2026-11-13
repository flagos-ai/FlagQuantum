# Automatic native resource widths

Compiler binds supported resource identities in
`flagquantum/compiler/_component_trace/resource_width.py`. The existing recursive
query owns caching and failure detection. The connection-width owner accounts
for live bystanders. Drawer now uses these bindings when its optional
`logical_width` callback is omitted.

## Shortest workflow

```python
from flagquantum.compiler._component_trace.resource_generalization import (
    OnEachCall, PhaseCall, PowerCall,
)
from flagquantum.drawer.call_graph import export_resource_call_graph_dot

root = PowerCall(OnEachCall(PhaseCall("t"), 4), 3)
dot = export_resource_call_graph_dot(root)
assert '<TD>Qubits</TD><TD>4</TD>' in dot
assert '<TD>Ts</TD><TD>12</TD>' in dot
```

Callers with another decomposition owner can still supply the existing width
callback. Its invocation, depth behavior and sparse `None` values are preserved.
The width query traverses the complete decomposition independently of displayed
call depth. Input declaration mappings are caller-owned.

## Binding scope

- `ComponentCost` uses its existing local peak and per-call bystander ownership.
  Zero-repetition calls remain inactive under that existing component model.
- `CliffordCall`, `PhaseCall`, `FixedTCall` and width-bearing identity calls use
  atomic signatures. Concrete Clifford gate widths must match their signatures.
- Integer-layout split/join calls use their declared register width.
- A concrete `PowerCall` visits its child once, reusing the child's sequential
  peak. Symbolic powers retain the source's unsuccessful decomposition boundary.
- Concrete `OnEachCall` visits split, its one-wire child and join. The connected
  peak includes every other live wire. Equal independent occurrences share one
  representative connection calculation and a single cached child query.
- Python integer zero visits split/join only and returns zero. A Sympy zero
  follows the source symbolic fallback, visits the child and returns one for a
  one-qubit atomic child. Symbolic counts also retain that fallback.

A decomposition retains its width ownership when explicit gate call-graph rows
are supplied. Fallback rows visit all declared callees, including zero
multiplicity, following the source width rule. Fallback may underestimate
simultaneously live workspace. Unsupported bindings raise `NotImplementedError`.
Automatic arithmetic, conjunction, controlled-swap and general register-program
binding remain open under Q-0734. Their existing explicit width-provider path is
available.

The native keys and source objects can equate Python and Sympy numeric values
while decomposition dispatch differs. Use a fresh cache when the concrete versus
symbolic representation changes, as with other decomposition-owner changes.

## Reference and acceptance

The locked reference is Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. The tests verify frozen Git blobs for
`QubitCount`, `OnEach` and `Power`. No production module imports source objects.

`tests/unit/test_resource_width_query.py` compares atomic widths, complete
wrapper caches, symbolic/zero dispatch and hidden workspace against unchanged
source. Every factory comparison uses a source-computed width table as the
explicit-provider reference, across five aggregations and three display depths.
A one-wire child with local peak eight requires eleven qubits when applied
independently to four wires, accounting for three bystanders. Gate declarations
and component width ownership are independently tested.

This resource/drawing work emits zero primitive gates, work qubits and
measurements. Existing execution regression verifies complete complex
amplitudes, measurement branches, clean workspace and `circuit.append(op)`.
Measured primitive counts and analytical model counts remain separate.
