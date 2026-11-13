# CAdd resource width and loaded-word ownership

`registers/arithmetic/controlled_addition_width.py` owns the connected width
binding for the existing CAdd arithmetic identity. The Compiler dispatcher
composes it with the accepted Add width owner and existing AND width identities.
The existing greedy scheduler and connection estimator retain cache ownership
and account for idle registers. Physical emission remains owned by
`controlled_addition.py` and its shared Add and measured-AND components.

## Interface, ordering and ownership

`controlled_addition_width(call, child)` accepts positive Python integer widths
with the left word no wider than the right, either control value and either
direction. Control and both big-endian words are caller-owned through registers.
Each controlled-load AND creates one temporary bit. Add consumes the joined
loaded word and target, returning both words, and owns its internal carries.
Reverse-order AND adjoints consume the loaded bits. The graph has no separate
allocation/free nodes for those AND outputs. Physical scratch starts and ends
in zero, while classical slots retain the measurement outcomes.

The forward connection sequence follows the frozen CAdd composite, including
one-bit split/join boundaries and the unsigned-to-left-encoding cast. The
adjoint reverses edges, child direction, cast encoding order and view order.
The Add child retains its accepted scalar-view and carry rules. A child's
modeled local peak includes every live caller word and loaded bit as bystanders.
Gate declaration overrides cannot replace an available lifetime decomposition.

Cast uses the existing BookkeepingCall layout tuple
`(width, input_encoding, output_encoding)`. This binding accepts only positive
concrete scalar widths and the three supported arithmetic encodings. General
cast shapes and other dtype families remain open. Split/join identities retain
the existing width-only normalization. In particular, distinct source QBit,
QAny and QUInt one-bit cache entries coalesce. Complete source cache values and
insertion order are compared after this normalization. Arbitrary dtype-specific
seed overrides, including equal-width mixed encodings, remain open.

Sympy integers and symbolic widths retain the original signature/callee maximum
fallback and zero-multiplicity child visits. This fallback can underestimate
workspace. Concrete zero dimensions fail in the source width decomposition with
RuntimeError, while native admission raises ValueError before querying children.
Reversed operand widths also fail at native admission.

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
            "controlled_add", (2, 3), control_value=control_value, adjoint=adjoint
        )
        assert query_resource_logical_width(root) == 10
        assert '<TD>Qubits</TD><TD>10</TD>' in export_resource_call_graph_dot(
            root, declared_calls=arithmetic_call_graph(root)
        )
```

Users continue to append the existing controlled-add component with
`circuit.append(op)`. This resource binding adds no instructions or workspace.

## Physical counts and source model

For CAdd(2,3), both directions have four physical work qubits and four classical
slots. Either direction emits CX32, H12, T8, Tdg8, S4, measurement4,
conditional CZ4 and reset4. Control-on-zero adds X8. The separate source model
returns AND4, Clifford13 and measurement4 with connected width10. Its T
equivalent is 16. One-bit CAdd inherits the source negative Clifford count
of -2 while its real stream includes CX7. The emitted counts and model remain
separate in the batch evidence. The Add child's historical one-bit negative Clifford
formula and unequal-width accounting boundaries are retained. The frozen source
is never edited to reconcile its category model with its decomposed stream.

## Acceptance scope

Tests compare the unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, with four frozen Git blobs.
Complete normalized width caches cover every positive operand pair through
width five, a three/eight-bit pair, Sympy/mixed/symbolic widths, three encodings,
two control polarities and both directions. Nine mixed-encoding pairs at widths
two/three and seeded AND/erase/Add/cast/CNOT peaks verify child ownership.
There are 180 full-DOT comparisons over five aggregations and three depths.
Invalid-width admission/retry, unsupported casts and gate-override independence
have separate tests. Concentrated regression reuses the original per-gate and
feedback comparisons, complex amplitudes with entangled spectators, all small
measurement branches, work-zero and append workflows. Constant-add lifetimes
and general register-program bindings remain open under Q-0734.


Local acceptance passed 525 focused tests in 5.61 seconds, 4,442 concentrated
regression tests in 138.96 seconds and 87 integration tests in 17.80 seconds,
without skips. The concentrated run reports 1,112 upstream deprecation warnings
and integration reports two existing warnings. Strict typing covers four changed
Python files. Black, Ruff, team scope, architecture, internal IR, protected API,
nine Drawer doctests and the Markdown example pass. The new owner has 61
statements and 24 branches, all covered. The resource width dispatcher also has
full statement and branch coverage. Combined coverage over ten measured modules
is 99 percent. The first focused run exposed a symbolic-expression plus Python
bool signature-width error. Explicit integer control-width accounting fixes it,
and the complete symbolic matrix passes.
