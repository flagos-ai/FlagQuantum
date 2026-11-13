# Hamming count-tree resource connections

The bits domain owns `hamming_weight_width.py`, binding the existing
`ArithmeticCall("hamming_weight", ...)` to recursive width queries and Drawer.
`hamming_weight_structure.py` owns the big-endian register traversal.
`bit_compression_structure.py` owns the six logical sum/carry components.
The existing emitters share both sequences and retain their measured AND erasure.

## Interface and ownership

The width owner accepts the arithmetic identity and a child-width callback.
Forward computation receives the caller's input word, creates each junk bit at
its AND and allocates output bits on first use. Input, logical junk and output
leave the forward graph together. The adjoint consumes the complete computed
triple, restores input and consumes junk/output. The generic source adjoint
reverses the forward greedy traversal before assigning inverse instance order.
The width graph follows this order, including split/join and allocate/free.

Python widths from two upward use connected decomposition. Sympy integers and
symbols retain signature/callee fallback, including zero multiplicities and the
source worst-case population bound of one. Python zero/one fail before visiting
children, matching the source empty-register boundary. Native admission uses
ValueError beside the source RuntimeError. Existing one-bit executable support
continues to have separate acceptance evidence. A failed root never enters the
cache, and a later valid query can reuse it.

Bookkeeping identities normalize dtype to width. Equal-width QAny/QUInt views
collapse in the normalized cache. Arbitrary dtype-specific seeded overrides,
dirty/array allocations and general register/control inference remain open.
Symbolic fallback retains possible simultaneous-workspace underestimation.

## Executable example

```python
from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.resource_width import query_resource_logical_width
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import arithmetic_call_graph
from flagquantum.drawer.call_graph import export_resource_call_graph_dot

for adjoint in (False, True):
    call = ArithmeticCall("hamming_weight", (8,), adjoint=adjoint)
    assert query_resource_logical_width(call) == 19
    assert '<TD>Qubits</TD><TD>19</TD>' in export_resource_call_graph_dot(
        call, declared_calls=arithmetic_call_graph(call)
    )
```

Users keep `circuit.append(op)`. Existing execution tests compare ordered source
primitives, operands and feedback, complete complex amplitudes, every forced
measurement history for one through five input bits and clean junk/output after
the compute/erase pair. The shared compression consumers receive concentrated
regression covering integer products, squares, accumulation, fixed-point
functions and contiguous indexing.

## Separate emitted and modeled resources

Eight input bits declare seven logical junk bits and four output bits, giving
logical peak 19. Extra scratch is zero. The forward stream emits CX78, H14,
T14, T-dagger14 and S7. Erasure emits CX36, H7, measurement7, feedback CZ7 and
reset7. The source forward category model separately declares AND7, Clifford36,
measurement0. Its adjoint declares Clifford43 and measurement7. These categories
retain their source interpretation. The width binding introduces zero extra
gates, scratch or measurements.

## Acceptance scope

Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389` remains
unchanged. Five frozen blobs cover Hamming, generic adjoint, composite graph,
greedy scheduling and QubitCount. Full normalized caches and insertion order
match in both directions for Python widths 2, 3, 4, 5, 6, 8 and 16, Sympy
literals and positive/nonnegative expressions. Seeded child peaks cover CX,
AND, views and allocations. Fourteen complete weighted connection graphs match
source topology, and 120 full DOT comparisons cover five aggregations and three
depths. Declaration overrides retain connected ownership. Zero/one failure,
empty cache and retry are tested.

Local acceptance passed 237 new tests in 2.22 seconds, 8,646 concentrated tests
in 124.48 seconds and 87 real-kernel/CPU integration tests in 10.91 seconds,
without skips. Regression retains 1,112 upstream deprecation warnings and
integration two existing warnings. Strict typing covers seven changed Python
files. Formatting, lint, scope, architecture, internal IR and protected API
checks pass. The first comparison exposed a missing input-side symbolic maximum,
which was corrected. A resource-report fixture was corrected to read fields
from the source attrs record. Reference files were preserved throughout.

The width owner covers all 74 statements and 36 branches. Both shared structure
modules, the compression emitter and the width dispatcher have full statement
and branch coverage. The existing Hamming emitter reports 94 percent combined
coverage, with two earlier invalid resource-width branches outside this batch.
The 22 measured modules report 99 percent combined coverage. Nine Drawer doctests
and the Markdown example pass.
