# Out-of-place addition output lifetimes

The arithmetic domain owns `out_of_place_addition_structure.py` and
`out_of_place_addition_width.py`. The former supplies one literal logical
component sequence to the existing primitive emitter and the connected width
owner. The latter accepts the existing `ArithmeticCall` and recursive child
width callback. Compiler's resource-width dispatcher binds that owner, including
Drawer's automatic width column. The existing append workflow remains
`circuit.append(op)`.

## Interface, order and ownership

Inputs `a` and `b` are equal positive-width big-endian words. Output `c` has
`n + int(include_most_significant_bit)` bits. Forward creates the sum, and the
explicit adjoint consumes its correct value. Inputs remain caller-owned. The
lowest output bit is allocated on first use, while every AND produces its next
output bit. Adjoint AND consumes that bit through the existing measurement
feedback decomposition, and the low bit is freed. There is no separate quantum
work register. On physical execution, callers provide the clean output wires
and the adjoint's classical slots.

Views follow first operand use. Scalar words stay atomic. Both directions use
the existing stable connection scheduler, preserving bystander widths when a
cached child has a larger local peak. Cache keys normalize bookkeeping dtypes
to concrete scalar widths, as in the previous owners. Dtype-specific cache
seeds, dirty/array allocation and general register-program inference remain
open. Python zero width raises a native admission ValueError beside the source
RuntimeError. Sympy integers and symbols use the original signature/callee
fallback, including zero-count visitation and possible workspace underestimates.

The locked source's `.adjoint()` retains the truncation flag, while `** -1`
drops it. Tests record both behaviors against the unchanged source. The native
explicit adjoint retains the requested output width.

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

for include_msb in (False, True):
    for adjoint in (False, True):
        call = ArithmeticCall(
            "out_of_place_add", (3,), adjoint=adjoint,
            include_most_significant_bit=include_msb,
        )
        expected = 9 + int(include_msb)
        assert query_resource_logical_width(call) == expected
        assert f'<TD>Qubits</TD><TD>{expected}</TD>' in export_resource_call_graph_dot(
            call, declared_calls=arithmetic_call_graph(call)
        )
```

## Separate emitted and modeled resources

Three-bit inputs produce these measured primitive counts. Conditional CZ is
listed separately from the source category model. Output wires are logical data
until the adjoint clears them. Additional quantum work is zero in all rows.

| Output | Direction | Actual primitive stream | Classical slots |
| --- | --- | --- | --- |
| Truncated, 3 bits | Forward | CX24, H4, T4, Tdg4, S2 | 0 |
| Truncated, 3 bits | Adjoint | CX12, H2, measure2, conditional CZ2, reset2 | 2 |
| Full, 4 bits | Forward | CX33, H6, T6, Tdg6, S3 | 0 |
| Full, 4 bits | Adjoint | CX15, H3, measure3, conditional CZ3, reset3 | 3 |

| Output | Direction | Source category estimate | Logical peak |
| --- | --- | --- | --- |
| Truncated | Forward | AND2, Clifford12, measurement0, T equivalent8 | 9 |
| Truncated | Adjoint | AND0, Clifford14, measurement2, T equivalent0 | 9 |
| Full | Forward | AND3, Clifford15, measurement0, T equivalent12 | 10 |
| Full | Adjoint | AND0, Clifford18, measurement3, T equivalent0 | 10 |

## Acceptance scope

Reference Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389` remains unchanged. Eight frozen blobs
cover addition, Cirq import, register decomposition, graph iteration, allocation,
free and QubitCount/AND ownership. Tests compare complete normalized caches and
insertion order for both output modes and directions, Python widths one through
five/eight, Sympy literals, symbols and symbolic zero. Seeded child peaks cover
CNOT, both AND directions, allocation/free and input/output views. There are
240 complete DOT comparisons over five aggregations and three depths. Gate
declaration overrides retain the connected decomposition. Zero failures leave
an empty cache and allow retry.

Execution regression compares the full primitive operands and feedback stream
against the source for widths 1, 2, 3, 5 and 8. Coherent full/truncated addition
and erasure for widths 1, 2 and 3 preserve complete complex amplitudes and clear
every output wire. Each small measurement history occurs in the accepted
trajectory run. Shared logical sequence extraction preserves the physical stream.
The width binding itself adds zero gates, work qubits or measurement slots.

An initial unsupported-operation fixture supplied a two-width tuple to the
single-width less-than identity. It was corrected before acceptance. Ruff also
required an explicit non-strict zip for the forward path, whose classical slots
are empty by contract.

Local acceptance passed 878 focused tests in 3.34 seconds, 7,522 concentrated
regression tests in 109.21 seconds and 87 real-kernel/CPU integration tests in
15.80 seconds, without skips. Regression reports 1,112 upstream deprecations
and integration two existing warnings. Strict typing covers six changed Python
files. Black, Ruff, team scope, architecture, internal IR, protected API, nine
Drawer doctests and the executable Markdown example pass. Shared structure
has nine statements/four branches, the lifetime owner 60/34 and the existing
emitter 25/14, all fully covered. The resource-width dispatcher also has full
statement/branch coverage. Sixteen measured modules have 99 percent combined
coverage. The generic Q-0734 parent remains open for further register bindings.
