# Conjunction and exact controlled-swap width bindings

Compiler's native width query now delegates concrete conjunction connections to
`registers/control/multi_and_width.py` and exact controlled-swap connections to
`registers/primitives/controlled_swap_width.py`. Each owner reuses existing
`WidthStep`, `WidthConnection` and `estimate_connection_width`. The And owner
also reuses `multi_and_stages`, the stage generator used by the native emitter.
The resource query retains recursive cache ownership and Drawer uses the
already supported automatic width path.

## Example

```python
from flagquantum.compiler._component_trace.resource_generalization import (
    ConjunctionCall, ControlledSwapCall,
)
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.drawer.call_graph import export_resource_call_graph_dot

predicate = ConjunctionCall((0, 1, 0, 1))
assert query_resource_logical_width(predicate) == 7
assert query_resource_logical_width(ConjunctionCall(predicate.controls, True)) == 7
swap = ControlledSwapCall(3)
assert query_resource_logical_width(swap) == 7
assert '<TD>Qubits</TD><TD>7</TD>' in export_resource_call_graph_dot(swap)
```

## Interfaces and ownership

`multi_and_width(call, child)` builds forward edges from the existing ladder
stages. Controls enter in declared big-endian order. Each stage emits its
controls and a logical prefix output. Prefix outputs remain owned by the
caller through the right boundary. Adjoint decomposition reverses those same
edges and visits adjoint stages in reverse order, consuming prefix outputs.
Binary And is atomic and uses width three. No physical primitive stream is
reconstructed for width counting.

The exact CSwap binding splits both words, threads the control through one
Fredkin per bit pair, then joins both output words. There is no workspace.
Concrete positive Python widths use those connections. Symbolic widths use the
original signature/callee fallback. Zero-width native exchange raises ValueError
at admission. The source width query also fails, wrapping its decomposition
error in RuntimeError. Source zero-width decomposition remains unchanged.

Unknown conjunction counts use the source fallback maximum of the full
signature and binary And child width. A concrete count represents all-positive
controls, whose cached binary child and root width match the normalized source.
Unsupported multi-control symbolic polarities remain explicit. A symbolic
fallback may underestimate simultaneous workspace.

Width decomposition visitation is independent from gate-category visitation.
Negative-polarity ladders cache original binary And polarities. A gate graph may
instead expose generalized positive Ands and explicit X costs. Those X nodes
receive no width row unless visited by the width query. Explicit gate-declaration
rows preserve the native decomposition's ownership.

The source exact CSwap is self-adjoint. Native resource keys retain their
existing direction field for the root while using identical exact child gates
and connections. This is an identity adaptation. Source comparisons translate
source object identities only at the test boundary.

## Acceptance

The pinned reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Tests verify unchanged Git blobs for
QubitCount, `and_bloq.py` and `swap.py` against the immutable inventory.

Tests exhaust every binary polarity for two through five controls in both
directions, comparing root widths, complete caches and cache insertion order.
Exact swaps cover Python and Sympy integer widths and symbolic widths in both
directions. Native automatic drawing matches source-computed width tables
across five aggregations, three display depths and seven component families.
Seeded larger child costs prove that live bystanders contribute to the peak.
Declaration ownership, zero-width failures and unsupported inputs are tested.

The reference equality covers width models and sparse cache visitation.
Existing concentrated regression separately verifies emitted gate order,
complete complex amplitudes, measurement branches, clean workspace and
`circuit.append(op)`. This batch adds zero primitive gates, physical work qubits
and measurements. Executed exact Fredkin remains CX8/H2/T4/Tdg3 with zero work
and measurements. The QEC model's four-T equivalent and the terminal seven-T
count retain their distinct accounting.

Approximate CSwap width still requires fanout identity and adjoint binding.
Its existing explicit width-provider workflow remains supported. Automatic
arithmetic and general register-program bindings remain open under Q-0734.
