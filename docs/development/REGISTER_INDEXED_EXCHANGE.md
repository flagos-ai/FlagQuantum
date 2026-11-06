# Relative-phase exchange and indexed routing

The shared register compiler ports the concrete Qualtran v0.7.0
`CSwapApprox` and `SwapWithZero` decompositions. Both emit native
Clifford+T instructions on caller-provided wires. They use zero scratch
qubits, zero measurement events and zero classical slots.

`relative_controlled_swap_ir` accepts one control and two equal nonempty
words. It applies the source rotation and CNOT layers around a dirty-target
fanout. This conjugation is self-adjoint and retains the original relative
signs. Sign extension and both modular negation emitters use the same
`bitwise.fanout_ir` helper.

`swap_with_zero_ir` accepts big-endian selection words, axis extents and
row-major target words. It visits the child subtrees before each axis'
exchange layers. The selected row arrives at row zero for in-range
addresses. Its adjoint reverses the exchange sequence and applies the same
self-adjoint relative-phase exchange at each edge. The source omits zero-bit
selection registers and fails to bind a later axis with actual exchanges.
The native entry point rejects that configuration explicitly. Truncated
axes, trailing zero-bit singleton axes and an identity singleton are
covered by source comparisons.

Gate tests read the locked Cirq decomposition of each relative-phase
exchange and the original SwapWithZero Soquet connections. Complex tests
compare every output amplitude of a dense input entangled with a spectator,
including padded addresses of truncated axes, and verify inverse restoration
at absolute tolerance 2e-12. The largest primitive exchange checked has
eight bits per word. The network checks cover one, two and three axes.

| Scope | Literal T + T-dagger | Literal CX | Source T model | Total qubits |
| --- | ---: | ---: | ---: | ---: |
| One exchange of eight-bit words | 32 | 47 | 32 | 17 |
| Six two-bit words on a 2-by-3 grid | 40 | 55 | 40 | 15 |

For word width b and N target rows, the network emits N-1 exchanges.
Each contributes 4b T-family gates, 6b-1 CX and 16b single-qubit Clifford
gates. The source QEC and qubit models are checked independently of the
instruction scan.

The local dependency regression passed 249 tests, including 15 new cases
and the paired lookup measurement tests. Both new emitters have 100%
statement coverage. The locked blobs for `cswap_approx.py`,
`swap_with_zero.py` and `multi_target_cnot.py` match the inventory.

## Generic adjoint and outer controls

The standalone adjoint of `relative_controlled_swap_ir` follows the source
`Adjoint` graph. It reverses the dependencies and conjugates the phase leaves.
`SwapWithZero` has its own adjoint rule and continues to reverse its network
while applying forward relative-phase swaps at its leaves.

`control_relative_swap_ir` applies the source generic outer control to the
forward or adjoint block. Before the outer control is threaded through, the
source imports its independent rotations in moment order. The native emitter
uses the same order and keeps Toffoli, controlled Hadamard and exact controlled
S/T phases as leaves. A negative outer control adds X conjugation. A multi-bit
outer predicate uses clean work and a classical slot per extra control, then
measures and clears them in reverse order.

For three-bit words and an outer pattern of `(1, 0, 1)`, the IR has 17 CCX,
24 CH and 36 controlled phase instructions. Predicate computation contributes
eight T/T† and two measurements, with two clean work qubits. The pinned
`QubitCount` is 12. The pinned `QECGatesCost` fails for the controlled abstract
`ArbitraryClifford` in the source cost graph. The eight emitted T/T† count only
the predicate, and the other controlled leaves retain their own opcodes.

## Outer control on the indexed network

`control_swap_with_zero_ir` uses the same row-major traversal and relative
swap leaves as the regular network. The outer predicate is computed once
for the entire exchange sequence, so the number of outer work qubits and
measurements is independent of the table size. For the network adjoint,
the exchange sequence reverses and the same forward controlled leaf is
used at each exchange. The empty network retains the source's negative
control conjugation or measured multi-control wrapper.

With three rows of two bits and outer pattern `(1, 0, 1)`, the two exchanges
emit 22 CCX, 32 CH and 48 controlled phase instructions. The outer predicate
adds eight T/T† instructions and two measurements, returning both work
qubits to zero. The source peak is 13 logical qubits. Its QEC gate query
encounters the same controlled `ArbitraryClifford` limitation as the leaf.

## Multiplexed exact exchange

`multiplexed_cswap_ir` exchanges the addressed row with a caller-owned
output word. The bounded traversal is shared with QROM. Selection words
are big-endian, the row layout is row-major, and all targets and the
output can hold coherent data. Any initial controls are positive.
Traversal qubits and classical slots are caller-owned and reused
between siblings. All work qubits return to zero.

The source defines the selected exchange on in-range inputs. Tests
also compare padded addresses to its actual gate stream, including
cases where a skipped high selection bit changes the padded behavior.
An empty selection axis and a singleton first axis with no control
are rejected because the pinned traversal cannot decompose those
cases.

For three two-bit rows and no explicit control, the exact stream emits
46 T/T† and one measurement with one work qubit. The source model
declares six CSwap, one And, four Clifford and one measurement with
peak width 11. Its default conversion gives 28 T equivalents. The
source's seven-T Fredkin conversion and its legacy T complexity both
give 46.

### Shape-only estimate for multiplexed exchange

Pass `source_multiplexed_cswap_cost` to the shared resource estimator
to price the operation before constructing its circuit.

```python
from flagquantum.compiler.registers.multiplexed_swap import (
    source_multiplexed_cswap_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

cost = source_multiplexed_cswap_cost((5, 7), (5, 6), 2, 4)
logical = estimate_ftqc_resources(cost).logical
assert logical.peak_logical_qubits == 164
assert logical.gates.total_t_count() == 700
assert logical.gates.total_t_count(ts_per_cswap=7) == 1120
```

The model counts one CSwap per row and word bit, one measured And per
bounded traversal fork, and the optional control reduction. Peak width
includes all declared input/output ports and the longest live And path.
Only the significant bits of each axis contribute to that path. For the
example above, the caller reserves 12 work wires while seven contribute
to the source peak. The estimate runs in time proportional to the number
of axes and does not emit instructions.

### Standalone adjoint of multiplexed exchange

Pass `adjoint=True` to reverse the source component traversal. A split
first computes its positive And and visits the right half. After the
predicate toggle and left half it erases with the original negative
polarity, using the same measured feedback and clean work wire. In the
uncontrolled root, the right half moves before the address flips. Each
swap uses the same exact self-adjoint Fredkin leaf.

The adjoint has the same source QEC gate categories, T-equivalent counts
and peak width. Its tests compare the locked adjoint graph's gate
operands and feedback, and verify round trips through separate forward
and inverse measurement histories across the entire address register.

### Generic outer control on multiplexed exchange

`control_multiplexed_cswap_ir` follows the pinned generic control graph
for two rows without a prior control, or one row with a prior positive
control. A positive outer predicate controls each register CSwap and
each root address flip. The adjoint reverses the root traversal. Each
controlled register CSwap reuses one measured scratch bit and slot
across its word pairs. Additional outer bits share a predicate computed
once around the network.

Other layouts need controlled And or controlled And-adjoint leaves with
non-through ports, which the pinned source cannot decompose. The emitter
rejects these layouts explicitly.

For two two-bit rows and outer pattern `(1, 0, 1)`, the exact stream
emits 52 T/T† and six measurements with three work qubits. The source
model reports four CSwap, six And, twelve Clifford, six measurements
and peak width 13. Its default T-equivalent count is 40 and its
seven-T Fredkin conversion is 52.
