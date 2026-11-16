# Select-swap QROM source audit

The implementation, notebook and tests match the pinned Qualtran
v0.7.0 inventory at `bcb17bb25360ed68ce3bc084645db3e929799389`.

| Inventory | Native evidence |
| --- | --- |
| Q-0483 tutorial | `register_select_swap_qrom.ipynb` covers the two concrete and four symbolic source examples, four independent formula queries, the default and explicit-block call graphs, and both bank policies through `circuit.append(op)`. |
| Q-0484 implementation | The emitter retains the source bank restoration order. The distinct immediate calls and batched QROM data are exposed independently of the aggregate QEC model. See the open boundaries below. |
| Q-0485 tests | The full original test file reported 27 passes and two expected failures for the original unenforced count checks. Native checks reuse the source gate and channel tests, and additionally compare immediate graph children and their parameters. |

## Immediate calls and execution

The two QROM calls share one identity because the source QROM is
self-adjoint. Its shape-only declaration keeps high selector widths,
control count and the bank-shaped targets. Concrete declarations use the
shared high-then-low packing, including zero padding at the final block.
XOR calls retain one target width each. A positive block exponent adds
two distinct route nodes for each target width, forward and adjoint.
Equal-width dataset contributions accumulate. A clean bank uses one
copy and one call in each route direction. A borrowed bank doubles
those three counts to cancel the original bank's contribution.

The direct comparison includes default and explicit blocks, fully blocked
and mixed-axis declarations, symbolic block sizes, repeated widths and
both bank policies. The executor continues to use the existing measured
QROM and signed route streams. Existing tests verify the original
operations, operands and feedback, the complete complex channel across
small measurement histories, arbitrary nonzero destinations, a
spectator correlated with borrowed bank data, and zero work. The new
tutorial exercises enabled and disabled controls in one coherent input.

## Count comparison

Both tutorial modes use the same four-row table, a block length of two
and seven wires. The clean mode emits 16 T/T†, 29 CX and two
measurements. The borrowed mode emits 24 T/T†, 40 CX and two
measurements. The source model reports eight explicit T and two AND
for the clean mode, and 16 explicit T and two AND for the borrowed
mode. Both models declare two measurements and a seven-qubit peak.
The Clifford model counts are 51 and 94 respectively.

## Remaining Q-0484 boundaries

The source can declare mixed or fully blocked call graphs, while its
executable graph loses a required low or high selector in these cases.
This audit verifies the declarations independently and keeps the
existing execution rejection. Shaped target arrays are outside the
original SelectSwapQROM interface. As in the original generic control
path, QROM lowering that would require a controlled non-through AND or
AND adjoint has no supported decomposition. Circuit diagrams use the
native renderer, with the original Cirq spacing covered by the
unchanged reference test.

The 17-cell reference notebook and three native lookup tutorials ran
in independent kernels. The native regression passed 265 tests with
no skips. Strict typing, formatting, team scope, architecture, IR
contracts and public API checks passed.

See [gate and branch validation](REGISTER_SELECT_SWAP_QROM.md).
