# Two-bit fermionic Fourier transform

This audit uses the locked Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Ledger | Source file | Git blob | Coverage |
| --- | --- | --- | --- |
| Q-0645 | `two_bit_ffft.ipynb` | `2f4ba8f66b47b01acf6a09e5f1725bfac72729ae` | Original `F(2, 3)` example, graphical signature, direct graph and native append |
| Q-0646 | `two_bit_ffft.py` | `a707f9c8025443e291933c0d43d6f45f7a4f9e58` | Signature, symbolic or concrete rotation precision, analytical tensor and ordered forward gates |
| Q-0647 | `two_bit_ffft_test.py` | `1fd5edf1be77b20570c8494cdca673137f20fa94` | Original operand grid, analytic tensor and call-graph assertions |

The two ordered occupation ports `x` and `y` are through qubits. The
algorithm emits the existing 15 instructions and its mathematical inverse.
The numerical callback separately exposes the exact 4-by-4 matrix in the
source tensor convention. With tensor axes arranged as outgoing `x, y`
followed by incoming `x, y`, the locked tensor callback applies the
conjugate transpose when its adjoint flag is set.

The circuit's forward global phase relative to the analytical matrix is
`exp(-i*pi*k/n)`. The native inverse conjugates the phase. Complete
complex-state comparisons through `Circuit.append` check every matrix
column for the original 12 parameter pairs, in both directions. No
working qubits, measurements or classical feedback are needed. Existing
fixed-wire tests separately check the source ordering and operands,
including layout reversal. The original source's flagged decomposition
continues to emit its forward stream, so Q-0646 remains open for that
adjoint defect.

The declared graph retains the leading `Rz` angle and supplied synthesis
epsilon, three S calls, six Hadamards, two T calls and three CNOTs. Source
grouping combines the T and adjoint T calls. For `F(2, 3)`, emitted
primitive counts are six H, three CX, three S, one T, one adjoint T and
one Rz. The source logical model reports two T, 12 Clifford and one
rotation. An explicit synthesis model would be needed to convert the
remaining rotation into T gates. The Hubbard consumers continue to use
the existing shared emitter and cost query.

All five locked notebook cells ran. The native notebook completed six
code cells and two SVGs in a clean kernel with reference imports blocked.
The source tests passed 17 cases, including the call graph check and the
full original matrix grid. Its generic count autotest is an expected
failure because upstream does not enforce it. Q-0647 remains open for
that test gap. The native tests verify the explicit count graph and both
analytical tensor directions. The 26 focused tests, 18 Hubbard consumer
tests, and 5,546 concentrated regressions passed. The only concentrated
warning comes from the original symbolic approximate-QFT helper. Strict
typing, format, both ownership scopes, architecture, internal IR, and
public API checks passed.
