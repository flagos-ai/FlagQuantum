# First-quantization PREPARE and SELECT audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, under
`qualtran/bloqs/chemistry/pbc/first_quantization`.

| Inventory | Source file | Git blob |
| --- | --- | --- |
| Q-0388 | `select_and_prepare.py` | `5f5001c60acedf729fc4c4122431494047fb2f2e` |
| Q-0389 | `select_and_prepare_test.py` | `382ab4865f058124d6975b4ee84e63a40d131259` |

## Register and resource contracts

The algorithm owns the ordered PREPARE, SELECT, and momentum exchange
signatures. The models and the tutorial reuse these declarations.
Tests compare register encodings, iteration domains, shape, direction,
and order. PREPARE has a two-bit `w` with three values and preserves
the source nuclear-index iteration length equal to its bit width.
SELECT has a three-bit `w`, a full-width nuclear index domain, and a
system bank shaped by particle and axis.

PREPARE passes `plus_t` through untouched and binds the three
momentum-transfer words to the UV child's three axes in both directions.
SELECT schedules four momentum exchanges and tracks live momentum and
nuclear-position work around the two selection children. Its one-bit
momentum instance reports 101 Toffoli for the fixture with two particles,
two atoms, and two nuclear-position bits. The reference peak query fails
when it constructs the one-bit signed conversion dtype.

For the native tutorial's six-bit, ten-particle fixture, PREPARE
estimates 485 Toffoli, 21 And, 21 measurements, and 81 peak logical
qubits. SELECT estimates 576 Toffoli, 720 controlled swaps, 68 And,
68 measurements, and 298 peak logical qubits. These are source model
categories. The executable exchange tests separately count emitted
T and T-adjoint gates, measurements, and resets, and compare gate
operands and feedback with the source decomposition.

## Executable scope and remaining work

The momentum exchange runs through `circuit.append(op)`. Both directions
retain the source selector domain and the particle and axis order.
The execution cases cover nonzero exchange junk, arbitrary wire
placement, all small-fixture measurement histories, complex amplitudes,
and clean temporary work. The source fails to decompose a zero-width
selector. PREPARE also retains the source composition boundaries for
one particle or a one-value uniform register.

The unchanged upstream tests returned eight passes and four skips,
with the notebook test deselected for independent execution. The
original 58-cell tutorial and the native tutorial both ran independently.
The native concentrated regression passed 190 cases without skips.
This closes Q-0389 in the runnable reference scope.

Q-0388 remains open for constructor-level normalization metadata and
end-to-end physical PREPARE and SELECT. The locked TUV, uniform pair,
and SELECT T/UV leaves supply declarations without executable
decompositions. The scheduling and resource comparisons therefore
cover the source composite without giving it an emitted gate count.
