# Forward clean-bank QROAM

`QroamCleanTable` takes concrete row-major scalar datasets, their common
shape and an explicit block exponent per axis. It preserves the locked
Qualtran SelectSwapQROM batch order. High address axes identify the
batch. The low axes index the words within that batch. The last batch
on each axis is zero padded to the declared block size.

`source_qroam_clean_ir` binds the caller's zero-initialized output banks
and separate And scratch. The high address goes through the original
measured QROM. The low address runs the original relative-phase exchange
network over each bank. Word zero carries the requested value and the
remaining words are retained logical outputs for uncomputation. The
phase of this network is part of the output, including negative signs
when exchanges act on overlapping one bits. The controls apply to the
QROM load, and the exchange network acts on the resulting banks. When a
control is off, the banks stay zero. The QROM restores temporary And
wires to zero for every measurement history.

The source comparison walks the complete original Soquet graph. It
binds each clean allocation by tracing the real output port back through
the QROM and exchange connections. It reads the source Partition wire
maps and decomposes the original QROM and CSwapApprox leaves. The output
oracle evaluates the source batch arrays and the signed exchange
matrices independently of the native emitter.

For a one-control, five-row table of two-bit words with two rows per
block, the circuit emits 16 T-family gates, 32 CX, two measurements and
two resets. It occupies ten wires, including two temporary And wires,
two target wires and two retained bank wires. The independent source
QEC and qubit estimates agree for this instance. The static qubit
estimate differs from the concrete allocation for some constant and
singleton cases, as in the underlying QROM.

Coverage includes one- and two-dimensional blocks, zero padding,
multiple datasets, controlled and uncontrolled lookup, and the
unblocked path. Complex branch tests include a spectator and all
in-range batch addresses, including padded rows. They project every
history for cases with up to four measurement events and two endpoint
histories for the larger case. The And channel certificate covers all
histories before the unitary routing.

The locked source fails to bind omitted high or low selectors in blocked
layouts. Native validation rejects these configurations. For example, a
fully batched four-row table loses its high selector and the source
cannot finalize the original selection output. Mixed zero and positive
block exponents can also lose a selection port.

The separate QROAMCleanAdjoint wrapper has an executable bookkeeping
graph with a resource-only leaf. The leaf raises
`DecomposeNotImplementedError`. Its padded data can lead to a different
model from the wrapper. With the five-row controlled example and
one-row erasure blocks, the wrapper counts three Toffolis and the
six-row padded leaf counts four. These estimates are tracked separately
from the accepted forward gate stream.

The dependency regression passed 138 tests. The final module run passed
14 cases at 100% statement coverage. Strict types passed for 975
production files. The locked QROAMClean and SelectSwapQROM blobs match
the inventory.

## Executable reversal of the forward stages

`source_qroam_clean_uncompute_ir` visits banks in reverse order and
uses each original SwapWithZero adjoint. It then executes the same
self-adjoint QROM to clear the loaded words. Matching signed output
banks return to zero with the source relative signs canceled. The
address, controls and entangled spectator amplitudes are preserved.

The uncompute stream matches the reversed sequence of original
self-adjoint exchange leaves followed by the original QROM gates.
Tests run each small measurement history in both directions, using
the independently signed forward state as the uncompute input. Each
branch returns the original complete vector scaled by its
unnormalized branch factor. Forward and reverse use the same work,
T-family, CX, measurement and reset counts. The five-row example
therefore uses 16 T-family gates, 32 CX and two measured And cleanups
per direction. The upstream QROAMCleanAdjoint cost remains a separate
model for its measurement-based method.

`optimal_qroam_clean_log_block_size` ports the original concrete
forward cost minimization over floor and ceiling candidates, retaining
the lower candidate on a tie. Six cases compare both the original
selector and the default block exponent of an original QROAMClean.
Callers supply the selected exponent to `QroamCleanTable`.

This batch passed 21 module cases and 146 dependency cases. Statement
coverage for the combined forward and reverse module reached 100%.
