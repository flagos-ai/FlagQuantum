# Elliptic-curve phase estimation

The frozen reference is Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, files
`qualtran/bloqs/cryptography/ecc/ec_phase_estimate_r.py` and its test file.

The two through ports preserve the unsigned source metadata. The symbolic
and concrete source declarations group the addition at the supplied point
with multiplicity `n // window` and call MeasureQFT once. The exact source
order starts with `n` positive-superposition bits. For each group of size
`window`, it advances the point by `2**(i * window)`, giving that group
a big-endian address. The measurement receives the original groups in
concatenation order. Both input and returned control operands were checked
against the locked graph for width 4 and window 2, with two distinct finite
point multiples. The native operation exposes the ordered final classical
slots and keeps them separate from all arithmetic feedback.

On the clean domain of the windowed circuit, one `circuit.append(op)`
prepares the supplied phase wires, applies the native window operations
using shared clean scratch and reads out through the textbook QFT. The
full emitted prefix equals the source operand ordering applied through
the verified window binders. The test compares the complete native state
against the source window coordinate maps followed by an independent
positive-exponent DFT, including each terminal result's projected
amplitudes and work wires. The two checked settings expose all 24
terminal branches on a coherent combination of infinity and a curve
point. Shared arithmetic and erasure components have independent
branch regressions. The width-4 fixture for the amplitude test has
an identity in its second group; the separate order test uses two
nonidentity window points.

| Width, modulus, window | Native emitted operations | Locked gate-count model |
|---|---|---|
| 3, 3, 3 | 7,688 T/T†, 19 CCX, 3 CP, 1 SWAP, 1,346 measurements; 77 bound wires | 1 T, 21 Toffoli, 336 CSwap, 1,331 AND, 8,816 Clifford, 1,334 measurements |
| 4, 3, 2 | 25,744 T/T†, 48 CCX, 6 CP, 2 SWAP, 4,336 measurements; 97 bound wires | 2 T, 48 Toffoli, 1,216 CSwap, 4,314 AND, 28,247 Clifford, 1 rotation, 4,318 measurements |

The native count includes the bank readout and concrete per-window
traversal. The source declaration multiplies one representative child
model by the number of windows and prices the lookup adjoint abstractly.
It also prices the Fourier operation through phase-gradient blocks. The
source concrete qubit-count query fails because MeasureQFT leaves its CBit
results unused. The native circuit supplies explicit result destinations.

The scalar fixed-point child has no executable decomposition or gate-count
model in the locked version. Its source declarations and powers of the
point are represented. The native operation accepts the executable
windowed domain. The curve-addition boundary also persists: for
`y^2=x^3+2x+1` modulo 3, width 4, window 2 and base point `(0,1)`,
the source staged simulation rejects four of 16 address words from
infinity at a clean release. A shared scratch binding assumes all its
branches meet the source child release promises.

The original test file ran with four passes, three skips and three
expected failures. Native tests cover both original example declarations,
the symbolic grouped call graph and the new windowed execution
evidence. Q-0462 closes. Q-0461 remains open for the opaque scalar path,
concrete source qubit-count failure and inherited clean-domain boundaries.

## Parameter-preserving model calls

The operation's `source_call_graph()` now retains its own constructor identity
and complete child keys. `ECAddR` carries width and the supplied point;
`ECWindowAddR` additionally carries both window sizes. `MeasureQFT` carries its
width. The existing module-level grouped graph remains available. The model
uses immutable parameter tuples for terminal calls, following the RSA model,
so a symbolic modulus does not trigger physical arithmetic validation.

The representative point remains the supplied point, as in the original
`build_call_graph`; concrete execution still uses the successive point powers.
The parameter regression compares unchanged source declarations across both
addition modes, two multiplication windows, and a nondividing width/window
model case. A terminal-cost consumer verifies that distinct multiplication
windows remain distinguishable. These declarations are model counts. They do
not certify executable scalar addition or supply missing leaf gate estimates.
The physical amplitude, workspace and resource evidence above still applies.

Batch validation on 2026-11-21 passed 44 concentrated cases and 29 integration
cases, strict typing for all three changed Python files, formatting, Algorithms
scope, architecture, IR and public API checks. The Cirq 1.6.1 lane passed all
20 selected model and physical phase cases with the native Kickmix extension
present.
