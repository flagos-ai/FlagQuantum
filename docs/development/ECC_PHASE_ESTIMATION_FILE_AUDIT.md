# Elliptic-curve phase estimation

## Original constructor and direct dependency closure

The complete `ec_phase_estimate_r.py` constructor now resolves through owned
phase signatures, point arithmetic, PlusState, fixed/windowed point addition
and the shared original MeasureQFT declaration. Compiler owns only ordered
register composition; Ecosystem translates foreign fields. Scalar controls are
passed individually; window controls retain MSB-first grouping, ascending
`2**(window*i)` point powers and the original concatenation at the terminal.
The scalar branch ignores multiplication-window fields as the source does.

Original shallow libraries at depths zero and one round-trip and resolve with
SDK imports blocked. Concrete three-bit scalar and window-three roots have
seven and five calls respectively. The original symbolic-width example remains
a declaration because the source refuses decomposition. Empty or unknown-size
control paths remain declarations rather than claimed finite execution.
Nondividing three-bit/window-two source decomposition raises IndexError;
the native finite builder rejects that domain with ValueError. Neither path
silently partitions, pads or truncates controls.

The original concrete depth-two library fails during MeasureQFT finalization
because measurement CBits are unused. Native resolution retains a bodyless
terminal and the scalar ECAddR declarations. Whole physical lowering, primitive
counts and complex channel are unavailable, not zero. A windowed root also
retains the original model-only QROAM erase leaf. The existing Algorithms
readout workflow and its previously verified 24 terminal branches remain
separate from this source-shaped root.

An actual stored window child is extracted without changing consumers and
explicitly supplied with certified native measurement erasure. All 32 small
address/curve columns, absolute complex amplitudes, applicable raw histories,
returned layouts and private work zero are checked, with two independent sparse
executions and SDKs blocked. The complete window proofs and original gate stream
receipts are reused; no original erase body is invented.

The directly shared MeasureQFT constructor was moved out of the RSA-specific
adapter into `_fourier_measurement_library.py`. Three complete old declarations
(empty, concrete and symbolic width) and one entire RSA Program compare equal
to published `dec49fcad`. The default source identifiers remain unchanged.
The window's earlier supplemental Cast wrapper was removed: existing source
type compatibility already admits all five direct connections. Entire physical
IR, outputs and pools compare equal at address widths one and five.

Q-0461 remains open for scalar, terminal, symbolic and inherited curve boundaries.
No source file, source callback, protected contract or readout API was changed.

| Scope, n=3, modulus=3, window=3 | Actual instructions | Actual T/T-dagger / CCX / measurements | Physical pool | Source default T estimate / measurements / qubits |
|---|---:|---|---:|---|
| Selected stored window child with explicit native erasure | 36,607 | 7,688 / 19 / 1,463 | 61 | 6,740 / 1,328 / 61 |
| Whole source-shaped single phase | Unavailable | Unavailable | Unavailable | 6,753 / 1,334 / source unused-CBit failure |
| Whole source-shaped double phase | Unavailable | Unavailable | Unavailable | Source generic-point integer-conversion failure; qubit query also fails |

The selected child's source model contains 21 Toffoli, 336 CSwap and 1,328 AND
calls; these are model categories, not emitted CCX/T counts. The full phase
model additionally prices its Fourier terminal. The two scopes and both count
conventions remain separate. Scalar phase gate cost also retains the original
ECAddR decomposition failure. Complete records are in the phase resource receipt.

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
