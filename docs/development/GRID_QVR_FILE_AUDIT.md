# Grid quantum variable rotation file audit

Q-0431 and Q-0432 cover the rotation declaration and its tests in Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`. The unchanged original test suite returned five passes and three skips. Native comparisons include the original 8-, 16- and 32-bit model cases and the 12-bit documentation example. This closes Q-0432.

The locked source declares a scalar opaque through word named `phi`. For width `w`, its call graph contains a single freshly allocated symbolic `Rz(theta)` child with multiplicity `w`. The native cost declaration records the opaque port width, one child of the same gate class and multiplicity, and the logical peak. These resource tests also verify that consecutive calls to the original allocator produce distinct symbols.

The native grid kinetic and potential append paths share an exact phase emitter. The emitter takes the big-endian `phi` wires and a concrete coefficient in radians per integer unit. It applies `phase(coefficient * 2**bit)` to each wire in order from least to most significant. It needs zero work wires and emits no measurement or feedback. Tests use a permuted three-wire binding and compare all eight complex state entries for positive, negative and zero coefficients. That example emits three phase instructions; the independent 12-bit source budget estimates 12 generic rotations.

The locked source raises `DecomposeNotImplementedError`. Its call graph leaves the angle unbound and supplies no ordered primitive sequence. The native phase convention includes the absolute phase on the register, while an individual `Rz` has a different global phase convention. The cost declaration captures the class and multiplicity, but does not retain the allocator's symbol identity. At the first audit, source symbol identity and custom wire labels were still missing. The complete-file follow-up below supplies both. A primitive stream remains unavailable in the unchanged source and is recorded as such, rather than invented.


## Source binding contract before the complete-file follow-up

The original root owns one named scalar THRU opaque phi word. Compiler owns
its width-only declaration, actual source constructor and fresh Rz(theta)
model child. The existing SymbolAllocator and RotationCall carry symbol
identity and default synthesis precision. Algorithms retains the positive-width
concrete cost and independently supplied radians-per-integer-unit phase emitter.
Drawer owns the literal root text and ordinary phi port box. No work, classical
feedback or numerical coefficient is implied by the bodyless source root.
The source allows logical zero/symbolic widths; these reuse existing opaque
Program declarations without changing concrete convenience admission.


## Complete five-definition source closure

| Frozen definition or asset | Native owner and evidence |
|---|---|
| QuantumVariableRotation and phi_bitsize | Immutable Compiler GridVariableRotationDeclaration retains the width without introducing a phase coefficient. |
| QuantumVariableRotation.signature | Its signature preserves the named scalar opaque THRU phi, including zero and symbolic logical widths. Existing positive-width Algorithms signature delegates to this owner. |
| QuantumVariableRotation.wire_symbol | Drawer grid_qvr_wire_symbol retains literal e^{i*phi} root text and phi TextBox; complete original graph geometry, JSON and rendered text match at widths 0/1/12. |
| QuantumVariableRotation.build_call_graph | direct_calls uses the existing SymbolAllocator and RotationCall: one fresh theta each query, Rz opcode, default 1e-11 precision and original width multiplicity. Three consecutive shared-scope queries and fresh query scope match source symbols exactly. |
| _qvr | The actual 12-bit factory binds original source libraries at depths 0/1/2 to the same typed declaration. |
| _QVR | Original class/example association and frozen source/test blobs are checked. |

The native constructor factory preserves source wire metadata and rejects an
injected numerical coefficient or incompatible nominal port. Every root remains
bodyless: the physical binder refuses it, just as the original supplies no gate
recipe. Native declaration, model and constructor binding run with reference
SDK imports blocked. Qualtran 0.7.0 and real Kickmix 0.1 Simulator are explicitly
verified in the source lane.

Generic model queries now support symbolic and zero widths through the same
declaration. QEC categories and QubitCount match the unchanged source. The
zero-width root has zero rotations but peak one: the original bodyless fallback
still visits its one-qubit child despite zero multiplicity. The symbolic peak
is Max(1,n). These are source estimates, not physical allocation assertions.
Positive-width concrete cost helpers and their admission remain unchanged.

## Execution and accounting

The independently selected absolute-phase emitter is unchanged, including its
MSB wire binding, least-to-most-significant emission, concrete coefficient,
complex amplitudes, inverse recovery and no work/measurement requirements.
Existing kinetic and potential tests retain all applicable small measurement
branches and cleanup for their executable arithmetic consumers. New declaration
and glyph work adds no quantum instructions. The original unknown Rz angle is
never assigned the numerical emitter's coefficient and no root gate parity is
claimed without an original decomposition.

| Path | Measured inventory | Separate source model |
|---|---|---|
| Independent numerical width 3 emitter | 3 phase instructions, zero clean work or measurements | 3 generic Rz rotations; original angle unspecified |
| Independent numerical width 12 emitter | 12 phase instructions, zero clean work or measurements | 12 generic Rz rotations, peak 12 |
| Original width 0 declaration | No executable recipe | Zero rotations, peak 1 |
| Original symbolic n declaration | No executable recipe | n rotations, peak Max(1,n) |

The concentrated regression passes 215 cases in each Cirq lane; an additional
SDK-blocked native declaration case passes. Local integration passes 145.
Strict five-file typing, formatting, lint, both scopes, language, links,
architecture, Program IR and public API checks pass. Original five passes and
three source skips are reused from the immutable source/test assets. The native
render is inspected separately. Receipts use g2-grid-qvr-source-*-20261126.
Q0431 closes with all five original definitions and its DocSpec disposition;
unspecified original angle/body remain explicit limitations, not fabricated code.
